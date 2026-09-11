"""
Resumable checkpointing for the starVLA trainers.

Upstream `_save_checkpoint` writes model weights only, so a Slurm requeue restarts warmup
and loses AdamW moments. This mixin adds:

  * full training state via `accelerator.save_state` (DeepSpeed engine = model + sharded
    optimizer, registered LR scheduler, RNG, stateful dataloaders) under
    `<run>/checkpoints/state/step_<N>/` with a `meta.json` (completed_steps, epoch counters,
    dataloader epoch indices) and a `latest` pointer, rotated to `keep_last_states`;
  * auto-resume from `latest` (or `trainer.resume_from_checkpoint`) at start-up;
  * a save trigger from any of: wall-clock interval (`trainer.save_every_minutes`), a
    `STOP_AND_SAVE` sentinel file in the checkpoint dir (touched by the sbatch USR1 trap), or
    SIGUSR1 / SIGTERM delivered to the process. The decision is taken on rank 0 and
    broadcast so all ranks save together;
  * weights-only milestone export (bf16-friendly `.pt`, training-only modules such as the
    Anchor teacher removed) compatible with `baseframework.from_pretrained`.

Constraints: DeepSpeed ZeRO-2 shards optimizer state per rank, so resume requires the same
number of processes. Exact dataloader position needs
`accelerator.dataloader_config.use_stateful_dataloader = True` (torchdata >= 0.8); without
it the epoch index is restored but the epoch restarts from its first batch.
"""
import glob
import json
import os
import shutil
import signal
import time
from typing import List, Optional, Sequence

import torch
import torch.distributed as dist


def _cfg_get(node, key, default=None):
    if node is None:
        return default
    if hasattr(node, "get"):
        try:
            v = node.get(key, default)
            return default if v is None else v
        except Exception:
            pass
    return getattr(node, key, default)


def filter_state_dict(state_dict: dict, exclude_prefixes: Sequence[str]) -> dict:
    if not exclude_prefixes:
        return state_dict
    return {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in exclude_prefixes)}


def _step_of(path: str) -> int:
    try:
        return int(os.path.basename(path).split("_")[-1])
    except ValueError:
        return -1


class ResumableCheckpointing:
    """Mixin. Host class must provide: config, model, optimizer, lr_scheduler, accelerator,
    completed_steps, checkpoint_dir (set by _init_checkpointing)."""

    def init_resumable(self, dataloaders: Sequence, epoch_counter_names: Sequence[str]) -> None:
        tcfg = self.config.trainer
        self.save_every_minutes = float(_cfg_get(tcfg, "save_every_minutes", 0.0) or 0.0)
        self.keep_last_states = int(_cfg_get(tcfg, "keep_last_states", 3))
        self.auto_resume = bool(_cfg_get(tcfg, "auto_resume", True))
        self.resume_from = _cfg_get(tcfg, "resume_from_checkpoint", None)
        self._resume_dataloaders = list(dataloaders)
        self._epoch_counter_names = list(epoch_counter_names)
        self._stop_requested = False
        self._last_save_time = time.time()

        self.state_root = os.path.join(self.checkpoint_dir, "state")
        self.latest_pointer = os.path.join(self.state_root, "latest")
        self.sentinel_path = os.path.join(self.checkpoint_dir, "STOP_AND_SAVE")
        self.done_path = os.path.join(self.config.output_dir, "DONE")

        unwrapped = self.accelerator.unwrap_model(self.model)
        fn = getattr(unwrapped, "milestone_exclude_prefixes", None)
        self.milestone_exclude_prefixes: List[str] = list(fn()) if callable(fn) else []

        # the LR scheduler is not passed through accelerator.prepare (DeepSpeed would step it
        # itself), so register it for save_state / load_state explicitly
        self.accelerator.register_for_checkpointing(self.lr_scheduler)

        self._install_signal_handlers()
        if self.accelerator.is_main_process:
            os.makedirs(self.state_root, exist_ok=True)
            if os.path.exists(self.sentinel_path):
                os.remove(self.sentinel_path)

    # ------------------------------------------------------------------ signals / triggers
    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            self._stop_requested = True

        for name in ("SIGUSR1", "SIGTERM"):
            sig = getattr(signal, name, None)  # SIGUSR1 does not exist on Windows
            if sig is None:
                continue
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass  # not the main thread

    def check_stop_and_save(self):
        """Returns (save_now, stop_now); identical on every rank."""
        save_now, stop_now = False, False
        if self.accelerator.is_main_process:
            if self._stop_requested or os.path.exists(self.sentinel_path):
                save_now, stop_now = True, True
            elif self.save_every_minutes > 0 and (time.time() - self._last_save_time) >= 60.0 * self.save_every_minutes:
                save_now = True
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flag = torch.tensor([int(save_now), int(stop_now)], device=self.accelerator.device)
            dist.broadcast(flag, src=0)
            save_now, stop_now = bool(flag[0].item()), bool(flag[1].item())
        return save_now, stop_now

    # ------------------------------------------------------------------ resume
    def _resolve_resume_path(self) -> Optional[str]:
        if self.resume_from:
            p = str(self.resume_from)
            if os.path.isdir(p) and os.path.exists(os.path.join(p, "meta.json")):
                return p
            raise FileNotFoundError(f"trainer.resume_from_checkpoint={p} is not a saved training state")
        if self.auto_resume and os.path.exists(self.latest_pointer):
            with open(self.latest_pointer) as f:
                name = f.read().strip()
            p = os.path.join(self.state_root, name)
            if os.path.exists(os.path.join(p, "meta.json")):
                return p
        return None

    def maybe_resume(self) -> bool:
        path = self._resolve_resume_path()
        if path is None:
            return False
        self.accelerator.print(f"[resume] loading training state from {path}")
        self.accelerator.load_state(path)
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)
        self.completed_steps = int(meta["completed_steps"])
        for name in self._epoch_counter_names:
            if name in meta.get("epoch_counters", {}):
                setattr(self, name, int(meta["epoch_counters"][name]))
        for dl, it in zip(self._resume_dataloaders, meta.get("dataloader_iterations", [])):
            if hasattr(dl, "iteration") and it is not None:
                dl.iteration = int(it)
        self._last_save_time = time.time()
        self.accelerator.print(
            f"[resume] completed_steps={self.completed_steps} lr={self.lr_scheduler.get_last_lr()[0]:.3e} "
            f"epochs={ {n: getattr(self, n, None) for n in self._epoch_counter_names} }"
        )
        return True

    # ------------------------------------------------------------------ save
    def save_training_state(self, reason: str = "interval") -> str:
        name = f"step_{self.completed_steps}"
        out = os.path.join(self.state_root, name)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(out)
        if self.accelerator.is_main_process:
            meta = {
                "completed_steps": int(self.completed_steps),
                "epoch_counters": {n: int(getattr(self, n, 0) or 0) for n in self._epoch_counter_names},
                "dataloader_iterations": [getattr(dl, "iteration", None) for dl in self._resume_dataloaders],
                "reason": reason,
                "time": time.time(),
                "world_size": int(self.accelerator.num_processes),
            }
            with open(os.path.join(out, "meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            tmp = self.latest_pointer + ".tmp"
            with open(tmp, "w") as f:
                f.write(name)
            os.replace(tmp, self.latest_pointer)
            self._rotate_states()
            if os.path.exists(self.sentinel_path):
                os.remove(self.sentinel_path)
            self.accelerator.print(f"[checkpoint] training state saved to {out} ({reason})")
        self._last_save_time = time.time()
        self.accelerator.wait_for_everyone()
        return out

    def _rotate_states(self) -> None:
        dirs = [d for d in glob.glob(os.path.join(self.state_root, "step_*")) if os.path.isdir(d)]
        dirs = sorted(dirs, key=_step_of)
        for d in dirs[: max(0, len(dirs) - self.keep_last_states)]:
            shutil.rmtree(d, ignore_errors=True)

    def export_weights(self, tag: Optional[str] = None, subdir: Optional[str] = None) -> Optional[str]:
        """Weights-only milestone. Layout `<run>/checkpoints/steps_<N>_pytorch_model.pt`
        (upstream naming, readable by `baseframework.from_pretrained`)."""
        state_dict = self.accelerator.get_state_dict(self.model)  # gathers ZeRO shards; all ranks call
        path = None
        if self.accelerator.is_main_process:
            state_dict = filter_state_dict(state_dict, self.milestone_exclude_prefixes)
            if subdir:
                d = os.path.join(self.config.output_dir, subdir)
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, "pytorch_model.pt")
            else:
                tag = tag or f"steps_{self.completed_steps}"
                path = os.path.join(self.checkpoint_dir, f"{tag}_pytorch_model.pt")
            torch.save(state_dict, path)
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps({"steps": int(self.completed_steps), "milestone": path}) + "\n")
            self.accelerator.print(f"[checkpoint] weights exported to {path}")
        self.accelerator.wait_for_everyone()
        return path

    def mark_done(self) -> None:
        if self.accelerator.is_main_process:
            with open(self.done_path, "w") as f:
                f.write(json.dumps({"completed_steps": int(self.completed_steps), "time": time.time()}))
