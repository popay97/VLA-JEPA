"""End-to-end CPU smoke test of the VLA_JEPA framework with a fake VLM, fake action head and
fake V-JEPA2 encoder: token layout, loss dict contract, bottleneck / Anchor-Align wiring,
video-only batches, per-frame encoder, diagnostics terms, and milestone export prefixes."""
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import starVLA.model.framework.VLA_JEPA as vj_mod
from starVLA.model.modules.world_model.target_encoders import TargetEncoder
from tests.test_target_encoders import FakeProcessor, FakeVJEPA2

H = 32
SPECIAL = re.compile(r"<\|[^|<>]+\|>")


class FakeTokenizer:
    def __init__(self):
        words = ["your", "task", "is", ".", "infer", "the", "temporal", "dynamics", "from", "frames", "and",
                 "produce", "corresponding", "policy", "actions", "pick", "up", "bowl", "<|image_pad|>"] + list(
            vj_mod.__dict__.get("DIRECTION_WORDS", ("forward", "backward", "left", "right", "up", "down")))
        self.vocab = {}
        for w in words:
            self.vocab.setdefault(w, len(self.vocab))
        self.padding_side = "left"

    def get_vocab(self):
        return dict(self.vocab)

    def add_tokens(self, toks, special_tokens=True):
        n = 0
        for t in toks:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
                n += 1
        return n

    def convert_tokens_to_ids(self, t):
        return self.vocab[t]

    def __len__(self):
        return len(self.vocab)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(w, len(self.vocab)) for w in text.lower().split()]}

    def encode_prompt(self, prompt):
        ids = []
        pos = 0
        for m in SPECIAL.finditer(prompt):
            ids += [self.vocab.setdefault(w, len(self.vocab)) for w in re.findall(r"[a-z]+|\.", prompt[pos : m.start()].lower())]
            ids.append(self.vocab.setdefault(m.group(0), len(self.vocab)))
            pos = m.end()
        ids += [self.vocab.setdefault(w, len(self.vocab)) for w in re.findall(r"[a-z]+|\.", prompt[pos:].lower())]
        return ids


class FakeLM(nn.Module):
    def __init__(self, tok, n_layers=2):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=H)
        self.embed = nn.Embedding(len(tok) + 64, H)
        self.layers = nn.ModuleList([nn.Linear(H, H) for _ in range(n_layers)])
        self.lm_head = nn.Linear(H, len(tok) + 64, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def resize_token_embeddings(self, n):
        pass

    def forward(self, input_ids=None, attention_mask=None, output_hidden_states=True, **kw):
        h = self.embed(input_ids)
        hs = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h))
            hs.append(h)
        return SimpleNamespace(hidden_states=tuple(hs), logits=self.lm_head(h))


class FakeVLMInterface(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        tok = FakeTokenizer()
        self.processor = SimpleNamespace(tokenizer=tok)
        self.model = FakeLM(tok)

    def forward(self, **kw):
        return self.model(**kw)

    def build_qwenvl_inputs(self, images, instructions, prompt_replace_dict=None, prompt_template=None, **kw):
        rows = []
        for imgs, ins in zip(images, instructions):
            prompt = (prompt_template or "{instruction}").replace("{instruction}", ins)
            for k, v in (prompt_replace_dict or {}).items():
                prompt = prompt.replace(k, v)
            ids = [self.processor.tokenizer.vocab["<|image_pad|>"]] * (2 * len(imgs)) + self.processor.tokenizer.encode_prompt(prompt)
            rows.append(ids)
        L = max(len(r) for r in rows)
        input_ids = torch.zeros(len(rows), L, dtype=torch.long)
        attn = torch.zeros(len(rows), L, dtype=torch.long)
        for i, r in enumerate(rows):  # left padding like the Qwen processor
            input_ids[i, L - len(r) :] = torch.tensor(r)
            attn[i, L - len(r) :] = 1
        return {"input_ids": input_ids, "attention_mask": attn}


class FakeActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        am = config.framework.action_model
        self.horizon = am.future_action_window_size + 1
        self.action_dim = am.action_dim
        self.proj = nn.Linear(H, self.action_dim)

    def forward(self, embodied, actions, state=None):
        pred = self.proj(embodied.float()).mean(1, keepdim=True).expand(-1, actions.shape[1], -1)
        return ((pred - actions.float()) ** 2).mean()

    def predict_action(self, embodied, state=None):
        return self.proj(embodied.float()).mean(1, keepdim=True).expand(-1, self.horizon, -1)


def make_cfg(**over):
    cfg = OmegaConf.create(
        {
            "framework": {
                "name": "VLA_JEPA",
                "qwenvl": {"base_vlm": "fake-Qwen3-VL"},
                "action_model": {"action_horizon": 7, "future_action_window_size": 6, "past_action_window_size": 0,
                                 "action_dim": 7, "state_dim": 8, "diffusion_model_cfg": {"cross_attention_dim": 0}},
                "vj2_model": {"base_encoder": "fake", "encoder_type": "vjepa2_clip", "num_frames": 8, "depth": 1,
                              "num_heads": 4, "special_action_token": "<|action_{}|>", "num_action_tokens_per_timestep": 2,
                              "embodied_action_token": "<|embodied_action|>", "num_embodied_action_tokens_per_instruction": 3,
                              "wm_loss_weight": 0.1},
                "latent_action": {"bottleneck": "none"},
                "anchor_align": {"enable_anchor": False, "enable_align": False},
            },
            "datasets": {
                "vla_data": {"CoT_prompt": "Your task is {instruction}. Infer the temporal dynamics from frames {actions} and produce the corresponding policy actions {e_actions}."},
                "video_data": {"CoT_prompt": "Infer the temporal dynamics from frames {actions}."},
            },
            "trainer": {"repeated_diffusion_steps": 2},
        }
    )
    return OmegaConf.merge(cfg, OmegaConf.create(over))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(vj_mod, "get_vlm_model", lambda config: FakeVLMInterface(config))
    monkeypatch.setattr(vj_mod, "get_action_model", lambda config: FakeActionHead(config))
    monkeypatch.setattr(vj_mod, "build_target_encoder", lambda vj_cfg: TargetEncoder(vj_cfg, model=FakeVJEPA2(img=32, patch=16, D=6), processor=FakeProcessor()))


def make_batch(B=2, T=8, with_action=True, with_state=True):
    rng = np.random.default_rng(0)
    ex = []
    for _ in range(B):
        e = {"image": ["img_a", "img_b"], "lang": "pick up the bowl", "video": rng.integers(0, 255, size=(2, T, 32, 32, 3), dtype=np.uint8)}
        if with_action:
            e["action"] = rng.normal(size=(7, 7)).astype(np.float16)
        if with_state:
            e["state"] = rng.normal(size=(1, 8)).astype(np.float16)
        ex.append(e)
    return ex


def test_baseline_forward_matches_upstream_contract(patched):
    m = vj_mod.VLA_JEPA(make_cfg())
    assert m.num_states == 4 and m.num_transitions == 3
    assert m.replace_prompt == "<|action_0|>" * 2 + "<|action_1|>" * 2 + "<|action_2|>" * 2
    assert m.latent_bottleneck.kind == "none" and m.anchor_align is None
    assert not any(p.requires_grad for p in m.vj_encoder.parameters())
    out = m(make_batch())
    assert set(out) == {"action_loss", "wm_loss"}
    total = sum(out.values())
    total.backward()
    assert torch.isfinite(total)
    assert m.vj_predictor.action_encoder.weight.grad is not None
    m.train()
    assert not m.vj_encoder.training  # frozen encoder stays in eval under train()


def test_video_only_batch_uses_video_weight(patched):
    m = vj_mod.VLA_JEPA(make_cfg(framework={"vj2_model": {"wm_loss_weight": 0.1, "video_wm_loss_weight": 1.0}}))
    out = m(make_batch(with_action=False, with_state=False))
    assert set(out) == {"wm_loss"}


def test_bottleneck_anchor_align_wiring(patched):
    cfg = make_cfg(
        framework={
            "latent_action": {"bottleneck": "vib", "bottleneck_dim": 4},
            "anchor_align": {"enable_anchor": True, "anchor_weight": 0.1, "enable_align": True, "align_weight": 0.02,
                             "teacher_source": "pretrained_checkpoint"},
        }
    )
    m = vj_mod.VLA_JEPA(cfg)
    assert m.anchor_align is not None and m.anchor_align.anchor_teacher is not None
    assert not any(p.requires_grad for p in m.anchor_align.anchor_teacher.parameters())
    assert m.milestone_exclude_prefixes() == ["anchor_align.anchor_teacher."]
    assert set(m.NEW_MODULE_PREFIXES) == {"latent_bottleneck.", "anchor_align."}
    # perturb the student, then the hook must copy student -> teacher
    with torch.no_grad():
        m.qwen_vl_interface.model.layers[0].weight.add_(1.0)
    m.on_pretrained_loaded()
    assert torch.equal(m.anchor_align.anchor_teacher.model.layers[0].weight, m.qwen_vl_interface.model.layers[0].weight)

    m.train()
    out = m(make_batch(B=3))
    losses = {k: v for k, v in out.items() if not k.startswith("metric/")}
    metrics = {k: v for k, v in out.items() if k.startswith("metric/")}
    assert set(losses) == {"action_loss", "wm_loss", "vib_kl_loss", "anchor_loss", "align_loss"}
    assert {"metric/vib_kl_raw", "metric/anchor_raw", "metric/align_acc", "metric/align_valid_frac"} <= set(metrics)
    total = sum(losses.values())
    total.backward()
    assert torch.isfinite(total)
    assert m.anchor_align.align_dir_proj.weight.grad is not None
    assert m.latent_bottleneck.down.weight.grad is not None
    assert all(p.grad is None for p in m.anchor_align.anchor_teacher.parameters())
    sd_keys = m.state_dict().keys()
    assert any(k.startswith("anchor_align.anchor_teacher.") for k in sd_keys)


def test_anchor_matches_teacher_at_init_only_on_kept_positions(patched):
    cfg = make_cfg(framework={"anchor_align": {"enable_anchor": True, "anchor_layers": "all"}})
    m = vj_mod.VLA_JEPA(cfg)
    out = m(make_batch())
    assert out["anchor_loss"].item() < 1e-10  # teacher is a copy of the student at init


def test_perframe_encoder_changes_token_layout(patched):
    cfg = make_cfg(framework={"vj2_model": {"encoder_type": "vjepa2_perframe", "normalize_targets": True, "state_stride": 1}})
    m = vj_mod.VLA_JEPA(cfg)
    assert m.num_states == 8 and m.num_transitions == 7
    assert m.replace_prompt.count("<|action_6|>") == 2 and "<|action_7|>" not in m.replace_prompt
    out = m(make_batch())
    assert torch.isfinite(out["wm_loss"])
    terms = m.world_model_terms(make_batch())
    tok = m.target_encoder.tokens_per_state
    assert terms["z"].shape == (2, 14, H)
    assert terms["input_states"].shape == (2, 7 * tok, 12) and terms["gt_states"].shape == (2, 7 * tok, 12)
    assert terms["actions_target"].shape == (2, 7, 7)
    assert terms["pre_action_pos"].shape == (2,)


def test_drop_bottleneck_zeroes_z_and_predict_action_runs(patched):
    m = vj_mod.VLA_JEPA(make_cfg(framework={"latent_action": {"bottleneck": "drop"}}))
    terms = m.world_model_terms(make_batch())
    assert terms["z"].abs().sum() == 0 and terms["z_raw"].abs().sum() > 0
    out = m.predict_action([["a", "b"]], ["pick up the bowl"], state=np.zeros((1, 1, 8), dtype=np.float32))
    assert out["normalized_actions"].shape == (1, 7, 7)
