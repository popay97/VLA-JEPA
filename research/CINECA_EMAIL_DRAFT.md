# Draft: CINECA support request (T0)

To: superc@cineca.it
Subject: AIFAC_P02_954 (Leonardo Booster) – requeue permission, $WORK quota, project extension

Dear CINECA support,

I am the PI-side user (vstajkic) of project AIFAC_P02_954 on Leonardo Booster (40,000 local
hours, valid 24 Aug – 24 Oct 2026). We are running a set of ~20 single-node fine-tuning jobs
(4 x A100, 14–26 h each) for a vision-language-action study, and I have three requests:

1. Requeue permission. Our jobs checkpoint on SIGUSR1 (`--signal=B:USR1@900`) and call
   `scontrol requeue` on themselves so a 24 h job continues as a chain. Could `--requeue` /
   `scontrol requeue` be enabled for our account on boost_usr_prod, or, if not possible,
   could you confirm the recommended pattern (dependency chains with `afterany`)?

2. $WORK quota. `/leonardo_work/AIFAC_P02_954` is 1 TB. Weights-only milestones are ~5 GB
   each and we expect ~40 of them plus tensorboard and evaluation videos; an increase to
   2–3 TB (or an equivalent $DRES allocation) would let us keep results without pruning
   mid-campaign.

3. Duration. `sbatch --test-only` currently estimates ~9 days of queue time for a 1-node
   24 h job. With a 24 Oct expiry that leaves roughly four usable submission windows. Would
   an extension of the project end date by 4–6 weeks (same budget) be possible?

Job shapes for reference: boost_usr_prod, 1 node, 4 GPUs, 24 h, --requeue; evaluation jobs
1 GPU, 12 h; dbg QoS for tests.

Thank you,
Vuk Stajkic
vuk.stajkic@invt.tech
