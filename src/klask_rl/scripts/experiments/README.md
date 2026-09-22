# Conditioning diagnostics and baseline learning curves

These entry points save raw CSVs, metadata JSON, and a basic Markdown report. They
do not generate figures automatically. Run from `src/klask_rl` locally or
`/workspace/klask_rl` inside the Isaac Lab container. Paths in the tournament YAML
are relative to that directory. The GPU simulations and training require the
container; offline diagnostics need PyTorch and the repository's Dreamer
dependencies; statistical analysis needs only NumPy.

## Baselines: three scratch seeds on the last two visible GPUs

Run these **sequentially**, inside the container. Each command launches seeds
0/1/2 from scratch, initially two in parallel. Each seed has a 36-hour training
budget, so the three-seed job takes approximately 72 hours plus startup/cleanup.
These are independent training processes, not distributed training of one model.
The launcher uses the last two CUDA devices visible **inside Docker**: e.g.
`cuda:2 cuda:3` if four are visible, `cuda:0 cuda:1` if two are visible. It records
the selected GPU properties. `--devices cuda:2 cuda:3` overrides the selection.
Locks prevent two of these launchers from occupying the same GPUs concurrently.

PPO-B architecture, from scratch:

```bash
nohup /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/baselines.py train --algorithm ppo --output logs/experiments/ppo_3seed --hours 36 > ppo_baselines_launcher.log 2>&1 &
```

After PPO finishes, FastSAC, from scratch:

```bash
nohup /workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/baselines.py train --algorithm fast_sac --output logs/experiments/fastsac_3seed --hours 36 > fastsac_baselines_launcher.log 2>&1 &
```

`tail -f ppo_baselines_launcher.log` shows scheduling; individual training logs
are under `logs/experiments/ppo_3seed/seed_*/run.log`. The same layout applies to
FastSAC. Use a fresh output directory for each experiment. Training does not
automatically resume an interrupted seed: already committed checkpoints remain
usable, but a fresh start must be labelled as a different run.

For a short container integration check first, run the same command in the
foreground with a different output and `--max-env-steps 131072`. It still runs
three seeds, then exits after two PPO rollouts (with defaults). This is a pipeline
check, not the 9M experiment. `--dry-run` prints commands/configs without training
but still validates visible GPU selection. For the long run, omit
`--max-env-steps`; the wall-clock budget is the stopping rule.

### Training recipe and step units

The PPO runner uses `scripts/rl_games/config/klask_ppo_config_with_pretraining.yaml`
as a **recipe**, explicitly replacing `load_checkpoint` with `false` and clearing
`load_path`. It never loads the older PPO model or the reconstructed model's
weights. The architecture is 20 inputs, 256/128/64 hidden units, 2 actions, with
input/value normalization. `--ppo-config` selects another explicit recipe.
Self-play starts from scratch; the historical external checkpoint curriculum is
not enabled. Experiment mode fixes the self-play score observer to count one
actual terminal outcome per completed game, rather than constructing fake
outcomes from zero-valued aggregate logs. This is a documented training behavior
change from the historical observer.

FastSAC uses the existing trainer's hyperparameters, self-play, normalization and
HER setting (currently HER ratio 0). The runner defaults to **512 environments
per seed for both algorithms**, configurable with `--num-envs`. This affects the
update-to-data ratio and must remain fixed across seeds of a reported curve.
FastSAC's existing reset range and reward recipe differ from PPO's; these curves
compare the repository implementations, not an ablation with identical training
rewards and reset distributions. Every resolved training config is saved.

One `environment_steps` unit means one individual environment transition,
summed across all parallel environments. PPO's `frame` is in this unit;
FastSAC's stored `global_step` is a **vector iteration**, so the CSV counter is
`global_step * num_envs`. Do not compare the two raw checkpoint fields directly.
The Dreamer trainer uses its own active-step counter; the supplied Dreamer
checkpoint's `step` is not proof of exactly the same transition accounting.

Checkpoints are published atomically at the first completed update reaching
1M, 3M, 6M, **9M**, 15M, 30M, 60M, 100M, 200M, 400M, 800M, 1.6B, 3.2B, 6.4B,
12.8B and 25.6B transitions, plus the final completed update. CSVs retain the
requested thresholds and **actual** steps. With 512 environments and the PPO
128-step rollout, the first checkpoint reaching 9M is **9,043,968**; FastSAC's is
**9,000,448**. These are approximate matched-step comparisons, not exactly 9M.
No partial PPO optimization rollout is invented to hide this difference.

Each seed saves:

- `training.csv`: environment steps, training iteration, elapsed seconds and
  training reward where available. Rows are correlated training observations.
- `checkpoints.csv`: one row per saved checkpoint, seed, actual steps, requested
  thresholds, path, SHA256, timestamp, elapsed time and final-checkpoint flag.
- `metadata.json`: initialization, seed, resolved configuration, arguments,
  step definition, timestamps, counter and completion status.
- `checkpoints/step_*.pth` (PPO) or `step_*.pt` (FastSAC).
- `trainer/`: the existing trainer's config/TensorBoard artifacts.

The launcher also saves `metadata.json` with all commands/GPU assignments/source
hashes and a basic `report.md`. Training reward is **not** evaluation score.
Interruption preserves previously published checkpoints; an interrupted optimizer
update is not saved under the last completed update's step count.

### Score saved checkpoints against a fixed opponent

The opponent has intentionally **not been chosen for you**. Supply its checkpoint
and config later. Use the same frozen opponent for every seed/checkpoint of a
curve. For a Dreamer opponent its opponent-action input is fixed at zero.
For example, after filling in the two paths:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/baselines.py evaluate \
  --training logs/experiments/ppo_3seed \
  --output logs/experiments/ppo_curve \
  --opponent-kind dreamer \
  --opponent-checkpoint /absolute/path/to/chosen_opponent.pt \
  --opponent-config /absolute/path/to/its/.hydra/config.yaml \
  --games 200
```

Change the training/output directories for FastSAC. `--opponent-kind ppo` plus
its agent YAML or `--opponent-kind fast_sac` (no config required) is also supported.
Use the reconstructed PPO-B if choosing the supplied PPO opponent. The common
evaluation environment still comes from the D-RSSM config in
`scripts/tournament/config.yaml`; it uses the same 30-second, balanced-seat
protocol as the existing tournament.

`--requested-steps 9000000` evaluates just the first checkpoint reaching the 9M
threshold. Omit it for the full saved curve. `--games` is the total per
checkpoint, split between the two seats. Both GPUs evaluate legs concurrently.
Completed evaluation legs can be reused by rerunning the same evaluation
command; incompatible manifests are rejected. Partial legs are rerun from their
seed. The training checkpoint and fixed-opponent checkpoint hashes are checked.

Output includes `evaluations.csv`, one tournament directory per seed/checkpoint,
per-game `games.csv`, per-leg `metadata.json`, resolved configs, and
`analysis/report.md`. Learning-curve intervals resample **seed scores**; per-seed
gameplay intervals resample individual games. Three seeds give limited precision.

## Conditioning puzzle: experiments 5–7

The focal model is the supplied `latest.pt` in `scripts/tournament/config.yaml`.
PPO-B is the reconstructed export specified there. No stock-Dreamer checkpoint
is inferred, and no model is retrained for these diagnostics.

The in-distribution opponent remains an explicit input. Pass its saved composed
config and checkpoint and document its relationship to the training population
with `--id-provenance`. A checkpoint described only as Dreamer-lineage is an
**ID proxy**, not proof that it was in the focal model's training opponent pool.
The argument accepts either a separated or a stock Dreamer checkpoint with a
compatible image policy/config. Missing or incompatible models fail preflight.

### 5–6. Collect held-out trajectories once, then evaluate offline

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/puzzle.py collect \
  --id-checkpoint /absolute/path/to/lineage_checkpoint.pt \
  --id-config /absolute/path/to/its/.hydra/config.yaml \
  --id-provenance 'Describe training-pool membership here; say proxy if unverified' \
  --games 64 --num-envs 8 --seed 10000 \
  --output logs/experiments/conditioning_heldout
```

This collects 64 games per opponent domain (ID lineage and OOD reconstructed
PPO-B), balanced across seats. Collection uses the focal D-RSSM's exact input and
keeps the lineage opponent's own input fixed at zero. These are new simulation
trajectories with no model updates. They must not subsequently be used for
training or checkpoint selection if they are to remain held out.

Each leg writes `trajectories/observations.csv`, one row per logged frame, with
game/environment ID, domain, seat, time index and both agents' normalized
commands in their **own policy frames**. Image pixels are stored losslessly as
uint8 in compressed per-game NPZ shards; the CSV provides the shard/image index.
Images are not expanded into millions of CSV columns. Frames/commands are captured
before stepping, before wrapper mutation. Only completed quota-accepted games
are committed. A terminal transition's auto-reset image is never used as a
prediction target; the last command in a shard has no successor image.

Run the frozen model over those same trajectories:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/conditioning.py \
  --dataset logs/experiments/conditioning_heldout \
  --output logs/experiments/conditioning_diagnostics \
  --device cuda:0 --horizons 15 --burn-in 32 --stride 15 --draws 1
```

`conditioning.csv` contains one row per game/origin/condition/timing/latent draw/
horizon, with image loss, per-pixel image loss, posterior reconstruction loss,
raw KL, target index, seed and checkpoint. `metadata.json` specifies definitions,
configs, checkpoint and trajectory hashes, timestamps and collection step count.
`--checkpoint` can locate a copied checkpoint on another machine, but its hash
must match the collection manifest. No simulator is needed for this stage.

Definitions and controls:

- Conditions are exact logged opponent commands, independent Uniform[-1,1]
  random commands per component, and zero commands. Own commands and target
  images are always the same logged values.
- After a 32-step burn-in, each open-loop rollout starts from a shared posterior
  filtered with the exact history for that timing convention. Only the rollout's
  opponent commands are intervened on. This estimates a conditional prediction
  gap, not a separate steady-state filter under a permanently zeroed history.
- `image_nll = -decoder.log_prob(image/255)`. The supplied decoder is `MSEDist`,
  so this is **sum squared pixel error**, the repository's image likelihood loss,
  not a normalized/calibrated probabilistic NLL. Per-pixel loss is also saved.
- The primary image loss decodes the open-loop prior. Target images are used
  only to score it and construct a diagnostic target posterior; they are never
  fed back into the rollout. Posterior reconstruction loss is saved separately.
- KL is `KL(q(z | imagined h, target image) || p(z | imagined h))`, summed over
  stochastic variables, in nats, using the repository's raw-logit training
  definition without free-nat clipping or loss weights. It is not a clipped
  training objective or a KL between images.
- The `k=1` rows supply experiment 5. `k=1..15` supply experiment 6. The control
  timestep is recorded (20 ms for this config). Do not conflate physics steps
  (1 ms) and control steps.
- Posterior/prior latent draws use paired RNG seeds across conditions; distinct
  logged games use distinct derived seeds. Increase `--draws` for Monte Carlo
  precision. Repeated draws, origins and horizons are dependent measurements.
- By default both timing conventions are evaluated: `aligned` uses `(a[t], o[t])`
  to predict image `t+1`; `legacy` uses `(a[t], o[t-1])`, with zero at episode
  start. The repository's historical action/replay handling motivates this
  additional diagnostic. `--timings aligned` selects only physical alignment.

### 7. Gameplay against the ID opponent

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/experiments/puzzle.py gameplay \
  --id-checkpoint /absolute/path/to/lineage_checkpoint.pt \
  --id-config /absolute/path/to/its/.hydra/config.yaml \
  --id-provenance 'Describe training-pool membership here; say proxy if unverified' \
  --games 200 --num-envs 32 --seed 20000 \
  --output logs/experiments/id_gameplay
```

This plays 200 games **per condition** (600 total), with both seats. Only the
focal D-RSSM changes exact/random/zero input. The opponent's input stays zero.
Random inputs use a dedicated seeded generator, so generating the intervention
does not consume the shared simulator/model RNG. `--timing legacy` provides an
explicit alternative to the default aligned convention; it does not reinstate
the old terminal/reset bugs. Results are saved per game and analysed automatically.

## Change analysis without rerunning experiments

```bash
python scripts/experiments/analyze.py \
  --input logs/experiments/conditioning_diagnostics \
  --output logs/experiments/conditioning_analysis
```

Use a tournament or baseline evaluation directory as `--input` for gameplay or
learning curves. This reads CSVs and writes summary CSVs, an analysis metadata
JSON and `report.md`. No figures are generated. `--plots` explicitly requests
basic diagnostic PNGs and requires Matplotlib; it does not change raw data.
`--bootstrap-samples` and `--seed` change the statistical analysis only.

Gameplay statistics include `N=W+L+D`, score `(W+0.5D)/N`, decisive ratio `W/L`,
and a 95% percentile bootstrap interval over individual game payoffs 1/0.5/0.
Ratio is `infinity` if `L=0<W` and `undefined` if `W=L=0`.
Conditioning gaps are **corrupted minus exact**. Analysis first pairs matching
origins/draws, averages within each game, then bootstraps game means with equal
weight per game. It does not treat 15 horizons or thousands of frames from one
game as independent games. Intervals are pointwise, not simultaneous across
horizons. Raw metrics and absolute losses remain available alongside gaps.

The basic report lists incomplete game sets and the metadata files. Evidence
for a near-zero OOD gap requires a specified equivalence margin, not merely a
nonsignificant difference. These experiments test inference sensitivity and
timing. They **cannot by themselves establish that opponent conditioning is
necessary for training stability**; that requires an additional training ablation.

## Data transfer and repository files

Keep the whole experiment directory, including CSVs, JSON/YAML configs and
trajectory shards (if collecting images). `games.npz` is retained for compatibility
with the original tournament reporter; `games.csv` is the public analysis input.
Container `/workspace/klask_rl/logs/experiments` normally corresponds to host
`/home/student/klask_rl_repo/src/klask_rl/logs/experiments` under the existing bind
mount. Check the mount before assuming a separate `docker cp` is needed.
Baseline `evaluations.csv` uses relative evaluation directories so analysis also
works after copying the output directory locally. Stored checkpoint paths record
their original provenance; statistical analysis does not reload models.

Changes are in `scripts/experiments/`, the tournament exporter/adapters, and the
two existing baseline trainers plus PPO observer. The pre-existing local edits
to `scripts/dreamer/train_dreamer.py` are unrelated to this implementation.

CPU tests:

```bash
python -m pytest -q scripts/experiments/tests scripts/tournament/tests
```
