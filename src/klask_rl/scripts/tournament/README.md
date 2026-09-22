# Three-agent tournament

This runner evaluates the supplied **reconstructed PPO-B**, **D-RSSM**, and
**FastSAC** checkpoints. Stock DreamerV3 is excluded because no checkpoint was
provided. Nothing is trained, and no historical PPO checkpoint is loaded.

## Plan and implementation

1. Identify the supplied weights and their saved training configurations.
2. Reuse the repository's environment, action wrappers, observation layouts and
   checkpoint loaders for both board seats.
3. Run D-RSSM/PPO-B with exact and zeroed opponent commands, then complete the
   three-agent round robin with zeroed D-RSSM input.
4. Save individual games, apply mutually exclusive outcome accounting, and
   bootstrap the game payoffs. Validate the accounting and recurrent timing on CPU.

The shared builders are in [`../dreamer/evaluation.py`](../dreamer/evaluation.py)
and [`../dreamer/checkpoint_loaders.py`](../dreamer/checkpoint_loaders.py).
The original Dreamer evaluator also uses these builders. Its CLI is preserved.

## Run inside the SSH machine's Docker container

The repository's Docker helper mounts host `src/klask_rl` at
`/workspace/klask_rl`; it separately mounts both `third_party` submodules under
`scripts/dreamer/r2dreamer` and `scripts/fast_sac/klask_her`.
Initialize the submodules on the host as described in the root README. The
checkpoint files, `.hydra/config.yaml`, PPO `params/agent.yaml`, and actuator
checkpoint must also be present in the mounted project; logs are not fetched by Git.

```bash
cd /workspace/klask_rl

# Inspect every required artifact and write preflight.json; exit 1 on blockers.
/workspace/isaaclab/_isaac_sim/python.sh scripts/tournament/run_tournament.py \
  --preflight-only --output logs/tournament/preflight

# First exercise all eight legs with a small simulation run.
/workspace/isaaclab/_isaac_sim/python.sh scripts/tournament/run_tournament.py \
  --games 32 --num-envs 8 --output logs/tournament/smoke

# Full experiment: 40,000 games across four matchup/condition pairs.
/workspace/isaaclab/_isaac_sim/python.sh scripts/tournament/run_tournament.py \
  --output logs/tournament/three_agents_10k
```

All paths in [`config.yaml`](config.yaml) are relative to the project containing
`scripts` and `logs`, so the same config works locally and in Docker. Run the
script by its absolute path from any directory, or override `--project-root`.
Absolute `/workspace/klask_rl/...` paths in saved configs are remapped to this
root; unrelated absolute paths are never guessed. If the exact actuator file
is elsewhere, supply `--actuator-checkpoint /actual/path/to/the/same/model.pt`.

Useful overrides: `--devices cuda:0 cuda:1`, `--num-envs 128`, `--games 10000`,
`--seed 0`, and `--episode-length-s 30`. GPU indices are those **inside** the
container. **The default is two GPUs: `cuda:0` and `cuda:1`.** The launcher runs
one independent leg per GPU, with at most two workers alive at a time. Each
worker keeps its simulation and policies on its assigned device; this does not
split a single policy across GPUs. Training-time multi-GPU model replicas and
multi-GPU rendering are disabled. Preflight checks both CUDA devices exist.
For a single GPU, pass `--devices cuda:0` (or the shorthand `--device cuda:0`).
`--num-envs` is the number of environments **per worker**, so running two workers
uses twice that many environments overall. CPU sprite rendering and memory
bandwidth may limit speedup.

To evaluate only D-RSSM vs PPO-B with the default `latest.pt`, run:

```bash
/workspace/isaaclab/_isaac_sim/python.sh scripts/tournament/run_tournament.py \
  --config scripts/tournament/config.yaml \
  --matches drssm_ppo_exact drssm_ppo_zero \
  --games 200 --num-envs 8 \
  --output logs/tournament/drssm_ppo_latest_200
```

This runs 200 games **per condition** (400 total), with 100 games per board
seat. Use only `--matches drssm_ppo_zero` for a 200-game primary-only run.
Unselected matchups are not run and do not appear as missing in the report;
this selection does not require the FastSAC checkpoint or submodule.

Compilation is disabled for evaluation. Progress is written to each leg's
`run.log`, whose path the launcher prints (use `tail -f` in another terminal).

To resume, repeat the exact full-run command with `--resume`. Checkpoint,
config, source, GPU assignments and package fingerprints must still match. Complete legs are
skipped; partial legs restart from their original seed and replace that leg's
partial result, so games are never appended twice. Ordinary interruptions and
exceptions save completed games; an uncatchable process kill can lose games
since the last periodic save (every approximately 1,000 accepted games).
GPU assignments are saved in the manifest before launching; completion order
and skipped completed legs do not change them. Only the parent writes summary
reports, so parallel workers cannot overwrite each other's reports. On failure
or Ctrl-C, no further jobs start and active workers get a shared 30-second grace
period to save and exit before they are killed. Use a fresh output directory
when switching an older run to this implementation, and finish or stop existing
evaluations before pulling code changes into their mounted directory.

## Protocol

| Matchup | D-RSSM input | Games | Use |
|---|---|---:|---|
| D-RSSM vs PPO-B | Exact | 10,000 | Diagnostic ablation |
| D-RSSM vs PPO-B | Zero | 10,000 | **Primary/deployable** and round robin |
| D-RSSM vs FastSAC | Zero | 10,000 | Primary round robin |
| FastSAC vs PPO-B | N/A | 10,000 | Primary round robin |

Each pair is split into two legs of 5,000 games, swapping the physical board
seats. Odd requested counts put the extra game in the first leg. Each parallel
environment receives a predetermined episode quota; completion speed does not
decide which environments contribute the last games. A "game" is one repository
episode, ending on a goal, a peg falling into a goal, or timeout, **not** a
multi-point match to six.

The common environment comes from the saved D-RSSM sprite config, with 30-second
episodes, symmetric ball resets `x=[-0.15,0.15], y=[-0.1,0.1]`, all five scoring
terminations enabled, no observation noise and no domain randomization. The
saved `sim_dt=0.001`, decimation 20, max velocity 0.4 m/s, max acceleration
10 m/s², collision avoidance and actuator model are retained. Both sides use
the same dynamics. FastSAC's training config instead used
`ball_reset_y=[-0.1,-0.04]`; its asymmetric training reset area is deliberately
not used as the tournament protocol. The duration is an explicit experimental
choice matching the existing Dreamer README's evaluation example; the supplied
training configs used 10 seconds. Change the CLI override for a separate run.

PPO-B reads the existing 20-dimensional state group; FastSAC reads its own
18-dimensional observation group; D-RSSM reads the 128×128 sprite image. The
opponent groups already rotate the board and swap peg roles. The original
wrappers scale normalized commands to m/s, apply acceleration limits and the
actuator model, and convert the opponent command to world coordinates.

All actors use their evaluation mode/mean action. Dreamer's RSSM still samples
its latent posterior, as implemented by the repository. Each leg is seeded
after model construction (rl_games resets the seed while loading). The exact
and zero runs use identical checkpoint hashes and corresponding leg seeds;
they are separate simulations, not paired identical game trajectories.

## Opponent separation and opponent-action timing

`train_dreamer.py` parses top-level `opponent_separation.enabled` (or its legacy
boolean form) and writes it into `config.model.opponent_separation` before
constructing the model. `r2dreamer/dreamer.py` doubles the RSSM action width
from 2 to 4 when enabled. The saved D-RSSM YAML contains
`model.opponent_separation: false` as an unmodified default **and** top-level
`opponent_separation.enabled: true`; the top-level setting is authoritative,
as in training. The supplied checkpoint independently confirms a `(512,4)`
RSSM action-input weight. It also uses `rep_loss: dreamer`.

A stock agent must be trained with separation disabled (a 2D RSSM input).
Toggling separation off when loading the supplied 4D checkpoint does not create
Stock DreamerV3 and is rejected. Its training recipe and weights cannot be
uniquely identified because no stock checkpoint was supplied.

[`policies.py:DreamerPolicy.action`](policies.py) controls exact versus zero
input without changing the trained model or the simulator's real actions:

- Given observation `x[t+1]`, the RSSM receives the preceding transition's own
  command `a[t]` and the other policy's actual normalized command `o[t]`.
- In `zero` mode, only the RSSM's `o[t]` slot is replaced with zeros.
- Commands are in each policy's egocentric frame, before velocity/actuator
  transformations, matching the training buffer's opponent-action convention.
- After an auto-reset, the new observation has `is_first=True` and zero previous
  commands. No action is selected from the previous game's terminal image.

The legacy `evaluate_dreamer.py` controls zeroing with
`--opponent_action_method zero`; omitting it leaves exact observations exposed
by the PPO/FastSAC opponent wrappers. However, `Dreamer.act()` consumes
`state.prev_action` before it stores `obs.opponent_action`, causing the legacy
player path to use an opponent command one transition late. The tournament
adapter explicitly fills `state.prev_action` **before** calling `act()` and
uses the raw post-reset observation. These corrected timing/reset semantics
mean its results need not reproduce historical numbers from the legacy loop
bit for bit. Training code and the vendored Dreamer implementation are unchanged.

## Outputs and statistics

```text
<output>/
  manifest.json                    # settings, resolved env, paths and SHA-256 fingerprints
  <matchup>_leg0/ (and leg1/)
    evaluation_config.yaml         # resolved configuration used by this process
    games.npz                      # one outcome per completed game; atomically replaced
    run.log
  summary.json
  summary.csv
  report.md                        # exact/zero side by side; full primary round robin
```

`EvalMetricsTracker.save_npz()` writes outcomes to `games.npz` via the worker.
Arrays include `outcome`, `outcome_legend`, raw `termination_flags`,
`termination_names`, `env_id`, `episode_id`, episode length and existing
contact/speed metrics. The file stores the leg, condition, seed, run fingerprint
and completeness flag. `(leg, env_id, episode_id)` identifies a sample.

Canonical outcomes follow the repository's existing priority:
`goal_scored > goal_conceded > player_in_goal > opponent_in_goal > time_out`.
Codes 0 and 3 are wins; 1 and 2 losses; 4 a draw. Simultaneous flags count as
one game under that priority; raw flag counts remain available for auditing.
Disabled terms no longer shift these codes. Auxiliary contact/speed metrics
retain the existing tracker's documented post-reset-tick approximation;
termination outcomes are read from the termination manager.

For each matchup, legs are combined in the named agent's perspective:
`N=W+L+D`, `score=(W+0.5*D)/N`, `decisive_ratio=W/L`. A loss-free result has
ratio `infinity` if W>0, or `undefined` for all draws. The bootstrap explicitly
samples the individual `{1,0.5,0}` game payoffs with replacement 10,000 times
and reports the 2.5th and 97.5th percentiles of the mean. It does not bootstrap
termination counters, environments, seeds or precomputed win rates. These are
the requested game-level intervals, not estimates of variation across training
seeds. Reverse-perspective entries use `1-score` and `[1-upper,1-lower]`.

Rebuild reports on a machine with NumPy, without Isaac Lab or checkpoints:

```bash
python scripts/tournament/report.py logs/tournament/three_agents_10k
```

Missing/partial legs produce an explicitly incomplete report and a nonzero exit
status. No historical tournament result files were found locally. Legacy NPZs
without the corrected outcome schema are refused because active-term ordering
must be checked before interpreting their codes; aggregate termination counts
alone cannot resolve simultaneous outcomes.

## Inspected checkpoints and local validation

All paths below are relative to `logs/`:

| Agent | Checkpoint | Required config / format |
|---|---|---|
| D-RSSM | `dreamer/training/2026-05-09/09-16-19/latest.pt` | Adjacent `.hydra/config.yaml`; `agent_state_dict`, training step 9,502,656 |
| PPO-B | `rl_games/klask/new_agent_from_pretrained/new_AM/export/klask_ppo_nn_v1.1.pth` | Run's `params/agent.yaml`; rl_games `model`, including running observation/value statistics; 256/128/64 MLP |
| FastSAC | `fast_sac_isaaclab/training/2026-05-17/11-25-36/model_final.pt` | Embedded `config`, `actor_state_dict`, `obs_normalizer_state`; normalizer count 2,048,000,000 |

Inspected SHA-256 values:

```text
D-RSSM  e60b1d5750d90554c2d5095927349e49f9c150351c59d72b0a0edc580f99e138
PPO-B   d7be05e731c868417bdcf3f83addbdc4b4f8b1538b0ca4d4355423a04c7e23c2
FastSAC 9ae815a092ed11a9f0cffb809ea149a7fac20c7184caa041597ca41b185687eb
```

PPO-B's saved config contains historical training initialization metadata;
the loader replaces its `load_path` with the explicitly selected PPO-B file
before constructing the player and loading weights. There is no fallback.

Existing entry points inspected: `dreamer/evaluate_dreamer.py` (Dreamer vs
Dreamer/PPO/FastSAC), `rl_games/play_klask.py` (PPO vs PPO), and
`fast_sac/play_fast_sac_isaaclab.py` (FastSAC vs FastSAC). Their saved results
use `EvalMetricsTracker`; the Dreamer script writes under
`logs/dreamer/head_to_head_results`, and FastSAC writes beside its checkpoint.
Those existing scripts did not provide the complete mixed three-agent schedule
or the requested bootstrap report.

The local Mac passed CPU accounting/timing/report tests and real checkpoint
load/forward checks for D-RSSM and PPO-B. FastSAC's tensors and normalization
state were inspected, but its forward pass could not be checked because its
submodule is empty. No simulator tournaments or numerical tournament results
have been produced locally. Remaining runtime dependencies are:

- Isaac Lab with a supported NVIDIA GPU inside the existing container.
- The FastSAC `klask_her` submodule/mount.
- The exact configured actuator checkpoint:
  `logs/actuator_model/data/new/checkpoints/model_data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_train_seed0.pt`.
  A differently named actuator file exists in the source tree; its equivalence
  is unknown and it is not silently substituted.

Run the regression suite from the project directory with a Python environment
containing `pytest`, `numpy`, `torch`, `PyYAML`, `omegaconf`, `gymnasium`, and `tqdm`:

```bash
python -m pytest scripts/tournament/tests -q
```
