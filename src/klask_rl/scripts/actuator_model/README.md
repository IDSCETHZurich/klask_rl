# Actuator Model <!-- omit in toc -->

The actuator model approximates the nonlinear behavior of the real system so that the agent can learn the interaction with the board in simulation as close as possible to the real world.

This directory contains all the code to train, evaluate and compare different actuator models. All the scripts are designed to run inside the provided container. For more information on how to use the container, please refer to the dedicated documentation.

- [Capture Trajectories](#capture-trajectories)
- [Prepare Dataset](#prepare-dataset)
- [Train Actuator Model](#train-actuator-model)
- [Evaluate Actuator Model](#evaluate-actuator-model)
  - [Compare Simulated Trajectories with Real Trajectories](#compare-simulated-trajectories-with-real-trajectories)
  - [Compare Model Variation over different Seeds](#compare-model-variation-over-different-seeds)
  - [Compare different Actuator Models](#compare-different-actuator-models)

## Capture Trajectories

The first step is to capture trajectories of the real system. The [klask_hardware](https://github.com/IDSCETHZurich/klask_hardware) repo contains a ROS package called `klask_system_id` with a node `velocity_profile_node`. This node can be used to send velocity commands to the board and record the resulting trajectories. Further details on how to use this node can be found in the dedicated documentation of the `klask_system_id` package.

Once the ROS bag files are recorded, they can be moved to a folder accessible to this container for further processing (e.g. `/home/student/klask_rl_repo/src/klask_rl/logs/actuator_model/data/train_traj`). Make sure to move/copy both the `train` and the `validation` directory.

## Prepare Dataset

The next step is to prepare the dataset for training the actuator model. This involves processing the recorded trajectories and extracting the relevant data from the ROS bag files. The default parameters of the `create_dataset.py` script are set to process the trajectories recorded according to the most recent setup but can be adapted via the command line arguments if needed.

```bash
# Create dataset from recorded trajectories
python /workspace/klask_rl/scripts/actuator_model/create_dataset.py

# If the trajectories are not placed in the default folder you can pass the path as following:
python /workspace/klask_rl/scripts/actuator_model/create_dataset.py --data_dir /workspace/klask_rl/logs/actuator_model/data/train_traj
```

The script will create a `.npz` file for each trajectory in its respective subfolder which can be used for later analysis with the script `visualize_dataset_trajectory.py` and one `.npz`file in the `train` and `validation` folder containing the combined dataset for training the actuator model.

## Train Actuator Model

Once the dataset is prepared, the next step is to train the actuator model. The `train_actuator_model.py` provides a k-fold training procedure with the option to train multiple models with different random seeds for later comparison. The default parameters of the script are set to train the most recent setup but can be adapted via the command line arguments if needed.

```bash
python /workspace/klask_rl/scripts/actuator_model/train_actuator_model.py
```

If you train multiple models with different seeds, the training can be parallelized for better GPU utilization. The default parameters are set to run 4 different seed training runs in parallel which strikes a good balance between training time and GPU utilization.

## Evaluate Actuator Model

To evaluate the trained actuator model, we can use the validation trajectories captured on the real system. The `playback_actions.py` script takes the recorded initial state and velocity commands and plays them back in simulation using the trained actuator model. It will store a new `sim2real_xxx.npz` file in the same directory as the original trajectory which contains the simulated trajectory resulting from the playback of the velocity commands and the original real trajectory. It is advised to move these files after creation to a separate directory for better organization (e.g. `/workspace/klask_rl/logs/actuator_model/data/new/evaluation/sim2real_seed0/traj`).

```bash
# The script will playback each trajectory found in the given directory (if there exists a /train and /validation sub-directory it wil only playback the trajectories in the validation directory)
python /workspace/klask_rl/scripts/actuator_model/playback_actions.py --trajectory_dir /workspace/klask_rl/logs/actuator_model/data/train_traj

# You can specify a specific actuator model with:
python /workspace/klask_rl/scripts/actuator_model/playback_actions.py \
  --trajectory_dir /workspace/klask_rl/logs/actuator_model/data/train_traj \ 
  --actuator_model_checkpoint /workspace/klask_rl/logs/actuator_model/data/new/checkpoints/model_data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_train_seed0.pt

# Or you can playback the trajectories without an actuator model for comparison with:
python /workspace/klask_rl/scripts/actuator_model/playback_actions.py \
  --trajectory_dir /workspace/klask_rl/logs/actuator_model/data/train_traj \
  --no_actuator_model

# Since the new trajectories were collected without the boarder collision safety mechanism, you also want to pass the `--no_collision_avoidance` flag. This is not set by default for backwards compatibility but should be passed for each new collected trajectory unless configured otherwise in the `velocity_profile_node` of the `klask_system_id` package.
python /workspace/klask_rl/scripts/actuator_model/playback_actions.py \
  --trajectory_dir /workspace/klask_rl/logs/actuator_model/data/train_traj \
  --no_collision_avoidance
```

### Compare Simulated Trajectories with Real Trajectories

The `sim2real_evaluation.py` script analyses the `sim2real_xxx.npz` files produced by `playback_actions.py` for a single actuator model (i.e. a single seed). It overlays commanded, real, and simulated velocities, plots the per-trajectory velocity deviation and peg position, and writes the per-trajectory MSE/RMSE metrics to `sim2real_evaluation_metrics.json` next to the figures.

```bash
# Evaluate all trajectories of a single seed
python /workspace/klask_rl/scripts/actuator_model/sim2real_evaluation.py \
  /workspace/klask_rl/logs/actuator_model/data/new/evaluation/sim2real_seed0/traj \
  --separate
```

The `--separate` flag will produce one set of figures per trajectory. If not set, the script will produce one figure with all trajectories placed in subplots which can be useful for a quick overall comparison or for are final paper output.

### Compare Model Variation over different Seeds

The `sim2real_seed_analysis.py` script aggregates the per-seed outputs of `sim2real_evaluation.py` to quantify how much the trained actuator model varies between seeds. It expects a root directory containing one subfolder per seed (e.g. `seed0`, `seed1`, ...), each holding the `.npz` trajectories and the `sim2real_evaluation_metrics.json` produced in the previous step. The script generates velocity comparison plots with a cross-seed mean and ±1σ band, RMSE distribution boxplots over seeds, and JSON summary files with the aggregated statistics.

```bash
# Aggregate all seed subfolders under the evaluation root
python /workspace/klask_rl/scripts/actuator_model/sim2real_seed_analysis.py \
  /workspace/klask_rl/logs/actuator_model/data/new/evaluation/from_new_model

# Emit one velocity-comparison PNG per trajectory
python /workspace/klask_rl/scripts/actuator_model/sim2real_seed_analysis.py \
  /workspace/klask_rl/logs/actuator_model/data/new/evaluation/from_new_model \
  --separate
```

### Compare different Actuator Models

The `analyse_model.py` script compares the input/output behavior of one or more trained actuator model checkpoints directly, without needing recorded trajectories. It excites the models with synthetic command sequences and produces single-input Bode plots, simultaneous-excitation Bode plots, and step response plots, with all models overlaid in the same figures for visual comparison.

```bash
python /workspace/klask_rl/scripts/actuator_model/analyse_model.py \ 
  --models /workspace/klask_rl/logs/actuator_model/data/old/checkpoints/model_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_seed0.pt /workspace/klask_rl/logs/actuator_model/data/new_with_bug/checkpoints/model_data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_seed0.pt /workspace/klask_rl/logs/actuator_model/data/new/checkpoints/model_data_odrive_new_estimator_history_10_interval_0.02_delay_0.0_horizon3_with_states_train_seed0.pt \
  --labels "original with bug" "new with bug" "new"
```
