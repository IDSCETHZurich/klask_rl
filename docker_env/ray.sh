# This is currently just a collection of commands and is not yet an executable script.

docker build -t isaac-lab-ray:local -f Ray.Dockerfile .

docker run -it \
   --gpus all \
   --net=host \
   --entrypoint /bin/bash \
   isaac-lab-ray:local

# Start the Ray server within the tuning image
echo "import ray; ray.init(); import time; [time.sleep(10) for _ in iter(int, 1)]" | ./isaaclab.sh -p



/workspace/isaaclab/_isaac_sim/python.sh -m pip install -e /workspace/klask_rl/source/klask_rl


/workspace/isaaclab/_isaac_sim/python.sh /workspace/klask_rl/scripts/sb3/train_sac.py --config /workspace/klask_rl/scripts/sb3/config/experiments/klask_sac_two_stage_her.yaml

docker exec -it -d isaac-lab-klask-rl_container_local /workspace/isaaclab/_isaac_sim/python.sh /workspace/klask_rl/scripts/sb3/train_sac.py --config /workspace/klask_rl/scripts/sb3/config/experiments/klask_sac_two_stage_her.yaml


/workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py --run_mode local --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py --cfg_class KlaskSacHerTuneJobCfg --workflow /workspace/klask_rl/scripts/sb3/train_sac.py --metric rollout_ep_rew_mean --mode max --num_workers_per_node 2 --num_samples 2
docker exec -it -d isaac-lab-klask-rl_container_local bash -c "cd /workspace/isaaclab && /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py --run_mode local --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py --cfg_class KlaskSacHerTuneJobCfg --workflow /workspace/klask_rl/scripts/sb3/train_sac.py --metric rollout_ep_rew_mean --mode max --num_workers_per_node 3 --num_samples 2"

/workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py --run_mode local --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py --cfg_class KlaskSacTwoStageHerJobCfg --workflow /workspace/klask_rl/scripts/sb3/train_sac.py --metric two_stage_goal_scores_count --mode max --num_workers_per_node 3 --num_samples 50 --repeat_run_count 1
docker exec -it -d isaac-lab-klask-rl_container_local bash -c "cd /workspace/isaaclab && /workspace/isaaclab/_isaac_sim/python.sh /workspace/isaaclab/scripts/reinforcement_learning/ray/tuner.py --run_mode local --cfg_file /workspace/klask_rl/scripts/sb3/ray/hyperparameter_tuning/klask_sac_base_cfg.py --cfg_class KlaskSacTwoStageHerJobCfg --workflow /workspace/klask_rl/scripts/sb3/train_sac.py --metric two_stage_goal_scores_count --mode max --num_workers_per_node 3 --num_samples 50 --repeat_run_count 1"