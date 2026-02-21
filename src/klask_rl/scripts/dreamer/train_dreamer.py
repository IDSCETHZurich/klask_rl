# Launch Isaac Sim Simulator first.

import argparse
import sys
import pathlib

from isaaclab.app import AppLauncher
from datetime import datetime

import numpy as np

# parse dreamer + isaaclab args in two stages
parser = argparse.ArgumentParser(description="Train DreamerV3 for KLASK.")
parser.add_argument("--config", nargs="+")
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining = parser.parse_known_args()

# DreamerV3 always needs cameras for visual observations
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import functools
import signal

import gymnasium as gym
import ruamel.yaml as yaml
import torch
from torch import distributions as torchd

# Isaac Sim may override the default SIGINT handler, preventing
# Python's KeyboardInterrupt from firing on Ctrl+C. Restore it
# so that try/finally cleanup (wandb.finish, etc.) works properly.
signal.signal(signal.SIGINT, signal.default_int_handler)

sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3torch import tools
from dreamerv3torch.envs.isaaclab import IsaacLabVecEnv
from dreamerv3torch.dreamer import Dreamer, count_steps, make_dataset

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.sim import RenderCfg

# import isaaclab tasks to trigger gym.register calls
import isaaclab_tasks  # noqa: F401
import klask_rl.tasks  # noqa: F401

from klask_rl.tasks.manager_based.klask_rl.actuator_model import ActuatorModelWrapper
from klask_rl.tasks.manager_based.klask_rl.wrappers import (
    CurriculumWrapper,
    KlaskRlRandomOpponentWrapper,
)
from klask_rl.tasks.manager_based.klask_rl.utils_manager_based import set_terminations


def make_isaac_env(config):
    """Create a vectorized IsaacLab environment wrapped for DreamerV3."""
    gym_id = config.task

    env_cfg_class = gym.spec(gym_id).kwargs["env_cfg_entry_point"]
    # resolve string to class
    if isinstance(env_cfg_class, str):
        module_name, class_name = env_cfg_class.rsplit(":", 1)
        import importlib

        mod = importlib.import_module(module_name)
        env_cfg_class = getattr(mod, class_name)

    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = config.envs
    env_cfg.decimation = config.action_repeat
    env_cfg.seed = config.seed
    env_cfg.episode_length_s = config.episode_length_s

    # IsaacLab defaults to DLSS which smooths the image significantly
    # so we disable the antialiasing for a more pixelated (and hence more realistic) image.
    env_cfg.sim.render = RenderCfg(antialiasing_mode="Off")

    # Create the environment
    isaac_env = gym.make(gym_id, cfg=env_cfg, render_mode="rgb_array")

    # Convert to single-agent instance if required
    if isinstance(isaac_env.unwrapped, DirectMARLEnv):
        isaac_env = multi_agent_to_single_agent(isaac_env)

    # Set bounded action space
    action_dim = isaac_env.unwrapped.single_action_space.shape[-1]
    isaac_env.unwrapped.single_action_space = gym.spaces.Box(
        low=-config.max_velocity, high=config.max_velocity, shape=(action_dim,), dtype=np.float32
    )
    isaac_env.unwrapped.action_space = gym.vector.utils.batch_space(
        isaac_env.unwrapped.single_action_space, isaac_env.unwrapped.num_envs
    )

    # Apply actuator model wrapper
    if getattr(config, "actuator_model", False):
        isaac_env = ActuatorModelWrapper(isaac_env)

    # Configure reward weights
    rewards_cfg = getattr(config, "rewards", None)
    if rewards_cfg:
        num_steps = config.steps / config.envs
        isaac_env = CurriculumWrapper(isaac_env, rewards_cfg, num_steps=num_steps, dynamic=True)

    # Configure terminations
    terminations_cfg = getattr(config, "terminations", None)
    if terminations_cfg:
        set_terminations(isaac_env, terminations_cfg)

    # Random opponent (single-agent training)
    isaac_env = KlaskRlRandomOpponentWrapper(isaac_env)

    vec_env = IsaacLabVecEnv(isaac_env)

    return vec_env


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    if config.logdir is None:
        config.logdir = (
            pathlib.Path().cwd() / "logs" / "dreamer" / config.task / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    config.logdir = pathlib.Path(config.logdir).expanduser()
    config.traindir = config.traindir or config.logdir / "train_eps"
    config.evaldir = config.evaldir or config.logdir / "eval_eps"

    print("Logdir", config.logdir)
    config.logdir.mkdir(parents=True, exist_ok=True)
    config.traindir.mkdir(parents=True, exist_ok=True)
    config.evaldir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    # step in logger is environmental step
    logger = tools.Logger(config.logdir, step, config)

    logger.print("Create envs.")
    if config.offline_traindir:
        directory = config.offline_traindir.format(**vars(config))
    else:
        directory = config.traindir
    train_eps = tools.load_episodes(directory, limit=config.dataset_size)
    if config.offline_evaldir:
        directory = config.offline_evaldir.format(**vars(config))
    else:
        directory = config.evaldir
    eval_eps = tools.load_episodes(directory, limit=1)

    isaac_env = make_isaac_env(config)

    # In IsaacLab, action_repeat is just physics decimation handled inside
    # env.step(). There is no agent-level action repeat, so set to 1 to
    # prevent the Dreamer class from scaling step counts.
    config.action_repeat = 1

    acts = isaac_env.action_space
    logger.print("Action Space", acts)
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        logger.print(f"Prefill dataset ({prefill} steps).")
        random_actor = torchd.independent.Independent(
            torchd.uniform.Uniform(
                torch.tensor(acts.low).repeat(config.envs, 1).to(config.device),
                torch.tensor(acts.high).repeat(config.envs, 1).to(config.device),
            ),
            1,
        )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate_vec(
            random_agent,
            isaac_env,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill
        logger.print(f"Logger: ({logger.step} steps).")

    logger.print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        isaac_env.observation_space,
        isaac_env.action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (config.logdir / "latest.pt").exists():
        checkpoint = torch.load(config.logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    exit_code = 0
    try:
        # make sure eval will be executed once after config.steps
        while agent._step < config.steps + config.eval_every:
            logger.write()
            if config.eval_episode_num > 0:
                logger.print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate_vec(
                    eval_policy,
                    isaac_env,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", tools.to_np(video_pred))
                state = None
            logger.print("Start training.")
            state = tools.simulate_vec(
                agent,
                isaac_env,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
            items_to_save = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            torch.save(items_to_save, config.logdir / "latest.pt")
    except KeyboardInterrupt:
        logger.print("\nTraining interrupted by user.")
        exit_code = 1
    finally:
        isaac_env.close()
        logger.close(exit_code=exit_code)


if __name__ == "__main__":

    _yaml = yaml.YAML(typ="safe", pure=True)
    configs = _yaml.load((pathlib.Path(__file__).parent / "configs.yaml").read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args_cli.config] if args_cli.config else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser2 = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser2.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    config = parser2.parse_args(remaining)

    main(config)
    simulation_app.close()
