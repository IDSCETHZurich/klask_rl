# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents
from .klask_rl_env_cfg import (
    ActionsCfgPlayerOnly,
    EventCfgSac,
    KlaskRlEnvCfg,
    KlaskRlGoalEnvCfg,
    KlaskRlHerEnvCfg,
    KlaskRlSacEnvCfg,
    RewardsCfgSparseBallHit,
    RewardsCfgSparseGoal,
    TerminationsCfgSac,
)
from .klask_rl_env_wrapper import (
    ActionHistoryWrapper,
    CurriculumWrapper,
    KlaskRlAgentOpponentWrapper,
    KlaskRlCollisionAvoidanceWrapper,
    KlaskRlRandomOpponentWrapper,
    ObservationNoiseWrapper,
    OpponentObservationWrapper,
    RlGamesGpuEnvSelfPlay,
    find_wrapper,
)

##
# Register Gym environments.
##


gym.register(
    id="Klask-Rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.klask_rl_env_cfg:KlaskRlEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)


gym.register(
    id="Klask-Rl-SAC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.klask_rl_env_cfg:KlaskRlSacEnvCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)


gym.register(
    id="Klask-Rl-HER-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.klask_rl_env_cfg:KlaskRlHerEnvCfg",
        "sb3_sac_cfg_entry_point": f"{agents.__name__}:sb3_sac_cfg.yaml",
    },
)
