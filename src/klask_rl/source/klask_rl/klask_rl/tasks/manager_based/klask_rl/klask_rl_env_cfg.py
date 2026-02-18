from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass


from klask_rl.assets.robots.klask import KLASK_PARAMS

from .env_cfg import KlaskRlSceneCfg
from .env_cfg import ActionsCfg, ActionsCfgPlayerOnly
from .env_cfg import ObservationsCfg, GoalObservationsCfg, TwoStageHerObservationsCfg
from .env_cfg import EventCfg, EventCfgSac
from .env_cfg import (
    RewardsCfg,
    RewardsCfgDenseBallHit,
    RewardsCfgSparseHer,
    RewardsCfgTwoStageHer,
)
from .env_cfg import TerminationsCfg, TerminationsCfgSac, TerminationsCfgTwoStageHer


@configclass
class KlaskRlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the klask RL environment."""

    sim = SimulationCfg(
        physx=PhysxCfg(bounce_threshold_velocity=0.0),
        render_interval=KLASK_PARAMS["decimation"],
    )
    # Scene settings
    scene = KlaskRlSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = KLASK_PARAMS["timeout"]

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS["decimation"]
        # simulation settings
        self.sim.dt = KLASK_PARAMS["physics_dt"]


@configclass
class KlaskRlSacEnvCfg(KlaskRlEnvCfg):
    """Configuration for SAC training with minimal observations and dense rewards."""

    observations = (
        ObservationsCfg()
    )  # Minimal obs: direction_to_ball, player_pos, ball_pos
    actions = ActionsCfgPlayerOnly()
    events = EventCfgSac()
    rewards = RewardsCfgDenseBallHit()
    terminations = TerminationsCfgSac()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.decimation = KLASK_PARAMS["decimation"]
        self.sim.dt = KLASK_PARAMS["physics_dt"]
        self.max_episode_length = int(
            self.episode_length_s / (self.decimation * self.sim.dt)
        )


@configclass
class KlaskRlHerSacEnvCfg(KlaskRlSacEnvCfg):
    """Configuration for SAC+HER training for ball hitting task.

    Uses the same observations as KlaskRlSacEnvCfg but with sparse rewards
    suitable for HER (Hindsight Experience Replay).

    Key differences from KlaskRlSacEnvCfg:
    - Sparse reward (only on ball hit) instead of dense proximity reward
    - Same observations, terminations, and episode length

    HER will relabel failed experiences by substituting the achieved goal
    (where the player ended up) as the desired goal, creating successful
    trajectories from failures.
    """

    observations = ObservationsCfg()  # Same observations as dense SAC
    actions = ActionsCfgPlayerOnly()
    events = EventCfgSac()
    rewards = RewardsCfgSparseHer()  # Sparse reward for HER
    terminations = TerminationsCfgSac()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()


@configclass
class KlaskRlTwoStageHerEnvCfg(KlaskRlEnvCfg):
    """Configuration for two-stage HER training for goal scoring.

    This environment is designed for hierarchical goal-conditioned learning:
    1. Stage 1: Learn to hit the ball (player → ball)
    2. Stage 2: Learn to score a goal (ball → opponent goal)

    Key features:
    - Observations: 14 dims (12 base + 2 for opponent goal XY)
    - Rewards: collision_player_ball (500) and goal_scored (5000) - sparse
    - Terminations: timeout or goal scored (ball hit does NOT terminate)
    - Episode length: 4s (allows time for ball dynamics after hit)

    The Sb3TwoStageHerWrapper:
    - Converts flat obs to GoalEnv dict (observation, achieved_goal, desired_goal)
    - Tracks phase (pre-hit vs post-hit) for goal extraction
    - Provides compute_reward for HER goal relabeling

    Note: The model can run at inference WITHOUT the HER wrapper since
    observations and rewards are handled by the env, not the wrapper.
    """

    observations = TwoStageHerObservationsCfg()
    actions = ActionsCfgPlayerOnly()
    events = EventCfgSac()
    rewards = RewardsCfgTwoStageHer()
    terminations = TerminationsCfgTwoStageHer()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.decimation = KLASK_PARAMS["decimation"]
        self.sim.dt = KLASK_PARAMS["physics_dt"]
        self.max_episode_length = int(
            self.episode_length_s / (self.decimation * self.sim.dt)
        )
