import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import PhysxCfg, RenderCfg, SimulationCfg
from isaaclab.utils import configclass
from klask_rl.assets.robots.klask import KLASK_PARAMS

from .env_cfg import (
    ActionsCfg,
    ActionsCfgPlayerOnly,
    DreamerObservationsCfg,
    DreamerSpriteObservationsCfg,
    EventCfg,
    EventCfgDreamer,
    EventCfgSac,
    FastSACObservationsCfg,
    KlaskRlDreamerSceneCfg,
    KlaskRlDreamerSpriteSceneCfg,
    KlaskRlSceneCfg,
    ObservationsCfg,
    ObservationsExtendedCfg,
    RewardsCfg,
    RewardsCfgDenseBallHit,
    RewardsCfgSparseHer,
    RewardsCfgTwoStageHer,
    TerminationsCfg,
    TerminationsCfgSac,
    TerminationsCfgTwoStageHer,
    TwoStageHerObservationsCfg,
)


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
    observations = ObservationsExtendedCfg()
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

    observations = ObservationsCfg()
    actions = ActionsCfgPlayerOnly()
    events = EventCfgSac()
    rewards = RewardsCfgDenseBallHit()
    terminations = TerminationsCfgSac()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        self.decimation = KLASK_PARAMS["decimation"]
        self.sim.dt = KLASK_PARAMS["physics_dt"]
        self.max_episode_length = int(self.episode_length_s / (self.decimation * self.sim.dt))


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

    observations = ObservationsCfg()
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
        self.max_episode_length = int(self.episode_length_s / (self.decimation * self.sim.dt))


@configclass
class KlaskRlContactPriorityEnvCfg(KlaskRlEnvCfg):
    """Configuration for contact-priority SAC training for goal scoring.

    Same env structure as Two-Stage HER (14-dim obs, sparse ball hit + goal
    scored rewards, ball_hit_timeout termination) but WITHOUT HER wrappers.
    Instead, the replay buffer over-samples transitions from episodes where
    ball contact occurred.

    Key features:
    - Observations: 14 dims (12 base + 2 for opponent goal XY)
    - Rewards: collision_player_ball and goal_scored (weights set from YAML)
    - Terminations: timeout, goal scored, or ball_hit_timeout
    - No HER / no goal relabeling — uses flat MlpPolicy
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
        self.max_episode_length = int(self.episode_length_s / (self.decimation * self.sim.dt))


@configclass
class KlaskRlFastSACEnvCfg(KlaskRlEnvCfg):
    """Configuration for FastSAC + HER training with self-play.

    Uses the 18-dim FastSAC observation layout (ball pos/vel, peg pos/vel,
    ball-peg diff, ball-goal diff, goal pos) in IsaacLab native coordinates.
    Both players are controlled (4-dim action) for self-play.
    Reward weights are set at runtime in the training script.
    """

    observations = FastSACObservationsCfg()
    actions = ActionsCfg()  # 4-dim: both player and opponent
    events = EventCfg()
    rewards = RewardsCfg()  # weights set to ±1000 at runtime
    terminations = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()


@configclass
class KlaskRlDreamerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Klask Dreamer environment."""

    sim = SimulationCfg(
        physx=PhysxCfg(bounce_threshold_velocity=0.0),
        render_interval=KLASK_PARAMS["decimation"],
        render=RenderCfg(antialiasing_mode="Off"),
    )
    # Scene settings
    scene = KlaskRlDreamerSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = DreamerObservationsCfg()
    actions = ActionsCfg()
    events = EventCfgDreamer()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = KLASK_PARAMS["timeout"]

    # Configurable ball reset area — override from yaml config to change
    # where the ball spawns at the start of each episode.
    ball_reset_position_x: tuple = KLASK_PARAMS["ball_reset_position_x"]
    ball_reset_position_y: tuple = KLASK_PARAMS["ball_reset_position_y"]

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS["decimation"]
        # simulation settings
        self.sim.dt = KLASK_PARAMS["physics_dt"]
        # Propagate ball reset ranges into the event config.
        self.events.reset_ball_position.params["pose_range"]["x"] = self.ball_reset_position_x
        self.events.reset_ball_position.params["pose_range"]["y"] = self.ball_reset_position_y


# Default sprite asset paths, resolved relative to this file.
_SPRITE_ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "..", "scripts", "dreamer", "sprite_renderer", "assets"
    )
)


@configclass
class KlaskRlDreamerSpriteEnvCfg(KlaskRlDreamerEnvCfg):
    """Klask Dreamer env using sprite renderer instead of TiledCamera."""

    scene = KlaskRlDreamerSpriteSceneCfg(num_envs=1, env_spacing=1.0)
    observations = DreamerSpriteObservationsCfg()

    # Sprite asset paths (overridable from YAML).
    sprite_dir: str = os.path.join(_SPRITE_ASSETS_DIR, "sprites")
    background_path: str = os.path.join(_SPRITE_ASSETS_DIR, "background", "median_background.png")
