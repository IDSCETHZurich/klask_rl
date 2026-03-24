from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# from . import mdp
import isaaclab.envs.mdp as mdp

from klask_rl.assets.robots.klask import KLASK_PARAMS


from ..utils_manager_based import (
    reset_joints_by_offset,
    reset_ball_hit_tracking,
    reset_ball_hit_timer,
)


@configclass
class EventCfg:
    """Configuration for events.

    Domain randomization events are always defined with placeholder ranges.
    The training script configures them at runtime from agent_cfg YAML:
    - Sets actual ranges from the YAML config
    - Disables events (sets to None) when enable=false or no DR config present
    """

    # -- domain randomization (configured at runtime from YAML) --
    add_ball_mass: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.0, 0.0),  # overridden from YAML
            "operation": "abs",
        },
    )

    randomize_material_ball: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (0.0, 0.0),  # overridden from YAML
            "dynamic_friction_range": (0.0, 0.0),  # overridden from YAML
            "restitution_range": (0.0, 0.0),  # overridden from YAML
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_material_klask: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask"),
            "static_friction_range": (0.0, 0.0),  # overridden from YAML
            "dynamic_friction_range": (0.0, 0.0),  # overridden from YAML
            "restitution_range": (0.0, 0.0),  # overridden from YAML
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_actuator: EventTerm | None = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask"),
            "stiffness_distribution_params": (0.0, 0.0),  # overridden from YAML
            "damping_distribution_params": (0.0, 0.0),  # overridden from YAML
            "operation": "abs",
        },
    )

    # -- non-DR reset events (always active) --
    reset_x_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            # "position_range": (0.0202, 0.0202),
            # "velocity_range": (0.086, 0.086)
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_x_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_1 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": (-0.025, 0.085),
            # "position_range": (-0.1103, -0.1103),
            # "velocity_range": (-0.0043, -0.0043)
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_2 = EventTerm(
        func=reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": (-0.085, 0.025),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": KLASK_PARAMS["ball_reset_position_x"],
                "y": KLASK_PARAMS["ball_reset_position_y"],
                "z": (0.032, 0.032),
            },
            "velocity_range": {"x": (-0.0, 0.0), "y": (-0.00, 0.00)},
        },
    )

    reset_ball_hit_flag = EventTerm(
        func=reset_ball_hit_tracking,
        mode="reset",
        params={},
    )

    reset_ball_hit_timer_event = EventTerm(
        func=reset_ball_hit_timer,
        mode="reset",
        params={},
    )


@configclass
class EventCfgSac(EventCfg):
    """Event configuration for SAC training.

    Modifications from base EventCfg:
    - Ball spawns only in player's half (y < 0)
    - Opponent spawns at random position but stays stationary (no velocity actions)
    """

    # Override ball reset to spawn only in player's half (y < 0)
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": KLASK_PARAMS["ball_reset_position_x"],  # Full x range
                "y": (-0.10, -0.02),  # (-0.21, -0.02),  # Only player's half (y < 0), avoiding goal area
                "z": (0.032, 0.032),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
            },  # Ball starts stationary
        },
    )


@configclass
class EventCfgDreamer(EventCfg):
    """Event configuration for Dreamer training."""

    # Override ball reset to spawn only in player's half (y < 0)
    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": KLASK_PARAMS["ball_reset_position_x"],  # Full x range
                "y": (-0.10, -0.02),  # (-0.21, -0.02),  # Only player's half (y < 0), avoiding goal area
                "z": (0.032, 0.032),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
            },  # Ball starts stationary
        },
    )
