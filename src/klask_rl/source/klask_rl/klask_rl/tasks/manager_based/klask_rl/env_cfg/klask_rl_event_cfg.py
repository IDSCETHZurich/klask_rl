# from . import mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from klask_rl.assets.robots.klask_params import KLASK_PARAMS

from ..utils_manager_based import (
    reset_ball_hit_timer,
    reset_ball_hit_tracking,
    reset_joints_by_absolute,
    set_joint_position_limits,
    set_rigid_body_material,
    reset_player_velocity_toward_ball,
)


@configclass
class EventCfg:
    """Configuration for events.

    Domain randomization events are always defined with placeholder ranges.
    The training script configures them at runtime from agent_cfg YAML:
    - Sets actual ranges from the YAML config
    - Disables events (sets to None) when enable=false or no DR config present
    """

    # -- startup: override joint position limits from KLASK_PARAMS --
    init_joint_limits_x: EventTerm = EventTerm(
        func=set_joint_position_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1", "slider_to_peg_2"]),
            "lower": KLASK_PARAMS["joint_x_pos_limit"][0],
            "upper": KLASK_PARAMS["joint_x_pos_limit"][1],
        },
    )

    init_joint_limits_y1: EventTerm = EventTerm(
        func=set_joint_position_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "lower": KLASK_PARAMS["joint_y1_pos_limit"][0],
            "upper": KLASK_PARAMS["joint_y1_pos_limit"][1],
        },
    )

    init_joint_limits_y2: EventTerm = EventTerm(
        func=set_joint_position_limits,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "lower": KLASK_PARAMS["joint_y2_pos_limit"][0],
            "upper": KLASK_PARAMS["joint_y2_pos_limit"][1],
        },
    )

    # -- startup: set initial material properties from KLASK_PARAMS --
    init_material_board: EventTerm = EventTerm(
        func=set_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Wall_.*", "Ground", "peg_.*_slider"]),
            "static_friction": KLASK_PARAMS["board_static_friction"],
            "dynamic_friction": KLASK_PARAMS["board_dynamic_friction"],
            "restitution": KLASK_PARAMS["board_restitution"],
        },
    )

    init_material_peg: EventTerm = EventTerm(
        func=set_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1", "Peg_2"]),
            "static_friction": KLASK_PARAMS["peg_static_friction"],
            "dynamic_friction": KLASK_PARAMS["peg_dynamic_friction"],
            "restitution": KLASK_PARAMS["peg_restitution"],
        },
    )

    # -- domain randomization (defaults = nominal values, i.e. no-op) --
    # When no DR config is provided, ranges equal the nominal KLASK_PARAMS
    # value so the "randomization" is effectively disabled.  Training scripts
    # widen these ranges (or set to None) at runtime from YAML.
    add_ball_mass: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (
                KLASK_PARAMS["ball_mass_initial"],
                KLASK_PARAMS["ball_mass_initial"],
            ),
            "operation": "abs",
        },
    )

    randomize_material_ball: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": (
                KLASK_PARAMS["ball_static_friction"],
                KLASK_PARAMS["ball_static_friction"],
            ),
            "dynamic_friction_range": (
                KLASK_PARAMS["ball_dynamic_friction"],
                KLASK_PARAMS["ball_dynamic_friction"],
            ),
            "restitution_range": (
                KLASK_PARAMS["ball_restitution"],
                KLASK_PARAMS["ball_restitution"],
            ),
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_material_board: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Wall_.*", "Ground", "peg_.*_slider"]),
            "static_friction_range": (
                KLASK_PARAMS["board_static_friction"],
                KLASK_PARAMS["board_static_friction"],
            ),
            "dynamic_friction_range": (
                KLASK_PARAMS["board_dynamic_friction"],
                KLASK_PARAMS["board_dynamic_friction"],
            ),
            "restitution_range": (
                KLASK_PARAMS["board_restitution"],
                KLASK_PARAMS["board_restitution"],
            ),
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_material_peg: EventTerm | None = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1", "Peg_2"]),
            "static_friction_range": (
                KLASK_PARAMS["peg_static_friction"],
                KLASK_PARAMS["peg_static_friction"],
            ),
            "dynamic_friction_range": (
                KLASK_PARAMS["peg_dynamic_friction"],
                KLASK_PARAMS["peg_dynamic_friction"],
            ),
            "restitution_range": (
                KLASK_PARAMS["peg_restitution"],
                KLASK_PARAMS["peg_restitution"],
            ),
            "num_buckets": 100,
            "make_consistent": True,
        },
    )

    randomize_actuator_x: EventTerm | None = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1", "slider_to_peg_2"]),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (
                KLASK_PARAMS["actuator_x_damping"],
                KLASK_PARAMS["actuator_x_damping"],
            ),
            "operation": "abs",
        },
    )

    randomize_actuator_y: EventTerm | None = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1", "ground_to_slider_2"]),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (
                KLASK_PARAMS["actuator_y_damping"],
                KLASK_PARAMS["actuator_y_damping"],
            ),
            "operation": "abs",
        },
    )

    # -- non-DR reset events (always active) --
    reset_x_position_peg_1 = EventTerm(
        func=reset_joints_by_absolute,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_1"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_x_position_peg_2 = EventTerm(
        func=reset_joints_by_absolute,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["slider_to_peg_2"]),
            "position_range": (-0.1, 0.1),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_1 = EventTerm(
        func=reset_joints_by_absolute,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_1"]),
            "position_range": (-0.140, -0.030),  # absolute y in player half [-0.210, -0.020]
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_y_position_peg_2 = EventTerm(
        func=reset_joints_by_absolute,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("klask", joint_names=["ground_to_slider_2"]),
            "position_range": (0.030, 0.140),  # absolute y in opponent half [0.020, 0.210]
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

    # Apply scheduled player init velocity (no-op when InitializationWrapper is absent).
    # Runs last so that ball and player positions are already finalized.
    reset_player_velocity = EventTerm(
        func=reset_player_velocity_toward_ball,
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
                "y": KLASK_PARAMS["ball_reset_position_y"],  # (-0.10, -0.02),  # Only player's half (y < 0)
                "z": (0.032, 0.032),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
            },  # Ball starts stationary
        },
    )
