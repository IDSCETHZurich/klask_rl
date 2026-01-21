# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# from . import mdp
import isaaclab.envs.mdp as mdp

from klask_rl.assets.robots.klask import KLASK_CFG, KLASK_PARAMS
from .utils_manager_based import (
    angle_ball_goal,
    angle_ball_opp,
    ball_in_goal,
    ball_in_own_half,
    ball_speed,
    ball_stationary,
    body_xy_pos_w,
    collision_player_ball,
    distance_ball_goal,
    distance_ball_to_player,
    distance_player_ball_own_half,
    distance_to_goal,
    distance_to_wall,
    in_goal,
    opponent_goal_obs,
    peg_in_defense_line_with_rebounds,
    reset_joints_by_offset,
    root_lin_xy_vel_w,
    root_xy_pos_w,
    shot_over_middle,
)


@configclass
class KlaskRlSceneCfg(InteractiveSceneCfg):
    """Configuration for Klask scene."""

    # ground plane
    # ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.007,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(rigid_body_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=KLASK_PARAMS["ball_mass_initial"]),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=KLASK_PARAMS["ball_restitution"],
                static_friction=KLASK_PARAMS["ball_static_friction"],
                dynamic_friction=KLASK_PARAMS["ball_dynamic_friction"],
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.032)),
    )

    # contact_sensor = ContactSensorCfg(
    #    prim_path="{ENV_REGEX_NS}/Ball",
    #    filter_prim_paths_expr=["{ENV_REGEX_NS}/Klask/Peg_1"],
    #    history_length=KLASK_PARAMS["decimation"]
    # )

    klask = KLASK_CFG.replace(prim_path="{ENV_REGEX_NS}/Klask")


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    player_x = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["slider_to_peg_1"])

    player_y = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["ground_to_slider_1"])

    opponent_x = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["slider_to_peg_2"])

    opponent_y = mdp.JointVelocityActionCfg(asset_name="klask", joint_names=["ground_to_slider_2"])

    """ player = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["board_to_peg_1"]
    )

    opponent = mdp.JointVelocityActionCfg(
        asset_name="klask",
        joint_names=["board_to_peg_2"]
    ) """


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"])},
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"])},
        )

        peg_2_pos = ObsTerm(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )

        peg_2_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"])},
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"])},
        )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        if KLASK_PARAMS.get("additional_observations", 0):
            angle_pegball_pegoppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
            )

            angle_oppball_oppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "goal": KLASK_PARAMS["player_goal"],
                },
            )

            angle_pegball_pegopp = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            angle_oppball_opppeg = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_goal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["player_goal"]},
            )
            distance_ball_oppgoal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["opponent_goal"]},
            )

            distance_ball_player = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_opp = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

        if KLASK_PARAMS.get("action_history", 0):
            action_history_x = ObsTerm(
                func=mdp.last_action, params={"action_name": "player_x"}, history_length=KLASK_PARAMS["action_history"]
            )

            action_history_y = ObsTerm(
                func=mdp.last_action, params={"action_name": "player_y"}, history_length=KLASK_PARAMS["action_history"]
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OpponentCfg(ObsGroup):
        """Observations for opponent"""

        # observation terms (order preserved)
        peg_2_pos = ObsTerm(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )

        peg_2_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"])},
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"])},
        )

        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"])},
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"])},
        )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        if KLASK_PARAMS.get("additional_observations", 0):
            angle_oppball_oppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "goal": KLASK_PARAMS["player_goal"],
                },
            )

            angle_pegball_pegoppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
            )

            angle_oppball_opppeg = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            angle_pegball_pegopp = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            distance_ball_oppgoal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["opponent_goal"]},
            )
            distance_ball_goal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["player_goal"]},
            )

            distance_ball_opp = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            distance_ball_player = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

        if KLASK_PARAMS.get("action_history", 0):
            action_history_x = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "opponent_x"},
                history_length=KLASK_PARAMS["action_history"],
            )

            action_history_y = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "opponent_y"},
                history_length=KLASK_PARAMS["action_history"],
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    opponent: OpponentCfg = OpponentCfg()


@configclass
class GoalObservationsCfg:
    """Observation specifications for the environment."""

    # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )  # scale=2/BOARD_WIDTH)

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"])},
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"])},
        )

        peg_2_pos = ObsTerm(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )  # scale=2/BOARD_LENGTH)

        peg_2_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"])},
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"])},
        )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AchievedGoalCfg(ObsGroup):
        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class DesiredGoalCfg(ObsGroup):
        ball_in_goal = ObsTerm(func=opponent_goal_obs, params={"goal": (0.0, 0.176215)})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    observation: PolicyCfg = PolicyCfg()
    desired_goal: DesiredGoalCfg = DesiredGoalCfg()
    achieved_goal: AchievedGoalCfg = AchievedGoalCfg()

    """Observation specifications for the environment."""

    # TODO: noise corruption
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_1"])},
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_1"])},
        )

        peg_2_pos = ObsTerm(
            func=body_xy_pos_w,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask",
                    body_names=["Peg_2"],
                )
            },
        )

        peg_2_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["slider_to_peg_2"])},
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=["ground_to_slider_2"])},
        )

        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        if KLASK_PARAMS.get("additional_observations", 0):
            angle_pegball_pegoppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
            )

            angle_oppball_oppgoal = ObsTerm(
                func=angle_ball_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "goal": KLASK_PARAMS["player_goal"],
                },
            )

            angle_pegball_pegopp = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

            angle_oppball_opppeg = ObsTerm(
                func=angle_ball_opp,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_1_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                    "player_2_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_goal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["player_goal"]},
            )
            distance_ball_oppgoal = ObsTerm(
                func=distance_to_goal,
                params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": KLASK_PARAMS["opponent_goal"]},
            )

            distance_ball_player = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"]),
                },
            )

            distance_ball_opp = ObsTerm(
                func=distance_ball_to_player,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "player_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"]),
                },
            )

        if KLASK_PARAMS.get("action_history", 0):
            action_history_x = ObsTerm(
                func=mdp.last_action, params={"action_name": "player_x"}, history_length=KLASK_PARAMS["action_history"]
            )

            action_history_y = ObsTerm(
                func=mdp.last_action, params={"action_name": "player_y"}, history_length=KLASK_PARAMS["action_history"]
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


@configclass
class EventCfg:
    """Configuration for events."""

    if KLASK_PARAMS["domain_randomization"]["use_domain_randomization"]:
        # on reset
        add_ball_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "mass_distribution_params": KLASK_PARAMS["domain_randomization"]["ball_mass_range"],
                "operation": "abs",
            },
        )

        randomize_material_ball = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("ball"),
                "static_friction_range": KLASK_PARAMS["domain_randomization"]["static_friction_range"],
                "dynamic_friction_range": KLASK_PARAMS["domain_randomization"]["dynamic_friction_range"],
                "restitution_range": KLASK_PARAMS["domain_randomization"]["restitution_range"],
                "num_buckets": 100,
                "make_consistent": True,
            },
        )

        randomize_material_klask = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("klask"),
                "static_friction_range": KLASK_PARAMS["domain_randomization"]["static_friction_range"],
                "dynamic_friction_range": KLASK_PARAMS["domain_randomization"]["dynamic_friction_range"],
                "restitution_range": KLASK_PARAMS["domain_randomization"]["restitution_range"],
                "num_buckets": 100,
                "make_consistent": True,
            },
        )

        randomize_actuator = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("klask"),
                "stiffness_distribution_params": KLASK_PARAMS["domain_randomization"]["stiffness_range"],
                "damping_distribution_params": KLASK_PARAMS["domain_randomization"]["damping_range"],
                "operation": "abs",
            },
        )

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
            "position_range": (-0.14, -0.03),
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
            "position_range": (0.03, 0.14),
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


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    time_punishment = RewTerm(func=mdp.is_alive, weight=0.0)

    time_out_punishment = RewTerm(func=mdp.time_out, weight=0.0)

    shot_over_middle_line = RewTerm(
        func=shot_over_middle,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )
    player_in_goal = RewTerm(
        func=in_goal,
        params={"asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]), "goal": KLASK_PARAMS["player_goal"]},
        weight=0.0,
    )

    opponent_in_goal = RewTerm(
        func=in_goal,
        params={"asset_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]), "goal": KLASK_PARAMS["opponent_goal"]},
        weight=0.0,
    )

    goal_scored = RewTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
        weight=0.0,
    )

    goal_conceded = RewTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
        weight=0.0,
    )

    distance_player_ball_own_half = RewTerm(
        func=distance_player_ball_own_half,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )

    distance_ball_opponent_goal = RewTerm(
        func=distance_ball_goal,
        params={"ball_cfg": SceneEntityCfg("ball"), "goal": KLASK_PARAMS["opponent_goal"]},
        weight=0.0,
    )
    distance_ball_own_goal = RewTerm(
        func=distance_ball_goal,
        params={"ball_cfg": SceneEntityCfg("ball"), "goal": KLASK_PARAMS["player_goal"]},
        weight=0.0,
    )

    ball_speed = RewTerm(
        func=ball_speed,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )

    ball_stationary = RewTerm(
        func=ball_stationary,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )

    collision_player_ball = RewTerm(
        func=collision_player_ball,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )

    ball_in_own_half = RewTerm(func=ball_in_own_half, params={"ball_cfg": SceneEntityCfg("ball")}, weight=0.0)

    close_to_boundaries = RewTerm(
        func=distance_to_wall, params={"player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"])}, weight=0.0
    )
    player_strategically_positioned = RewTerm(
        func=peg_in_defense_line_with_rebounds,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "opponent_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=0.0,
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    goal_scored = DoneTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
    )

    goal_conceded = DoneTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
    )

    player_in_goal = DoneTerm(
        func=in_goal,
        params={"asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]), "goal": KLASK_PARAMS["player_goal"]},
    )

    opponent_in_goal = DoneTerm(
        func=in_goal,
        params={"asset_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]), "goal": KLASK_PARAMS["opponent_goal"]},
    )


@configclass
class KlaskRlEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    sim = SimulationCfg(physx=PhysxCfg(bounce_threshold_velocity=0.0), render_interval=KLASK_PARAMS["decimation"])
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
        self.decimation = KLASK_PARAMS["decimation"]  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = KLASK_PARAMS["physics_dt"]  # sim step every 5ms: 200Hz


@configclass
class KlaskRlGoalEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    sim = SimulationCfg(physx=PhysxCfg(bounce_threshold_velocity=0.0), render_interval=KLASK_PARAMS["decimation"])
    # Scene settings
    scene = KlaskRlSceneCfg(num_envs=1, env_spacing=1.0)
    # Basic settings
    observations = GoalObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    episode_length_s = KLASK_PARAMS["timeout"]

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = (0.0, 0.0, 6.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # step settings
        self.decimation = KLASK_PARAMS["decimation"]  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = KLASK_PARAMS["physics_dt"]  # sim step every 5ms: 200Hz
