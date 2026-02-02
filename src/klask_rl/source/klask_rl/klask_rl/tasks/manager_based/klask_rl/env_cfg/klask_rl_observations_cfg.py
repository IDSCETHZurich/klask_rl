from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# from . import mdp
import isaaclab.envs.mdp as mdp

from klask_rl.assets.robots.klask import KLASK_PARAMS

from ..utils_manager_based import (
    angle_ball_goal,
    angle_ball_opp,
    body_xy_pos_w,
    distance_ball_to_player,
    distance_to_goal,
    opponent_goal_obs,
    root_lin_xy_vel_w,
    root_xy_pos_w,
)


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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

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
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["player_goal"],
                },
            )
            distance_ball_oppgoal = ObsTerm(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
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
                func=mdp.last_action,
                params={"action_name": "player_x"},
                history_length=KLASK_PARAMS["action_history"],
            )

            action_history_y = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "player_y"},
                history_length=KLASK_PARAMS["action_history"],
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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
        )

        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

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
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
            )
            distance_ball_goal = ObsTerm(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["player_goal"],
                },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AchievedGoalCfg(ObsGroup):
        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )
        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}
        )

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
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["player_goal"],
                },
            )
            distance_ball_oppgoal = ObsTerm(
                func=distance_to_goal,
                params={
                    "ball_cfg": SceneEntityCfg(name="ball"),
                    "goal": KLASK_PARAMS["opponent_goal"],
                },
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
                func=mdp.last_action,
                params={"action_name": "player_x"},
                history_length=KLASK_PARAMS["action_history"],
            )

            action_history_y = ObsTerm(
                func=mdp.last_action,
                params={"action_name": "player_y"},
                history_length=KLASK_PARAMS["action_history"],
            )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True


@configclass
class TwoStageHerObservationsCfg:
    """Observation specifications for two-stage HER goal-scoring task.

    This config provides observations suitable for learning to:
    1. Hit the ball (player → ball)
    2. Score a goal (ball → opponent goal)

    The observation structure is:
    - peg_1_pos (2): player XY position
    - peg_1_vel (2): player XY velocity
    - peg_2_pos (2): opponent XY position
    - peg_2_vel (2): opponent XY velocity
    - ball_pos (2): ball XY position
    - ball_vel (2): ball XY velocity
    Total: 12 dimensions

    Note: opponent_goal_center is added by the Sb3TwoStageHerWrapper,
    not in the base observation, so HER can relabel it.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group - two-stage HER compatible."""

        # Player position and velocity (indices 0-3)
        peg_1_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_1"])},
        )

        peg_1_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_1"]
                )
            },
        )

        peg_1_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_1"]
                )
            },
        )

        # Opponent position and velocity (indices 4-7)
        peg_2_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=["Peg_2"])},
        )

        peg_2_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["slider_to_peg_2"]
                )
            },
        )

        peg_2_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="klask", joint_names=["ground_to_slider_2"]
                )
            },
        )

        # Ball position and velocity (indices 8-11)
        ball_pos_rel = ObsTerm(
            func=root_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="ball")},
        )

        ball_vel_rel = ObsTerm(
            func=root_lin_xy_vel_w,
            params={"asset_cfg": SceneEntityCfg(name="ball")},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
