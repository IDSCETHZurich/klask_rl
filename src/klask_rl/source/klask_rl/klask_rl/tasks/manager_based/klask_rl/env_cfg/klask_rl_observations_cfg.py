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
    goal_position_obs,
    root_lin_xy_vel_w,
    root_xy_pos_w,
)


def _peg_obs_group(
    own_body: str,
    own_x_joint: str,
    own_y_joint: str,
    other_body: str,
    other_x_joint: str,
    other_y_joint: str,
    own_goal: tuple,
    other_goal: tuple,
) -> type:
    """Factory: returns a @configclass ObsGroup for one player's full policy observations.

    The observation order is always: own peg (pos+vel) → other peg (pos+vel) → ball (pos+vel).
    This ensures both players receive structurally identical observations from their own frame.
    """

    @configclass
    class _PegObsGroup(ObsGroup):
        own_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=[own_body])},
        )
        own_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[own_x_joint])},
        )
        own_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[own_y_joint])},
        )
        other_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=[other_body])},
        )
        other_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[other_x_joint])},
        )
        other_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[other_y_joint])},
        )
        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")})
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    return _PegObsGroup


def _peg_obs_group_extended(
    base: type,
    own_body: str,
    other_body: str,
    own_goal: tuple,
    other_goal: tuple,
) -> type:
    """Factory: returns a @configclass ObsGroup that extends the basic peg obs with
    angle and distance auxiliary terms, from the perspective of ``own``.
    """

    @configclass
    class _PegObsGroupExtended(base):
        # Angle from own-peg through ball toward other's goal (offensive)
        angle_own_ball_goal = ObsTerm(
            func=angle_ball_goal,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_cfg": SceneEntityCfg(name="klask", body_names=[own_body]),
                "goal": other_goal,
            },
        )
        # Angle from other-peg through ball toward own goal (defensive)
        angle_other_ball_goal = ObsTerm(
            func=angle_ball_goal,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_cfg": SceneEntityCfg(name="klask", body_names=[other_body]),
                "goal": own_goal,
            },
        )
        # Angle: own-peg → ball → other-peg
        angle_own_ball_other = ObsTerm(
            func=angle_ball_opp,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_1_cfg": SceneEntityCfg(name="klask", body_names=[own_body]),
                "player_2_cfg": SceneEntityCfg(name="klask", body_names=[other_body]),
            },
        )
        # Angle: other-peg → ball → own-peg
        angle_other_ball_own = ObsTerm(
            func=angle_ball_opp,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_1_cfg": SceneEntityCfg(name="klask", body_names=[other_body]),
                "player_2_cfg": SceneEntityCfg(name="klask", body_names=[own_body]),
            },
        )
        # Distance: ball → own goal (defensive)
        distance_ball_own_goal = ObsTerm(
            func=distance_to_goal,
            params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": own_goal},
        )
        # Distance: ball → other goal (offensive)
        distance_ball_other_goal = ObsTerm(
            func=distance_to_goal,
            params={"ball_cfg": SceneEntityCfg(name="ball"), "goal": other_goal},
        )
        # Distance: ball → own peg
        distance_ball_own = ObsTerm(
            func=distance_ball_to_player,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_cfg": SceneEntityCfg(name="klask", body_names=[own_body]),
            },
        )
        # Distance: ball → other peg
        distance_ball_other = ObsTerm(
            func=distance_ball_to_player,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_cfg": SceneEntityCfg(name="klask", body_names=[other_body]),
            },
        )

    return _PegObsGroupExtended


def _action_history_obs_group(base: type, action_name: str, history_length: int) -> type:
    """Factory: returns a @configclass ObsGroup that extends a base obs group with a history of past actions."""

    @configclass
    class _ActionHistoryObsGroup(base):

        action_history_x = ObsTerm(
            func=mdp.last_action,
            params={"action_name": f"{action_name}_x"},
            history_length=history_length,
        )

        action_history_y = ObsTerm(
            func=mdp.last_action,
            params={"action_name": f"{action_name}_y"},
            history_length=history_length,
        )

    return _ActionHistoryObsGroup


def _peg_obs_group_with_goal(base: type, goal: tuple) -> type:
    """Factory: extends a peg obs group with a fixed goal position term."""

    @configclass
    class _PegObsGroupWithGoal(base):
        opponent_goal = ObsTerm(func=goal_position_obs, params={"goal": goal[:2]})

    return _PegObsGroupWithGoal


# ---------------------------------------------------------------------------
# Concrete ObsGroup classes (instantiated from factories)
# ---------------------------------------------------------------------------

_PlayerPolicyCfg = _peg_obs_group(
    own_body="Peg_1",
    own_x_joint="slider_to_peg_1",
    own_y_joint="ground_to_slider_1",
    other_body="Peg_2",
    other_x_joint="slider_to_peg_2",
    other_y_joint="ground_to_slider_2",
    own_goal=KLASK_PARAMS["player_goal"],
    other_goal=KLASK_PARAMS["opponent_goal"],
)
_OpponentPolicyCfg = _peg_obs_group(
    own_body="Peg_2",
    own_x_joint="slider_to_peg_2",
    own_y_joint="ground_to_slider_2",
    other_body="Peg_1",
    other_x_joint="slider_to_peg_1",
    other_y_joint="ground_to_slider_1",
    own_goal=KLASK_PARAMS["opponent_goal"],
    other_goal=KLASK_PARAMS["player_goal"],
)

_PlayerPolicyExtendedCfg = _peg_obs_group_extended(
    base=_PlayerPolicyCfg,
    own_body="Peg_1",
    other_body="Peg_2",
    own_goal=KLASK_PARAMS["player_goal"],
    other_goal=KLASK_PARAMS["opponent_goal"],
)
_OpponentPolicyExtendedCfg = _peg_obs_group_extended(
    base=_OpponentPolicyCfg,
    own_body="Peg_2",
    other_body="Peg_1",
    own_goal=KLASK_PARAMS["opponent_goal"],
    other_goal=KLASK_PARAMS["player_goal"],
)

_PlayerPolicyActionHistoryCfg = _action_history_obs_group(
    base=_PlayerPolicyCfg, action_name="player", history_length=KLASK_PARAMS["action_history"]
)
_OpponentPolicyActionHistoryCfg = _action_history_obs_group(
    base=_OpponentPolicyCfg, action_name="opponent", history_length=KLASK_PARAMS["action_history"]
)

_TwoStageHerPlayerPolicyCfg = _peg_obs_group_with_goal(_PlayerPolicyCfg, KLASK_PARAMS["opponent_goal"])
_TwoStageHerOpponentPolicyCfg = _peg_obs_group_with_goal(_OpponentPolicyCfg, KLASK_PARAMS["player_goal"])


# ---------------------------------------------------------------------------
# Top-level observation configs
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Observation specifications for the standard (self-play) environment."""

    # observation groups
    policy: ObsGroup = _PlayerPolicyCfg()
    opponent: ObsGroup = _OpponentPolicyCfg()


@configclass
class ObservationsExtendedCfg:
    """Observation specifications with additional angle and distance terms."""

    # observation groups
    policy: ObsGroup = _PlayerPolicyExtendedCfg()
    opponent: ObsGroup = _OpponentPolicyExtendedCfg()


@configclass
class TwoStageHerObservationsCfg:
    """Observation specifications for two-stage HER goal-scoring task.

    Extends the standard observations by appending the respective goal position
    so the model knows what it is trying to achieve at inference time.
    """

    # observation groups
    policy: ObsGroup = _TwoStageHerPlayerPolicyCfg()
    opponent: ObsGroup = _TwoStageHerOpponentPolicyCfg()


@configclass
class DreamerObservationsCfg:
    """Observation specifications for DreamerV3."""

    @configclass
    class ImageObsGroup(ObsGroup):
        image = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("camera"), "data_type": "rgb"},
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = _PlayerPolicyExtendedCfg()
    opponent: ObsGroup = _OpponentPolicyExtendedCfg()
    visual: ObsGroup = ImageObsGroup()
