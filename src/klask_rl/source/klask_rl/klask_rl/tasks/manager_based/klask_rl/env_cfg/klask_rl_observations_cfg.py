# from . import mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from klask_rl.assets.robots.klask import KLASK_PARAMS

from ..utils_manager_based import (
    angle_ball_goal,
    angle_ball_opp,
    body_xy_pos_w,
    direction_ball_goal,
    direction_to_ball,
    distance_ball_to_player,
    distance_to_goal,
    goal_position_obs,
    padded_image,
    padded_image_rotated,
    root_lin_xy_vel_w,
    root_xy_pos_w,
    sprite_rendered_image,
    sprite_rendered_image_rotated,
)


def _peg_obs_group(
    own_body: str,
    own_x_joint: str,
    own_y_joint: str,
    other_body: str,
    other_x_joint: str,
    other_y_joint: str,
    rotate: bool = False,
) -> type:
    """Factory: returns a @configclass ObsGroup for one player's full policy observations.

    The observation order is always: own peg (pos+vel) → other peg (pos+vel) → ball (pos+vel).
    This ensures both players receive structurally identical observations from their own frame.

    When ``rotate=True`` a 180° rotation is applied (all positions and velocities are negated)
    so that the observations appear as if the player were on the other side of the board.
    """

    # scale factor: -1 for 180° rotation, None (no scaling) otherwise
    s2 = (-1.0, -1.0) if rotate else None  # for 2-component terms (pos, vel xy)
    s1 = -1.0 if rotate else None  # for 1-component terms (single joint vel)

    @configclass
    class _PegObsGroup(ObsGroup):
        own_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=[own_body])},
            scale=s2,
        )
        own_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[own_x_joint])},
            scale=s1,
        )
        own_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[own_y_joint])},
            scale=s1,
        )
        other_pos = ObsTerm(
            func=body_xy_pos_w,
            params={"asset_cfg": SceneEntityCfg(name="klask", body_names=[other_body])},
            scale=s2,
        )
        other_x_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[other_x_joint])},
            scale=s1,
        )
        other_y_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg(name="klask", joint_names=[other_y_joint])},
            scale=s1,
        )
        ball_pos_rel = ObsTerm(func=root_xy_pos_w, params={"asset_cfg": SceneEntityCfg(name="ball")}, scale=s2)
        ball_vel_rel = ObsTerm(func=root_lin_xy_vel_w, params={"asset_cfg": SceneEntityCfg(name="ball")}, scale=s2)

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


def _peg_obs_group_extended_vec(
    base: type,
    own_body: str,
    other_body: str,
    own_goal: tuple,
    other_goal: tuple,
) -> type:
    """Factory: returns a @configclass ObsGroup that extends the basic peg obs with
    vector auxiliary terms, from the perspective of ``own``.
    """

    @configclass
    class _PegObsGroupExtendedVec(base):

        ball_own_peg_vec = ObsTerm(
            func=direction_to_ball,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "player_cfg": SceneEntityCfg(name="klask", body_names=[own_body]),
            },
        )
        ball_oppo_goal_vec = ObsTerm(
            func=direction_ball_goal,
            params={
                "ball_cfg": SceneEntityCfg(name="ball"),
                "goal": other_goal,
            },
        )

    return _PegObsGroupExtendedVec


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


def _peg_obs_group_with_goal(base: type, goal: tuple, rotate: bool = False) -> type:
    """Factory: extends a peg obs group with a fixed goal position term."""

    s2 = (-1.0, -1.0) if rotate else None

    @configclass
    class _PegObsGroupWithGoal(base):
        opponent_goal = ObsTerm(func=goal_position_obs, params={"goal": goal[:2]}, scale=s2)

    return _PegObsGroupWithGoal


# ---------------------------------------------------------------------------
# Concrete ObsGroup classes (instantiated from factories)
# ---------------------------------------------------------------------------


def _fix_pickle(cls: type, name: str) -> type:
    """Make a factory-generated class picklable by fixing its qualname/name."""
    cls.__qualname__ = name
    cls.__name__ = name
    return cls


_PlayerPolicyCfg = _fix_pickle(
    _peg_obs_group(
        own_body="Peg_1",
        own_x_joint="slider_to_peg_1",
        own_y_joint="ground_to_slider_1",
        other_body="Peg_2",
        other_x_joint="slider_to_peg_2",
        other_y_joint="ground_to_slider_2",
    ),
    "_PlayerPolicyCfg",
)
_OpponentPolicyCfg = _fix_pickle(
    _peg_obs_group(
        own_body="Peg_2",
        own_x_joint="slider_to_peg_2",
        own_y_joint="ground_to_slider_2",
        other_body="Peg_1",
        other_x_joint="slider_to_peg_1",
        other_y_joint="ground_to_slider_1",
        rotate=True,
    ),
    "_OpponentPolicyCfg",
)

_PlayerPolicyExtendedCfg = _fix_pickle(
    _peg_obs_group_extended(
        base=_PlayerPolicyCfg,
        own_body="Peg_1",
        other_body="Peg_2",
        own_goal=KLASK_PARAMS["player_goal"],
        other_goal=KLASK_PARAMS["opponent_goal"],
    ),
    "_PlayerPolicyExtendedCfg",
)
_OpponentPolicyExtendedCfg = _fix_pickle(
    _peg_obs_group_extended(
        base=_OpponentPolicyCfg,
        own_body="Peg_2",
        other_body="Peg_1",
        own_goal=KLASK_PARAMS["opponent_goal"],
        other_goal=KLASK_PARAMS["player_goal"],
    ),
    "_OpponentPolicyExtendedCfg",
)

_PlayerPolicyExtendedVecCfg = _fix_pickle(
    _peg_obs_group_extended_vec(
        base=_PlayerPolicyCfg,
        own_body="Peg_1",
        other_body="Peg_2",
        own_goal=KLASK_PARAMS["player_goal"],
        other_goal=KLASK_PARAMS["opponent_goal"],
    ),
    "_PlayerPolicyExtendedVecCfg",
)
_OpponentPolicyExtendedVecCfg = _fix_pickle(
    _peg_obs_group_extended_vec(
        base=_OpponentPolicyCfg,
        own_body="Peg_2",
        other_body="Peg_1",
        own_goal=KLASK_PARAMS["opponent_goal"],
        other_goal=KLASK_PARAMS["player_goal"],
    ),
    "_OpponentPolicyExtendedVecCfg",
)

_PlayerPolicyActionHistoryCfg = _fix_pickle(
    _action_history_obs_group(
        base=_PlayerPolicyCfg, action_name="player", history_length=KLASK_PARAMS["action_history"]
    ),
    "_PlayerPolicyActionHistoryCfg",
)
_OpponentPolicyActionHistoryCfg = _fix_pickle(
    _action_history_obs_group(
        base=_OpponentPolicyCfg, action_name="opponent", history_length=KLASK_PARAMS["action_history"]
    ),
    "_OpponentPolicyActionHistoryCfg",
)

_TwoStageHerPlayerPolicyCfg = _fix_pickle(
    _peg_obs_group_with_goal(_PlayerPolicyCfg, KLASK_PARAMS["opponent_goal"]),
    "_TwoStageHerPlayerPolicyCfg",
)
_TwoStageHerOpponentPolicyCfg = _fix_pickle(
    _peg_obs_group_with_goal(_OpponentPolicyCfg, KLASK_PARAMS["player_goal"], rotate=True),
    "_TwoStageHerOpponentPolicyCfg",
)


_TwoStageHerPlayerPolicyExtendedCfg = _fix_pickle(
    _peg_obs_group_with_goal(_PlayerPolicyExtendedVecCfg, KLASK_PARAMS["opponent_goal"]),
    "_TwoStageHerPlayerPolicyExtendedCfg",
)
_TwoStageHerOpponentPolicyExtendedCfg = _fix_pickle(
    _peg_obs_group_with_goal(_OpponentPolicyExtendedVecCfg, KLASK_PARAMS["player_goal"], rotate=True),
    "_TwoStageHerOpponentPolicyExtendedCfg",
)


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
    policy: ObsGroup = _TwoStageHerPlayerPolicyExtendedCfg()
    opponent: ObsGroup = _TwoStageHerOpponentPolicyExtendedCfg()


@configclass
class DreamerObservationsCfg:
    """Observation specifications for DreamerV3."""

    @configclass
    class ImageObsGroup(ObsGroup):
        # TODO: change padding to transform once it is working.
        image = ObsTerm(
            func=padded_image,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                # Defaults for env.size=[128,128]. Overridden by train_dreamer.py.
                "target_h": 128,
                "target_w": 128,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class OpponentImageObsGroup(ObsGroup):
        image = ObsTerm(
            func=padded_image_rotated,
            params={
                "sensor_cfg": SceneEntityCfg("camera"),
                "data_type": "rgb",
                # Defaults for env.size=[128,128]. Overridden by train_dreamer.py.
                "target_h": 128,
                "target_w": 128,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: ObsGroup = _PlayerPolicyExtendedCfg()
    opponent: ObsGroup = _OpponentPolicyExtendedCfg()
    image: ObsGroup = ImageObsGroup()
    opponent_image: ObsGroup = OpponentImageObsGroup()


@configclass
class DreamerSpriteObservationsCfg(DreamerObservationsCfg):
    """DreamerV3 observations with sprite-rendered images instead of TiledCamera."""

    @configclass
    class SpriteImageObsGroup(ObsGroup):
        image = ObsTerm(
            func=sprite_rendered_image,
            params={
                "peg1_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
                "peg2_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
                "ball_cfg": SceneEntityCfg("ball"),
                # Defaults for env.size=[128,128]. Overridden by apply_camera_size_to_env_cfg.
                "target_h": 128,
                "target_w": 128,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SpriteOpponentImageObsGroup(ObsGroup):
        image = ObsTerm(
            func=sprite_rendered_image_rotated,
            params={
                "peg1_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
                "peg2_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
                "ball_cfg": SceneEntityCfg("ball"),
                # Defaults for env.size=[128,128]. Overridden by apply_camera_size_to_env_cfg.
                "target_h": 128,
                "target_w": 128,
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # Override only the image groups; policy + opponent inherited from DreamerObservationsCfg.
    image: ObsGroup = SpriteImageObsGroup()
    opponent_image: ObsGroup = SpriteOpponentImageObsGroup()
