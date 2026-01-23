from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# from . import mdp
import isaaclab.envs.mdp as mdp

from klask_rl.assets.robots.klask import KLASK_PARAMS


from ..utils_manager_based import (
    ball_in_goal,
    collision_player_ball_bool,
    in_goal,
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
class TerminationsCfgSac:
    """Termination terms for SAC training (hit the ball task).

    Episode ends when:
    - Player hits the ball (success!)
    - Timeout (failure)
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate on ball collision - this is the success condition!
    # Uses physics contact forces for accurate collision detection at 50Hz
    ball_hit = DoneTerm(
        func=collision_player_ball_bool,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
