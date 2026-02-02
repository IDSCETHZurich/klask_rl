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
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "goal": KLASK_PARAMS["player_goal"],
        },
    )

    opponent_in_goal = DoneTerm(
        func=in_goal,
        params={
            "asset_cfg": SceneEntityCfg("klask", body_names=["Peg_2"]),
            "goal": KLASK_PARAMS["opponent_goal"],
        },
    )


@configclass
class TerminationsCfgSac:
    """Termination terms for SAC training (hit the ball task).

    Episode ends when:
    - Timeout (allowing agent to track the ball continuously)

    Note: We don't terminate on ball hit anymore, so the agent learns to
    continuously track and hit the ball, not just reach it once.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    ball_hit = DoneTerm(
        func=collision_player_ball_bool,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )


@configclass
class TerminationsCfgTwoStageHer:
    """Termination terms for two-stage HER goal-scoring task.

    Episode ends when:
    - Timeout (truncation, handled by base env)
    - Goal scored (success termination, handled by wrapper)

    Note: Ball hit does NOT terminate the episode - it just transitions
    to stage 2 (scoring phase). The wrapper handles the termination logic
    for goal scoring.

    This config only specifies timeout, as the wrapper overrides
    termination for goal scored events.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Goal scored termination (monitored by wrapper, not base env)
    # The wrapper will override dones when goal is scored
    goal_scored = DoneTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
    )
