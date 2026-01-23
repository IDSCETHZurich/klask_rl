from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

# from . import mdp
import isaaclab.envs.mdp as mdp

from klask_rl.assets.robots.klask import KLASK_PARAMS

from ..utils_manager_based import (
    ball_in_goal,
    ball_in_own_half,
    ball_speed,
    ball_stationary,
    collision_player_ball,
    distance_ball_goal,
    distance_player_ball_own_half,
    distance_to_wall,
    in_goal,
    peg_in_defense_line_with_rebounds,
    shot_over_middle,
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
class RewardsCfgSparseBallHit(RewardsCfg):
    """Sparse rewards for SAC training - only reward for hitting the ball.

    This is Step 1 of the SAC+HER curriculum:
    - Ball starts in player's half (stationary)
    - Opponent is stationary
    - Reward for collision with ball
    - Small time penalty to encourage faster hitting
    - Episode terminates on hit (see TerminationsCfgSac)
    """

    # Small time penalty to encourage faster hitting
    # This gives a slight negative reward each timestep, encouraging the agent to hit quickly
    # time_punishment = RewTerm(func=mdp.is_alive, weight=-0.01)

    # Main reward: hitting the ball
    collision_player_ball = RewTerm(
        func=collision_player_ball,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=1.0,  # Sparse reward for hitting the ball
    )


@configclass
class RewardsCfgSparseGoal(RewardsCfg):
    """Sparse rewards for SAC+HER training - only reward for scoring goals.

    This is Step 3 of the SAC+HER curriculum:
    - Goal-conditioned learning with HER
    - Sparse reward only for scoring
    """

    # Enable goal scoring reward
    goal_scored = RewTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
        weight=10.0,  # Large positive reward for scoring
    )

    # Penalty for conceding a goal
    goal_conceded = RewTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["player_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
        weight=-10.0,  # Large negative reward for conceding
    )
