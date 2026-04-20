# from . import mdp
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from klask_rl.assets.robots.klask_params import KLASK_PARAMS

from ..utils_manager_based import (
    ball_in_goal,
    ball_in_own_half,
    ball_speed,
    ball_stationary,
    collision_player_ball,
    collision_player_ball_simple,
    collision_player_ball_time_decay,
    distance_ball_goal,
    distance_player_ball_own_half,
    distance_to_wall,
    in_goal,
    peg_in_defense_line_with_rebounds,
    peg_speed,
    peg_speed_exp,
    proximity_player_ball,
    shot_over_middle,
    termination_reward_time_decay,
)


# ---------------------------------------------------------------------------
# Factory: generate a dict of reward terms from a given perspective.
# Mirrors the observation factory pattern in klask_rl_observations_cfg.py.
# ---------------------------------------------------------------------------


def _perspective_reward_terms(
    own_body: str,
    other_body: str,
    own_goal: tuple,
    other_goal: tuple,
    y_pos_limit: tuple,
    own_half_sign: float,
) -> dict[str, RewTerm]:
    """Build reward terms parameterized for one player's perspective.

    Parameters
    ----------
    own_body : str
        Body name for the player ("Peg_1" or "Peg_2").
    other_body : str
        Body name for the opponent.
    own_goal : tuple
        ``(cx, cy, r)`` of the player's goal.
    other_goal : tuple
        ``(cx, cy, r)`` of the opponent's goal.
    y_pos_limit : tuple
        Y-axis joint position limits for the player's half (for wall distance).
    own_half_sign : float
        ``-1.0`` when own half is ``y < 0`` (player), ``+1.0`` when ``y > 0`` (opponent).
    """
    own_cfg = SceneEntityCfg("klask", body_names=[own_body])
    other_cfg = SceneEntityCfg("klask", body_names=[other_body])
    ball_cfg = SceneEntityCfg("ball")

    # direction_sign for shot_over_middle: +1 means ball moving in +y (toward
    # opponent goal), -1 means ball moving in -y (toward player goal).
    direction_sign = -own_half_sign  # player (own_half_sign=-1) → direction=+1

    return {
        "time_punishment": RewTerm(func=mdp.is_alive, weight=0.0),
        "time_out_punishment": RewTerm(func=mdp.time_out, weight=0.0),
        "shot_over_middle_line": RewTerm(
            func=shot_over_middle,
            params={"ball_cfg": ball_cfg, "direction_sign": direction_sign},
            weight=0.0,
        ),
        "player_in_goal": RewTerm(
            func=in_goal,
            params={"asset_cfg": own_cfg, "goal": own_goal},
            weight=0.0,
        ),
        "opponent_in_goal": RewTerm(
            func=in_goal,
            params={"asset_cfg": other_cfg, "goal": other_goal},
            weight=0.0,
        ),
        "goal_scored": RewTerm(
            func=ball_in_goal,
            params={
                "asset_cfg": ball_cfg,
                "goal": other_goal,
                "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
            },
            weight=0.0,
        ),
        "goal_conceded": RewTerm(
            func=ball_in_goal,
            params={
                "asset_cfg": ball_cfg,
                "goal": own_goal,
                "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
            },
            weight=0.0,
        ),
        "distance_player_ball_own_half": RewTerm(
            func=distance_player_ball_own_half,
            params={
                "player_cfg": own_cfg,
                "ball_cfg": ball_cfg,
                "own_half_sign": own_half_sign,
            },
            weight=0.0,
        ),
        "collision_player_ball_simple": RewTerm(
            func=collision_player_ball_simple,
            params={"player_cfg": own_cfg, "ball_cfg": ball_cfg},
            weight=0.0,
        ),
        "distance_ball_opponent_goal": RewTerm(
            func=distance_ball_goal,
            params={"ball_cfg": ball_cfg, "goal": other_goal},
            weight=0.0,
        ),
        "distance_ball_own_goal": RewTerm(
            func=distance_ball_goal,
            params={"ball_cfg": ball_cfg, "goal": own_goal},
            weight=0.0,
        ),
        "ball_speed": RewTerm(
            func=ball_speed,
            params={"ball_cfg": ball_cfg},
            weight=0.0,
        ),
        "ball_stationary": RewTerm(
            func=ball_stationary,
            params={"ball_cfg": ball_cfg},
            weight=0.0,
        ),
        "collision_player_ball": RewTerm(
            func=collision_player_ball,
            params={"player_cfg": own_cfg, "ball_cfg": ball_cfg},
            weight=0.0,
        ),
        "ball_in_own_half": RewTerm(
            func=ball_in_own_half,
            params={"ball_cfg": ball_cfg, "own_half_sign": own_half_sign},
            weight=0.0,
        ),
        "player_speed": RewTerm(
            func=peg_speed,
            params={"player_cfg": own_cfg},
            weight=0.0,
        ),
        "player_speed_exp": RewTerm(
            func=peg_speed_exp,
            params={"player_cfg": own_cfg, "sigma": 0.3},
            weight=0.0,
        ),
        "close_to_boundaries": RewTerm(
            func=distance_to_wall,
            params={"player_cfg": own_cfg, "y_pos_limit": y_pos_limit},
            weight=0.0,
        ),
        "player_strategically_positioned": RewTerm(
            func=peg_in_defense_line_with_rebounds,
            params={
                "player_cfg": own_cfg,
                "opponent_cfg": other_cfg,
                "ball_cfg": ball_cfg,
            },
            weight=0.0,
        ),
    }


# ---------------------------------------------------------------------------
# Pre-built term dicts for player and opponent perspectives.
# ---------------------------------------------------------------------------

_player_reward_terms = _perspective_reward_terms(
    own_body="Peg_1",
    other_body="Peg_2",
    own_goal=KLASK_PARAMS["player_goal"],
    other_goal=KLASK_PARAMS["opponent_goal"],
    y_pos_limit=KLASK_PARAMS["joint_y1_pos_limit"],
    own_half_sign=-1.0,
)

_opponent_reward_terms = _perspective_reward_terms(
    own_body="Peg_2",
    other_body="Peg_1",
    own_goal=KLASK_PARAMS["opponent_goal"],
    other_goal=KLASK_PARAMS["player_goal"],
    y_pos_limit=KLASK_PARAMS["joint_y2_pos_limit"],
    own_half_sign=1.0,
)


# ---------------------------------------------------------------------------
# RewardsCfg — built from the player perspective factory output.
# All weights default to 0.0; actual weights are set at runtime by the
# CurriculumWrapper from YAML config.
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    time_punishment = _player_reward_terms["time_punishment"]
    time_out_punishment = _player_reward_terms["time_out_punishment"]
    shot_over_middle_line = _player_reward_terms["shot_over_middle_line"]
    player_in_goal = _player_reward_terms["player_in_goal"]
    opponent_in_goal = _player_reward_terms["opponent_in_goal"]
    goal_scored = _player_reward_terms["goal_scored"]
    goal_conceded = _player_reward_terms["goal_conceded"]
    distance_player_ball_own_half = _player_reward_terms["distance_player_ball_own_half"]
    collision_player_ball_simple = _player_reward_terms["collision_player_ball_simple"]
    distance_ball_opponent_goal = _player_reward_terms["distance_ball_opponent_goal"]
    distance_ball_own_goal = _player_reward_terms["distance_ball_own_goal"]
    ball_speed = _player_reward_terms["ball_speed"]
    ball_stationary = _player_reward_terms["ball_stationary"]
    collision_player_ball = _player_reward_terms["collision_player_ball"]
    ball_in_own_half = _player_reward_terms["ball_in_own_half"]
    player_speed = _player_reward_terms["player_speed"]
    player_speed_exp = _player_reward_terms["player_speed_exp"]
    close_to_boundaries = _player_reward_terms["close_to_boundaries"]
    player_strategically_positioned = _player_reward_terms["player_strategically_positioned"]


@configclass
class RewardsCfgDenseBallHit:
    """Dense rewards for SAC training - dense reward for approaching and hitting the ball.

    Reward structure:
    - Proximity reward: Exponential decay with distance, provides dense gradient everywhere
    - Collision reward: Time-decaying bonus for making contact (faster contact = higher reward)

    The collision reward uses linear time decay to reward faster ball contact:
    - At episode start: full collision bonus
    - At episode end: zero collision bonus
    This naturally incentivizes quick approach and contact.

    IMPORTANT: The collision reward uses `termination_reward_time_decay` which reads
    directly from the termination manager. This guarantees DETERMINISTIC behavior:
    the reward is given if and only if the "ball_hit" termination was triggered.
    The termination term name "ball_hit" must match exactly with TerminationsCfgSac.
    """

    # Dense reward for being close to the ball (exponential, so gradient exists everywhere)
    # Weight=5.0 so at dist=0 reward=5.0, at dist=0.2 reward=1.8, at dist=0.5 reward=0.4
    proximity_player_ball_reward = RewTerm(
        func=proximity_player_ball,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
        },
        weight=5.0,
    )

    # Time-decaying bonus for hitting the ball - rewards faster contact!
    # Uses termination_reward_time_decay for DETERMINISTIC coupling with termination.
    # Weight=500.0 means hitting at t=0 gives +500.0, at t=50% gives +250.0, at t=100% gives 0.0
    collision_player_ball_reward = RewTerm(
        func=termination_reward_time_decay,
        params={
            "termination_term": "ball_hit",  # Must match the termination term name in TerminationsCfgSac
            "decay_type": "linear",
        },
        weight=500.0,
    )


@configclass
class RewardsCfgSparseHer:
    """Sparse rewards for HER training - only termination reward, no dense proximity."""

    # Sparse termination-based reward only - no proximity bonus
    # HER will relabel failed experiences to create successful trajectories
    collision_player_ball_reward = RewTerm(
        func=termination_reward_time_decay,
        params={
            "termination_term": "ball_hit",
            "decay_type": "linear",
        },
        weight=0.0,
    )


@configclass
class RewardsCfgTwoStageHer:
    """Rewards for two-stage HER goal-scoring task.

    This reward config is designed for hierarchical goal-conditioned learning:
    1. Stage 1: Sparse reward for hitting the ball (non-terminating)
    2. Stage 2: Larger sparse reward for scoring a goal (terminating)

    Reward structure:
    - Ball hit: 500.0 (given each time player hits the ball)
    - Goal scored: 5000.0 (given when ball enters opponent goal)

    These rewards are handled by the env's reward manager, so the
    trained model works at inference without the HER wrapper.
    """

    # Ball hit detection - sparse reward for making contact
    collision_player_ball = RewTerm(
        func=collision_player_ball_time_decay,
        params={
            "player_cfg": SceneEntityCfg("klask", body_names=["Peg_1"]),
            "ball_cfg": SceneEntityCfg("ball"),
            "decay_type": "linear",
        },
        weight=0.0,
    )

    # Goal scored detection - larger sparse reward for scoring
    goal_scored = RewTerm(
        func=ball_in_goal,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "goal": KLASK_PARAMS["opponent_goal"],
            "max_ball_vel": KLASK_PARAMS["max_ball_vel"],
        },
        weight=0.0,
    )
