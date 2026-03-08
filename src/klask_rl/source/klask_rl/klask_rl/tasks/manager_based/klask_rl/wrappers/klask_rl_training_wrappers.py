import torch
from gymnasium import Wrapper
from klask_rl.assets.robots.klask import KLASK_PARAMS


class RewardWeightWrapper(Wrapper):
    """Sets initial reward term weights from a config dict.

    Weights specified per-second are used as-is; others are converted from
    per-step to per-second by dividing by (decimation * physics_dt).
    For list-valued weights the first element is used as the initial value.
    """

    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg

        for term, weight in cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
            if type(weight) is dict:
                # List means [start, end] for curriculum — use start value.
                if type(weight["weight"]) is list:
                    _weight = weight["weight"][0]
                else:
                    _weight = weight["weight"]
                if not weight.get("per_second", False):
                    _weight /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
                    # TODO: Refactor this logic to make it more readable and saner. If per_second is True we need to scale it so that it applies per second and if per_second=False we do nothing and apply it. Now it is the other way around which is super confusing.
            else:
                _weight = weight / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = _weight


class CurriculumWrapper(RewardWeightWrapper):
    """Sets initial reward weights and adjusts them over training.

    Adds two per-step mechanisms:
      1. **Linear schedule** — for terms with weight=[start, end], the weight
         is linearly interpolated from start to end over ``num_steps``.
      2. **Exponential decay** (``dynamic=True``) — auxiliary shaping rewards
         decay exponentially and are zeroed after 20M steps so that only
         sparse game-outcome rewards remain.
    """

    # Terms excluded from dynamic decay (game-outcome / sparse signals).
    _DECAY_EXCLUDE = frozenset(
        {
            "ball_stationary",
            "time_out_punishment",
            "time_punishment",
            "goal_scored",
            "goal_conceded",
            "opponent_in_goal",
            "player_in_goal",
        }
    )

    def __init__(self, env, cfg, num_steps=None, dynamic=False):
        super().__init__(env, cfg)
        self.dynamic = dynamic
        self.num_steps = num_steps
        self._step = 0

    def step(self, actions):
        self._step += self.env.unwrapped.num_envs
        for term, weight in self.cfg.items():
            # 1. Linear weight schedule for [start, end] entries.
            if type(weight) is dict and type(weight["weight"]) is list:
                term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                weight_step = (weight["weight"][1] - weight["weight"][0]) / self.num_steps
                if not weight.get("per_second", False):
                    weight_step /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
                self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight += weight_step

            # 2. Exponential decay for auxiliary shaping rewards.
            if self.dynamic and term not in self._DECAY_EXCLUDE:
                term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = weight * (
                    torch.exp(
                        -torch.tensor(
                            self._step / 10000000,
                            device=self.env.unwrapped.device,
                            dtype=torch.float32,
                        )
                    )
                )  # Half-life ~6.9M steps; fully zeroed after 20M.
                if self._step > 20_000_000:
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = torch.tensor(
                        0.0, device=self.env.unwrapped.device, dtype=torch.float32
                    )

        return self.env.step(actions)


class KlaskRlCollisionAvoidanceWrapper(Wrapper):
    real_to_sim_factor_long_side = 0.0008285
    real_to_sim_factor_short_side = 1 / 1150
    DEACCELERATION_DISTANCE = 0.09
    PEG_RADIUS = 0.0075
    X_EDGE = (-0.16, 0.16)
    Y_EDGE_PLAYER = (-0.03, -0.22)
    Y_EDGE_OPPONENT = (-0.03, -0.22)
    board_dimensions = (0.32, 0.44)
    speed_limit_weight = 70.0

    def __init__(self, env, max_vel=0.2):
        super().__init__(env)

        self.x_min, self.x_max = 15.0, 360.0
        self.y_min_1, self.y_max_1 = 15.0, 235.0
        self.y_min_2, self.y_max_2 = 340.0, 530.0
        self.MAX_VEL = max_vel

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, info

    def step(self, actions, *args, **kwargs):
        left_zone = self.state_1[:, 0] <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        soft_dist_left = self.state_1[left_zone, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        actions[left_zone, 0] = torch.maximum(actions[left_zone, 0], -self.interpolate_vel(soft_dist_left))

        left_zone_opp = self.state_2[:, 0] <= self.X_EDGE[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        soft_dist_left_opp = self.state_2[left_zone_opp, 0] - self.X_EDGE[0] - self.PEG_RADIUS
        actions[left_zone_opp, 2] = torch.maximum(actions[left_zone_opp, 2], -self.interpolate_vel(soft_dist_left_opp))

        right_zone = self.state_1[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.X_EDGE[1]
        actions[right_zone, 0] = torch.minimum(
            actions[right_zone, 0],
            self.interpolate_vel(self.X_EDGE[1] - self.state_1[right_zone, 0] - self.PEG_RADIUS),
        )

        right_zone_opp = self.state_2[:, 0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.X_EDGE[1]
        actions[right_zone_opp, 1] = torch.minimum(
            actions[right_zone_opp, 1],
            self.interpolate_vel(self.X_EDGE[1] - self.state_2[right_zone_opp, 0] - self.PEG_RADIUS),
        )

        bottom_zone = self.state_1[:, 1] <= self.Y_EDGE_PLAYER[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        actions[bottom_zone, 1] = torch.maximum(
            actions[bottom_zone, 1],
            -self.interpolate_vel(self.state_1[:, 1][bottom_zone] - self.Y_EDGE_PLAYER[0] - self.PEG_RADIUS),
        )

        bottom_zone_opp = self.state_2[:, 1] <= self.Y_EDGE_OPPONENT[0] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS
        actions[bottom_zone_opp, 1] = torch.maximum(
            actions[bottom_zone_opp, 1],
            -self.interpolate_vel(self.state_2[:, 1][bottom_zone_opp] - self.Y_EDGE_OPPONENT[0] - self.PEG_RADIUS),
        )

        # Y - TOP
        top_zone = self.state_1[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.Y_EDGE_PLAYER[1]
        actions[top_zone, 1] = torch.minimum(
            actions[top_zone, 1],
            self.interpolate_vel(self.Y_EDGE_PLAYER[1] - self.state_1[:, 1][top_zone] + self.PEG_RADIUS),
        )

        top_zone_opp = self.state_2[:, 1] + self.DEACCELERATION_DISTANCE + self.PEG_RADIUS >= self.Y_EDGE_OPPONENT[1]
        actions[top_zone_opp, 1] = torch.minimum(
            actions[top_zone_opp, 1],
            self.interpolate_vel(self.Y_EDGE_OPPONENT[1] - self.state_2[:, 1][top_zone_opp] + self.PEG_RADIUS),
        )

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        self.state_1 = obs["policy"].clone()[:, :2]
        self.state_2 = obs["opponent"].clone()[:, :2]

        return obs, rew, terminated, truncated, info

    def interpolate_vel(self, distance):
        return (self.MAX_VEL / self.DEACCELERATION_DISTANCE) * distance


class ActionHistoryWrapper(Wrapper):
    def __init__(self, env, history_length):
        super().__init__(env)
        self.history_length = history_length
        self.history_x_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_player = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_x_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)
        self.history_y_opponent = torch.zeros(env.unwrapped.num_envs, history_length).to(env.unwrapped.device)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions, *args, **kwargs):
        self.history_x_player[:, :-1] = self.history_x_player.clone()[:, 1:]
        self.history_x_player[:, -1] = actions[:, 0]
        self.history_y_player[:, :-1] = self.history_y_player.clone()[:, 1:]
        self.history_y_player[:, -1] = actions[:, 1]
        self.history_x_opponent[:, :-1] = self.history_x_opponent.clone()[:, 1:]
        self.history_x_opponent[:, -1] = actions[:, 2]
        self.history_y_opponent[:, :-1] = self.history_y_opponent.clone()[:, 1:]
        self.history_y_opponent[:, -1] = actions[:, 3]

        obs, rew, terminated, truncated, info = self.env.step(actions, *args, **kwargs)
        obs["policy"][:, 12:] = torch.cat([self.history_x_player, self.history_y_player], dim=1)
        obs["opponent"][:, 12:] = torch.cat([self.history_x_opponent, self.history_y_opponent], dim=1)
        return obs, rew, terminated, truncated, info
