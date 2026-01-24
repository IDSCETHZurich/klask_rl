import torch
from gymnasium import Wrapper
from klask_rl.assets.robots.klask import KLASK_PARAMS


class CurriculumWrapper(Wrapper):
    def __init__(self, env, cfg, num_steps=None, mode="train", dynamic=False):
        super().__init__(env)
        self.dynamic = dynamic
        self.cfg = cfg
        self.num_steps = num_steps
        self.mode = mode
        self._step = 0
        for term, weight in cfg.items():
            term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
            if type(weight) is dict:
                if type(weight["weight"]) is list:
                    _weight = weight["weight"][0]
                else:
                    _weight = weight["weight"]
                if not weight.get("per_second", False):
                    _weight /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
            else:
                _weight = weight / (KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"])
            self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = _weight

    def step(self, actions):
        if self.mode == "train":
            self._step += self.env.unwrapped.num_envs
            for term, weight in self.cfg.items():
                if type(weight) is dict and type(weight["weight"]) is list:
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    weight_step = (weight["weight"][1] - weight["weight"][0]) / self.num_steps
                    if not weight.get("per_second", False):
                        weight_step /= KLASK_PARAMS["decimation"] * KLASK_PARAMS["physics_dt"]
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight += weight_step

                if self.dynamic and not (
                    term == "ball_stationary"
                    or term == "time_out_punishment"
                    or term == "time_punishment"
                    or term == "goal_scored"
                    or term == "goal_conceded"
                    or term == "opponent_in_goal"
                    or term == "player_in_goal"
                ):
                    term_idx = self.env.unwrapped.reward_manager.active_terms.index(term)
                    self.env.unwrapped.reward_manager._term_cfgs[term_idx].weight = weight * (
                        torch.exp(
                            -torch.tensor(
                                self._step / 10000000,
                                device=self.env.unwrapped.device,
                                dtype=torch.float32,
                            )
                        )
                    )  # coeff chosen sucht that half the max reward at 20 mio steps
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
