import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.self_play_manager import SelfPlayManager
from rl_games.common.algo_observer import AlgoObserver
from rl_games.torch_runner import Runner, _override_sigma, _restore


class KlaskRlAlgoObserver(AlgoObserver):

    def __init__(self):
        pass

    def after_init(self, algo):
        self.algo = algo
        self.games_to_track = self.algo.games_to_track
        self.mean_scores = torch_ext.AverageMeter(1, self.algo.games_to_track).to(self.algo.ppo_device)
        self.mean_scores.update(
            torch.zeros(self.algo.games_to_track, dtype=float).to(self.algo.ppo_device)
        )  # in order to not have the bump in the scores when no episode has yet terminated
        self.ep_infos = []
        self.direct_info = {}
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if not isinstance(infos, dict):
            classname = self.__class__.__name__
            raise ValueError(f"{classname} expected 'infos' as dict. Received: {type(infos)}")
        # store episode information
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        # log other variables directly
        if len(infos) > 0 and isinstance(infos, dict):  # allow direct logging from env
            self.direct_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v
                if k == "episode":
                    # Values are fractions (mean over all envs), convert to counts
                    # by multiplying by num_envs so we push the right number of
                    # entries into the rolling score buffer.
                    num_envs = self.algo.num_actors
                    for key, val in v.items():
                        if isinstance(val, torch.Tensor):
                            val = val.item()
                        count = max(1, round(float(val) * num_envs))
                        if key == "Episode_Termination/goal_scored":
                            self.mean_scores.update(torch.ones(count, dtype=torch.float).to(self.algo.ppo_device))
                        elif key == "Episode_Termination/goal_conceded":
                            self.mean_scores.update(-torch.ones(count, dtype=torch.float).to(self.algo.ppo_device))
                        elif key == "Episode_Termination/player_in_goal":
                            self.mean_scores.update(-torch.ones(count, dtype=torch.float).to(self.algo.ppo_device))
                        elif key == "Episode_Termination/opponent_in_goal":
                            self.mean_scores.update(torch.ones(count, dtype=torch.float).to(self.algo.ppo_device))
                        elif key == "Episode_Termination/time_out":
                            self.mean_scores.update(torch.zeros(count, dtype=torch.float).to(self.algo.ppo_device))

    def after_clear_stats(self):
        # clear stored buffers
        self.mean_scores.clear()
        self.mean_scores.update(
            torch.zeros(self.algo.games_to_track, dtype=torch.float).to(self.algo.ppo_device)
        )  # in order to not have the bump in the scores when no episode has yet terminated

    def after_print_stats(self, frame, epoch_num, total_time):
        # log scalars from the episode
        if self.ep_infos:
            for key in self.ep_infos[0]:
                info_tensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    info_tensor = torch.cat((info_tensor, ep_info[key].to(self.algo.device)))
                value = torch.mean(info_tensor)
                self.writer.add_scalar(key.replace("_", "/", 1), value, epoch_num)
            self.ep_infos.clear()
        # log scalars from env information
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}", v, frame)
        # log mean reward/score from the env
        if self.mean_scores.current_size > 0:
            mean_scores = self.mean_scores.get_mean()
            self.writer.add_scalar("scores/mean", mean_scores, frame)

        # log current reward weights
        rm = self.algo.vec_env.env.unwrapped.reward_manager
        for term, cfg in zip(rm.active_terms, rm._term_cfgs):
            w = cfg.weight
            val = w.item() if isinstance(w, torch.Tensor) else float(w)
            self.writer.add_scalar(f"rewards/weights/{term}", val, frame)

        # log step counters as explicit metrics so any can be used as x-axis
        self.writer.add_scalar("step/env_frames", frame, frame)
        self.writer.add_scalar("step/training_iteration", epoch_num, frame)
        self.writer.add_scalar("step/wall_time", total_time, frame)


class KlaskRlSelfPlayManager(SelfPlayManager):

    def update(self, algo):
        self.updates_num += 1
        if self.check_scores:
            data = algo.algo_observer.mean_scores
        else:
            data = algo.game_rewards

        if algo.vec_env.training_curriculum:
            algo.vec_env.should_update_agents(algo.get_weights())

        if len(data) >= self.games_to_check:
            mean_scores = data.get_mean()
            mean_rewards = algo.game_rewards.get_mean()
            if mean_scores > self.update_score:
                print("Mean scores: ", mean_scores, " mean rewards: ", mean_rewards, " updating weights")

                algo.clear_stats()
                self.writter.add_scalar("selfplay/iters_update_weigths", self.updates_num, algo.frame)
                algo.vec_env.set_weights(self.env_indexes, algo.get_weights())
                self.env_indexes = (self.env_indexes + 1) % (algo.num_actors)
                self.updates_num = 0


class KlaskRlRunner(Runner):

    def run_train(self, args):
        """Run the training procedure from the algorithm passed in.

        Args:
            args (:obj:`dict`): Train specific args passed in as a dict obtained from a yaml file or some other config format.

        """
        print("Started to train")
        agent = self.algo_factory.create(self.algo_name, base_name="run", params=self.params)
        _restore(agent, args)
        _override_sigma(agent, args)

        if agent.has_self_play_config:
            agent.self_play_manager = KlaskRlSelfPlayManager(agent.self_play_config, agent.writer)
        agent.train()
