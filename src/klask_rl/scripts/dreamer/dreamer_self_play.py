"""Dreamer-compatible self-play wrapper for KLASK.

This wrapper replaces the random opponent with a frozen copy of the training
agent.  It maintains its own RSSM state for the opponent so that the opponent
can act autoregressively just like the player.

The opponent weights are only updated when the agent's rolling win-rate
exceeds a configurable threshold (``update_score``), mirroring the
``SelfPlayManager`` logic from rl_games.

Usage in ``train_dreamer.py``::

    wrapper = DreamerSelfPlayWrapper(isaac_env, update_score=0.7)
    # ... later, after the Dreamer agent is created:
    wrapper.set_opponent(agent)

Call ``wrapper.maybe_update_opponent(agent)`` periodically (e.g. at
evaluation time).  It will only copy weights when the score threshold is
exceeded.
"""

import copy
from collections import deque

import gymnasium as gym
import torch
from gymnasium import Wrapper


class DreamerSelfPlayWrapper(Wrapper):
    """Gymnasium wrapper: opponent actions come from a frozen Dreamer agent.

    The wrapper:
      - Halves the action space (player only, opponent is internal).
      - Mirrors the ``"opponent"`` observation for the opponent agent.
      - On ``step()``, runs the opponent through its own encoder → RSSM → actor
        and concatenates the actions.
      - On ``reset()`` or episode-done, re-initialises the opponent RSSM state.
      - Tracks episode outcomes (+1 win, -1 loss, 0 draw) in a rolling buffer
        and gates opponent weight updates on the mean score exceeding a threshold.

    Parameters
    ----------
    env : gymnasium.Env
        The KLASK environment (after actuator / curriculum / termination wrappers).
    eval_mode : bool
        If True the opponent always uses the mode of the action distribution
        (deterministic); otherwise it samples (stochastic).
    update_score : float
        Mean-score threshold that must be exceeded before the opponent's weights
        are updated.  Matches ``self_play_config.update_score`` in rl_games.
    games_to_track : int
        Size of the rolling window used to compute the mean score.
        Matches ``games_to_track`` in rl_games.
    """

    def __init__(
        self,
        env,
        eval_mode: bool = False,
        update_score: float = 0.7,
        games_to_track: int = 4096,
    ):
        super().__init__(env)
        self._eval_mode = eval_mode

        # --- Score tracking (mirrors rl_games SelfPlayManager) ---
        self._update_score = update_score
        self._games_to_track = games_to_track
        self._score_buffer: deque[float] = deque(maxlen=games_to_track)
        # Pre-fill with zeros so we don't get a premature update from
        # a small sample — same approach as KlaskRlAlgoObserver.
        for _ in range(games_to_track):
            self._score_buffer.append(0.0)

        # Opponent state — populated by ``set_opponent()``.
        self._opponent_encoder = None
        self._opponent_rssm = None
        self._opponent_actor = None
        self._opp_stoch = None
        self._opp_deter = None
        self._opp_prev_action = None
        self._device = None

    # ------------------------------------------------------------------
    # Public API called from the training script
    # ------------------------------------------------------------------

    def set_opponent(self, agent):
        """Initialise the opponent from a Dreamer agent (deep-copies weights).

        Must be called once after the agent has been created and moved to device.
        """
        self._device = agent.device
        self._copy_weights(agent)
        num_envs = self.env.unwrapped.num_envs
        self._reset_opponent_state(num_envs)

    def update_opponent(self, agent):
        """Unconditionally copy the current training agent weights."""
        self._copy_weights(agent)

    def maybe_update_opponent(self, agent, logger=None, train_step=None):
        """Update opponent weights only if the mean score exceeds the threshold.

        Returns True if the update was performed.
        """
        mean_score = self.mean_score
        if logger is not None and train_step is not None:
            logger.scalar("selfplay/mean_score", mean_score)
        if mean_score > self._update_score:
            self._copy_weights(agent)
            # Reset the score buffer after updating so the agent must
            # re-establish dominance against the stronger opponent.
            self._score_buffer.clear()
            for _ in range(self._games_to_track):
                self._score_buffer.append(0.0)
            print(
                f"[step {train_step}] Self-play opponent updated (mean_score={mean_score:.3f} > {self._update_score})"
            )
            if logger is not None and train_step is not None:
                logger.scalar("selfplay/opponent_updated", 1.0)
            return True
        print(
            f"[step {train_step}] Self-play opponent NOT updated (mean_score={mean_score:.3f} <= {self._update_score})"
        )
        if logger is not None and train_step is not None:
            logger.scalar("selfplay/opponent_updated", 0.0)
        return False

    @property
    def mean_score(self) -> float:
        """Current rolling mean score (higher = agent is winning more)."""
        if len(self._score_buffer) == 0:
            return 0.0
        return sum(self._score_buffer) / len(self._score_buffer)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        if self._opponent_encoder is not None:
            num_envs = self.env.unwrapped.num_envs
            self._reset_opponent_state(num_envs)
            # Encode the reset observation so the opponent has seen the
            # initial state before its first action — matches the rl_games
            # flow where OpponentObservationWrapper captures obs on reset.
            is_first = torch.ones(num_envs, 1, dtype=torch.bool, device=self._device)
            self._encode_opponent_obs(obs, is_first)
        return obs, info

    def step(self, action, *args, **kwargs):
        if self._opponent_encoder is None:
            # No opponent loaded yet — fall back to random actions.
            opponent_actions = 2 * torch.rand_like(action) - 1
        else:
            opponent_actions = self._get_opponent_action()

        full_action = torch.cat([action, opponent_actions], dim=1)
        obs, reward, terminated, truncated, info = self.env.step(full_action, *args, **kwargs)

        # Update opponent RSSM state with the new opponent observation.
        done = terminated | truncated
        if self._opponent_encoder is not None:
            is_first = done.unsqueeze(-1).to(torch.bool)
            self._encode_opponent_obs(obs, is_first)
            # Re-initialise state for environments that just finished.
            if done.any():
                init_stoch, init_deter = self._opponent_rssm.initial(done.sum().item())
                self._opp_stoch[done] = init_stoch
                self._opp_deter[done] = init_deter
                self._opp_prev_action[done] = 0.0

        # --- Track episode outcomes for score-gated updates ---
        if done.any():
            self._record_episode_scores(done)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy_weights(self, agent):
        """Deep-copy encoder, RSSM, and actor from the training agent."""
        self._opponent_encoder = copy.deepcopy(agent.encoder)
        self._opponent_rssm = copy.deepcopy(agent.rssm)
        self._opponent_actor = copy.deepcopy(agent.actor)
        for module in (self._opponent_encoder, self._opponent_rssm, self._opponent_actor):
            module.eval()
            for p in module.parameters():
                p.requires_grad_(False)

    def _record_episode_scores(self, done):
        """Record +1 / -1 / 0 for finished episodes based on termination type.

        Reads the termination manager directly (same data source as
        ``KlaskRlAlgoObserver`` in the rl_games pipeline).

        Score mapping (mirrors ``KlaskRlAlgoObserver.process_infos``):
          - goal_scored      → +1  (player scored)
          - opponent_in_goal → +1  (opponent fell into its own goal)
          - goal_conceded    → -1  (opponent scored)
          - player_in_goal   → -1  (player fell into its own goal)
          - time_out         →  0  (draw)
        """
        term_mgr = self.env.unwrapped.termination_manager
        done_mask = done.bool()

        # Map termination term names to scores.
        score_map = {
            "goal_scored": 1.0,
            "opponent_in_goal": 1.0,
            "goal_conceded": -1.0,
            "player_in_goal": -1.0,
            "time_out": 0.0,
        }

        # Iterate over finished envs and record the score for whichever
        # termination was active.
        for term_name, score in score_map.items():
            if term_name not in term_mgr.active_terms:
                continue
            term_active = term_mgr.get_term(term_name)  # (num_envs,) bool
            # Count envs where this termination fired AND the episode ended.
            count = int((term_active & done_mask).sum().item())
            for _ in range(count):
                self._score_buffer.append(score)

    def _reset_opponent_state(self, num_envs):
        """Reset RSSM state for all environments."""
        stoch, deter = self._opponent_rssm.initial(num_envs)
        self._opp_stoch = stoch
        self._opp_deter = deter
        # The opponent action dimension is 2 (same as player).
        self._opp_prev_action = torch.zeros(num_envs, 2, dtype=torch.float32, device=self._device)

    @torch.no_grad()
    def _get_opponent_action(self):
        """Run the opponent's actor on the current RSSM state."""
        feat = self._opponent_rssm.get_feat(self._opp_stoch, self._opp_deter)
        action_dist = self._opponent_actor(feat)
        action = action_dist.mode if self._eval_mode else action_dist.rsample()
        self._opp_prev_action = action
        return action

    @torch.no_grad()
    def _encode_opponent_obs(self, obs, is_first):
        """Advance opponent RSSM state using the latest opponent observation.

        The opponent observations are already rotated to player frame by the
        ObservationsCfg, so no additional negation is needed here.

        Parameters
        ----------
        obs : dict
            The observation dict from the environment.
        is_first : torch.Tensor
            Float tensor of shape ``(num_envs, 1)`` — ``1.0`` for envs that
            just reset, ``0.0`` otherwise.
        """

        # Build an obs dict compatible with the encoder.
        opp_obs_dict = {}
        opp_obs_dict["policy"] = obs["opponent"]
        opp_obs_dict["image"] = obs["opponent_image"]

        # Preprocess (image normalisation).
        opp_obs_dict["image"] = opp_obs_dict["image"].float() / 255.0

        # Encode → RSSM obs step.
        embed = self._opponent_encoder(opp_obs_dict)

        self._opp_stoch, self._opp_deter, _ = self._opponent_rssm.obs_step(
            self._opp_stoch,
            self._opp_deter,
            self._opp_prev_action,
            embed,
            is_first,
        )
