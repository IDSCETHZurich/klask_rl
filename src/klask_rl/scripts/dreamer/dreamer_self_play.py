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
        compile: bool = False,
    ):
        super().__init__(env)
        self._eval_mode = eval_mode
        self._compile = compile

        # --- Halve the action space (player only) ---
        if hasattr(self.env.unwrapped, "single_action_space"):
            original_space = self.env.unwrapped.single_action_space
            if hasattr(original_space, "shape") and original_space.shape[0] == 4:
                self.env.unwrapped._klask_original_single_action_space = original_space
                self.env.unwrapped.single_action_space = gym.spaces.Box(
                    low=original_space.low[:2],
                    high=original_space.high[:2],
                    dtype=original_space.dtype,
                )

        # --- Score tracking (mirrors rl_games SelfPlayManager) ---
        self._update_score = update_score
        self._games_to_track = games_to_track
        self._score_buffer: deque[float] = deque(maxlen=games_to_track)
        # Pre-fill with zeros so we don't get a premature update from
        # a small sample — same approach as KlaskRlAlgoObserver.
        for _ in range(games_to_track):
            self._score_buffer.append(0.0)

        # Opponent state — populated by ``set_opponent()``.
        # _opponent_{enc,rssm,actor} are the inference-facing modules (may be
        # torch.compile-wrapped).  _opponent_{enc,rssm,actor}_orig are always
        # the raw nn.Module instances used for in-place weight updates.
        self._opponent_encoder = None
        self._opponent_rssm = None
        self._opponent_actor = None
        self._opponent_encoder_orig = None
        self._opponent_rssm_orig = None
        self._opponent_actor_orig = None
        self._opp_stoch = None
        self._opp_deter = None
        self._opp_prev_action = None
        self._device = None
        self._opp_is_first = None
        self._opp_pending_first = None

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
            self._opp_is_first = torch.ones(num_envs, dtype=torch.bool, device=self._device)
            self._opp_pending_first = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
        return obs, info

    def step(self, action, *args, **kwargs):
        if self._opponent_encoder is None:
            # No opponent loaded yet — fall back to random actions.
            opponent_actions = 2 * torch.rand_like(action) - 1
        else:
            opponent_actions = self._get_opponent_action()

        full_action = torch.cat([action, opponent_actions], dim=1)
        obs, reward, terminated, truncated, info = self.env.step(full_action, *args, **kwargs)

        # Expose opponent actions so they can be stored in the replay buffer.
        obs["opponent_action"] = opponent_actions

        # Update opponent RSSM state with the new opponent observation.
        done = terminated | truncated
        if self._opponent_encoder is not None:
            # Build 4D prev_action for the opponent's RSSM when opponent_separation is on.
            # From the opponent's perspective: [opp_action, player_action].
            if self._opponent_rssm._act_dim > opponent_actions.shape[-1]:
                self._opp_prev_action = torch.cat([opponent_actions, action], dim=-1)
            obs_for_encode = self._with_terminal_opponent_obs(obs, info)
            if self._opp_is_first is None:
                num_envs = self.env.unwrapped.num_envs
                self._opp_is_first = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
                self._opp_pending_first = torch.zeros(num_envs, dtype=torch.bool, device=self._device)
            is_first = (self._opp_is_first | self._opp_pending_first).unsqueeze(-1)
            self._encode_opponent_obs(obs_for_encode, is_first)
            self._opp_is_first.zero_()
            self._opp_pending_first.copy_(done.to(torch.bool))

        # --- Track episode outcomes for score-gated updates ---
        if done.any():
            self._record_episode_scores(done)

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _copy_weights(self, agent):
        """Copy encoder, RSSM, and actor weights from the training agent.

        First call: deep-copies the modules to create correctly-shaped tensors
        on the right device, then optionally wraps them with torch.compile for
        faster inference.

        Subsequent calls: updates weights in-place via load_state_dict on the
        uncompiled originals — much faster than deepcopy.  torch.compile shares
        the same underlying parameter tensors, so compiled graphs pick up the
        new weights automatically without recompilation.
        """
        if self._opponent_encoder is None:
            # First call — must deepcopy to get architecture / device / dtype.
            enc = copy.deepcopy(agent.encoder)
            rssm = copy.deepcopy(agent.rssm)
            actor = copy.deepcopy(agent.actor)
            for module in (enc, rssm, actor):
                module.eval()
                for p in module.parameters():
                    p.requires_grad_(False)
            # Keep uncompiled originals for efficient in-place weight updates.
            self._opponent_encoder_orig = enc
            self._opponent_rssm_orig = rssm
            self._opponent_actor_orig = actor
            if self._compile:
                self._opponent_encoder = torch.compile(enc, mode="reduce-overhead")
                self._opponent_rssm = torch.compile(rssm, mode="reduce-overhead")
                self._opponent_actor = torch.compile(actor, mode="reduce-overhead")
            else:
                self._opponent_encoder = enc
                self._opponent_rssm = rssm
                self._opponent_actor = actor
        else:
            # Subsequent calls — update parameters in-place; avoids deepcopy
            # overhead and preserves any torch.compile wrapping.
            self._opponent_encoder_orig.load_state_dict(agent.encoder.state_dict())
            self._opponent_rssm_orig.load_state_dict(agent.rssm.state_dict())
            self._opponent_actor_orig.load_state_dict(agent.actor.state_dict())

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

        # Accumulate per-env scores on GPU, then do a single GPU→CPU transfer.
        # This avoids up to 5 separate .item() syncs (one per term) in the original loop.
        env_scores = torch.zeros(done_mask.shape[0], dtype=torch.float32, device=done_mask.device)
        for term_name, score in score_map.items():
            if term_name not in term_mgr.active_terms:
                continue
            term_active = term_mgr.get_term(term_name)  # (num_envs,) bool
            env_scores += (term_active & done_mask).float() * score

        for s in env_scores[done_mask].tolist():
            self._score_buffer.append(s)

    def _reset_opponent_state(self, num_envs):
        """Reset RSSM state for all environments."""
        stoch, deter = self._opponent_rssm.initial(num_envs)
        self._opp_stoch = stoch
        self._opp_deter = deter
        self._opp_prev_action = torch.zeros(
            num_envs, self._opponent_rssm._act_dim, dtype=torch.float32, device=self._device
        )

    @torch.no_grad()
    def _get_opponent_action(self):
        """Run the opponent's actor on the current RSSM state."""
        feat = self._opponent_rssm.get_feat(self._opp_stoch, self._opp_deter)
        action_dist = self._opponent_actor(feat)
        action = action_dist.mode if self._eval_mode else action_dist.rsample()
        # Don't set _opp_prev_action here when opponent_separation is on;
        # step() will build the full 4D prev_action after both actions are known.
        if self._opponent_rssm._act_dim == action.shape[-1]:
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

    def _with_terminal_opponent_obs(self, obs, info):
        """Swap terminal observations for opponent keys on done envs when available."""
        if not isinstance(info, dict):
            return obs
        terminal_obs = info.get("terminal_obs")
        terminal_env_ids = info.get("terminal_env_ids")
        if terminal_obs is None or terminal_env_ids is None:
            return obs

        env_ids = torch.as_tensor(terminal_env_ids, device=self._device, dtype=torch.long)
        if env_ids.numel() == 0:
            return obs

        out_obs = dict(obs)
        for key in ("opponent", "opponent_image"):
            if key in out_obs and key in terminal_obs:
                out_obs[key] = out_obs[key].clone()
                out_obs[key][env_ids] = terminal_obs[key].to(out_obs[key].device)
        return out_obs
