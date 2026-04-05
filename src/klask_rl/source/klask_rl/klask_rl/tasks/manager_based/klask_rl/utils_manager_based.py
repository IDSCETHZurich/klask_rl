import cv2
import isaaclab.utils.math as math_utils
import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, RayCasterCamera, TiledCamera
from klask_rl.assets.robots.klask import KLASK_PARAMS


def _pad_image_to_target(images: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Zero-pad image tensor (N, H, W, C) to (N, target_h, target_w, C)."""
    _, h, w, _ = images.shape
    pad_bottom = target_h - h
    pad_right = target_w - w
    if pad_bottom > 0 or pad_right > 0:
        # For (N, H, W, C): pad W on the right and H on the bottom.
        images = F.pad(images, (0, 0, 0, pad_right, 0, pad_bottom), value=0)
    return images


def padded_image(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    target_h: int = 64,
    target_w: int = 64,
) -> torch.Tensor:
    """Return a raw (unnormalized) camera image zero-padded to (target_h, target_w).

    Padding is applied to the right and bottom so that the original image
    occupies the top-left corner.  The result has shape (N, target_h, target_w, C)
    with the same dtype as the sensor output (typically uint8 for RGB).
    """
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  # (N, H, W, C)
    return _pad_image_to_target(images, target_h, target_w)


def padded_image_rotated(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
    data_type: str = "rgb",
    target_h: int = 64,
    target_w: int = 64,
) -> torch.Tensor:
    """Return a 180°-rotated camera image zero-padded to (target_h, target_w).

    This provides the opponent's perspective by rotating the player camera
    image by 180° (equivalent to ``torch.rot90(…, k=2)``) without requiring
    a second physical camera in the scene.
    """
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]
    images = sensor.data.output[data_type]  # (N, H, W, C)
    # 180° rotation on the spatial (H, W) dimensions before padding
    images = torch.rot90(images, k=2, dims=[1, 2])
    return _pad_image_to_target(images, target_h, target_w)


# ---------------------------------------------------------------------------
# Sprite-rendered image observations
# ---------------------------------------------------------------------------

# Board aspect ratio: 420mm x 320mm = 21:16 base ratio (same as env_cfg_utils.py).
_BOARD_BASE_H = 21
_BOARD_BASE_W = 16

# Lazy renderer cache: keyed by (sprite_dir, bg_path, out_w, out_h)
_renderer_cache: dict = {}
# Per-step image cache so the opponent view can reuse the player's render.
_sprite_image_cache: dict = {}


def _sprite_camera_params(image_size: int) -> tuple[int, int]:
    """Derive (cam_h, cam_w) from a square image size using the board aspect ratio."""
    scale = min(image_size // _BOARD_BASE_H, image_size // _BOARD_BASE_W)
    return scale * _BOARD_BASE_H, scale * _BOARD_BASE_W


def _get_renderer(sprite_dir: str, background_path: str, cam_w: int, cam_h: int):
    """Return a cached BoardRenderer, creating one on first call."""
    key = (sprite_dir, background_path, cam_w, cam_h)
    if key not in _renderer_cache:
        import importlib.util
        import os

        # board_renderer.py lives two directories above sprite_dir
        # (sprite_dir = .../assets/sprites/, renderer = .../sprite_renderer/board_renderer.py).
        renderer_path = os.path.join(os.path.dirname(os.path.dirname(sprite_dir)), "board_renderer.py")
        spec = importlib.util.spec_from_file_location("board_renderer", renderer_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        BoardRenderer = mod.BoardRenderer

        _renderer_cache[key] = BoardRenderer(
            sprite_dir=sprite_dir,
            background_path=background_path,
            output_size=(cam_w, cam_h),
            fast_mode=False,
            target_frame="sim",
        )
    return _renderer_cache[key]


def sprite_rendered_image(
    env: ManagerBasedRLEnv,
    peg1_cfg: SceneEntityCfg,
    peg2_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    target_h: int = 128,
    target_w: int = 128,
) -> torch.Tensor:
    """Render sprite-based board images from sim state positions.

    Returns a ``(N, target_h, target_w, 3)`` uint8 RGB tensor on the env device.
    The unpadded render is cached so ``sprite_rendered_image_rotated`` can reuse it.
    """
    cfg = env.cfg
    cam_h, cam_w = _sprite_camera_params(min(target_h, target_w))
    renderer = _get_renderer(cfg.sprite_dir, cfg.background_path, cam_w, cam_h)

    # Extract positions from scene (sim frame, origin at board centre)
    peg1_pos = body_xy_pos_w(env, peg1_cfg)  # (N, 2) GPU tensor
    peg2_pos = body_xy_pos_w(env, peg2_cfg)  # (N, 2) GPU tensor
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2) GPU tensor

    # Transfer to CPU once (batch)
    peg1_np = peg1_pos.cpu().numpy()
    peg2_np = peg2_pos.cpu().numpy()
    ball_np = ball_pos.cpu().numpy()

    # Render per env, collect RGB images
    N = env.num_envs
    images_np = np.empty((N, cam_h, cam_w, 3), dtype=np.uint8)
    for i in range(N):
        bgr = renderer.render(peg1_np[i], peg2_np[i], ball_np[i])
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=images_np[i])

    # To GPU tensor and cache for opponent view
    tensor = torch.from_numpy(images_np).to(env.device)
    _sprite_image_cache.clear()
    _sprite_image_cache["images"] = tensor

    return _pad_image_to_target(tensor, target_h, target_w)


def sprite_rendered_image_rotated(
    env: ManagerBasedRLEnv,
    peg1_cfg: SceneEntityCfg,
    peg2_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    target_h: int = 128,
    target_w: int = 128,
) -> torch.Tensor:
    """Return 180-degree-rotated sprite image for the opponent view.

    Reuses the cached render from ``sprite_rendered_image`` to avoid rendering
    twice per step.
    """
    if "images" not in _sprite_image_cache:
        raise RuntimeError(
            "sprite_rendered_image_rotated called before sprite_rendered_image. "
            "Ensure the 'image' observation group is defined before 'opponent_image' "
            "in DreamerSpriteObservationsCfg so rendering happens first."
        )
    images = _sprite_image_cache.pop("images")
    rotated = torch.rot90(images, k=2, dims=[1, 2])
    return _pad_image_to_target(rotated, target_h, target_w)


def set_joint_position_limits(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    lower: float,
    upper: float,
):
    """Override joint position limits in simulation for the specified joints.

    Useful when the USD asset has incorrect limits and they cannot be edited.
    Writes the given lower/upper bounds to PhysX for all environments.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    limits = asset.data.joint_pos_limits.clone()
    limits[:, joint_ids, 0] = lower
    limits[:, joint_ids, 1] = upper
    asset.write_joint_position_limit_to_sim(limits[:, joint_ids, :], joint_ids=joint_ids, warn_limit_violation=False)


def set_rigid_body_material(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    static_friction: float,
    dynamic_friction: float,
    restitution: float,
):
    """Set rigid body material properties to fixed values.

    Unlike ``randomize_rigid_body_material``, this simply writes the given
    values to the PhysX material buffer — no sampling, no buckets.
    Respects ``asset_cfg.body_ids`` so it can target a subset of bodies.
    """
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    materials = asset.root_physx_view.get_material_properties()
    all_ids = torch.arange(env.scene.num_envs, device="cpu")

    if isinstance(asset, Articulation) and asset_cfg.body_ids != slice(None):
        num_shapes_per_body = []
        for link_path in asset.root_physx_view.link_paths[0]:
            link_view = asset._physics_sim_view.create_rigid_body_view(link_path)
            num_shapes_per_body.append(link_view.max_shapes)
        for body_id in asset_cfg.body_ids:
            start_idx = sum(num_shapes_per_body[:body_id])
            end_idx = start_idx + num_shapes_per_body[body_id]
            materials[:, start_idx:end_idx, 0] = static_friction
            materials[:, start_idx:end_idx, 1] = dynamic_friction
            materials[:, start_idx:end_idx, 2] = restitution
    else:
        materials[:, :, 0] = static_friction
        materials[:, :, 1] = dynamic_friction
        materials[:, :, 2] = restitution

    asset.root_physx_view.set_material_properties(materials, all_ids)


def reset_player_velocity_toward_ball(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Initialize player velocity toward the ball at episode reset.

    Reads ``env._init_velocity_speed`` (set by ``InitializationWrapper`` before
    calling ``env.reset()``) and writes the computed velocity into the physics
    simulation for the environments being reset.

    Running inside the event manager guarantees that the velocity is written
    *before* observations are read, so ``ActuatorModelWrapper`` sees the correct
    non-zero velocity in its state history buffer.

    If ``_init_velocity_speed`` is 0.0 (the default when the wrapper is absent),
    this function is a no-op.
    """
    speed = getattr(env, "_init_velocity_speed", 0.0)
    if speed <= 0.0:
        return

    klask_art: Articulation = env.scene["klask"]
    ball: RigidObject = env.scene["ball"]

    # Cache joint IDs lazily on first call.
    if not hasattr(env, "_init_vel_joint_ids"):
        x_ids, _ = klask_art.find_joints(["slider_to_peg_1"])
        y_ids, _ = klask_art.find_joints(["ground_to_slider_1"])
        env._init_vel_joint_ids = (x_ids[0], y_ids[0])
    x_id, y_id = env._init_vel_joint_ids

    # Player XY position from joint positions (prismatic joints; default = 0 = board centre).
    # joint_pos is updated by write_joint_state_to_sim in the earlier position-reset events.
    player_x = klask_art.data.joint_pos[env_ids, x_id]  # slider_to_peg_1 → X
    player_y = klask_art.data.joint_pos[env_ids, y_id]  # ground_to_slider_1 → Y

    # Ball XY relative to the environment origin.
    # root_pos_w is updated by mdp.reset_root_state_uniform in the ball-reset event.
    ball_xy = ball.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]

    direction_x = ball_xy[:, 0] - player_x
    direction_y = ball_xy[:, 1] - player_y
    norm = torch.sqrt(direction_x**2 + direction_y**2).clamp(min=1e-6)

    vel_2d = torch.stack([(direction_x / norm) * speed, (direction_y / norm) * speed], dim=1)  # [len(env_ids), 2]
    klask_art.write_joint_velocity_to_sim(vel_2d, joint_ids=[x_id, y_id], env_ids=env_ids)


def reset_ball_hit_tracking(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset the ball hit tracking flag for specified environments.

    This should be called on environment reset to clear the one-time collision
    reward tracking, allowing the agent to earn the reward again in the new episode.

    Args:
        env: The environment instance
        env_ids: The environment IDs to reset
    """
    # Initialize the tracking buffer if it doesn't exist
    if not hasattr(env, "ball_hit_this_episode"):
        env.ball_hit_this_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Reset the flag for the specified environments
    env.ball_hit_this_episode[env_ids] = False


def reset_ball_hit_timer(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
):
    """Reset the ball hit timer for specified environments.

    This should be called on environment reset to clear the timer tracking,
    allowing the timeout termination to work correctly in the new episode.

    Args:
        env: The environment instance
        env_ids: The environment IDs to reset
    """
    # Initialize the timer buffer if it doesn't exist
    if not hasattr(env, "ball_hit_timer"):
        env.ball_hit_timer = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)

    # Reset the timer for the specified environments (-1.0 means not hit yet)
    env.ball_hit_timer[env_ids] = -1.0


def ball_hit_timeout(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    timeout: float = 0.5,
    eps: float = 0.017,
    min_relative_vel: float = 0.08,
    min_ball_speed: float = 0.01,
) -> torch.Tensor:
    """Terminate episode after specified timeout following ball hit.

    This function tracks when the ball is hit and terminates the episode
    after the specified timeout duration. The timer starts on first collision
    and counts down using the simulation timestep.

    Args:
        env: The environment instance
        player_cfg: Configuration for the player entity
        ball_cfg: Configuration for the ball entity
        timeout: Time in seconds after ball hit to terminate (default: 0.5)
        eps: Distance threshold for collision detection (default: 0.017)
        min_relative_vel: Minimum relative velocity to count as hit (default: 0.08 m/s)
        min_ball_speed: Minimum ball speed to count as hit (default: 0.01 m/s)

    Returns:
        Boolean tensor indicating which environments should terminate
    """
    # Initialize the timer buffer if it doesn't exist
    if not hasattr(env, "ball_hit_timer"):
        env.ball_hit_timer = torch.full((env.num_envs,), -1.0, dtype=torch.float32, device=env.device)

    # Check for collision
    collision_now = collision_player_ball_bool(env, player_cfg, ball_cfg, eps, min_relative_vel, min_ball_speed)

    # Start timer on first collision (timer == -1.0 means not started)
    not_started = env.ball_hit_timer < 0.0
    env.ball_hit_timer[collision_now & not_started] = 0.0

    # Update timer for environments where ball was hit
    hit_envs = env.ball_hit_timer >= 0.0
    if hit_envs.any():
        env.ball_hit_timer[hit_envs] += env.step_dt

    # Terminate if timeout reached
    should_terminate = env.ball_hit_timer >= timeout

    return should_terminate


def reset_joints_by_absolute(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Reset the robot joints to an absolute position and velocity sampled from the given ranges.

    Unlike reset_joints_by_offset, this function sets the joint state directly without adding
    the default joint position as a bias.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    joint_pos = math_utils.sample_uniform(*position_range, (len(env_ids), len(asset_cfg.joint_ids)), env.device)
    joint_vel = math_utils.sample_uniform(*velocity_range, (len(env_ids), len(asset_cfg.joint_ids)), env.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids, :]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)


def reset_joints_by_offset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()[:, asset_cfg.joint_ids]
    joint_vel = asset.data.default_joint_vel[env_ids].clone()[:, asset_cfg.joint_ids]

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids, :]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids][:, asset_cfg.joint_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids, joint_ids=asset_cfg.joint_ids)


def in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
    weight: float | None = None,
) -> torch.Tensor:
    """
    Penalize asset being in goal.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # body_name = asset_cfg.body_names[0]

    # Check if asset located in circle
    cx, cy, r = goal
    asset_pos_rel = asset.data.body_pos_w[:, asset_cfg.body_ids, :].squeeze() - env.scene.env_origins
    bodies_in_goal = (asset_pos_rel[:, 0] - cx) ** 2 + (asset_pos_rel[:, 1] - cy) ** 2 <= r**2

    if weight is not None:
        bodies_in_goal *= weight

    return bodies_in_goal


def ball_in_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
    max_ball_vel: float = 2.0,
    weight: float | None = None,
) -> torch.Tensor:
    """
    Penalize asset being in goal.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # body_name = asset_cfg.name

    # Check if ball located inside goal
    cx, cy, r = goal
    ball_pos_rel = asset.data.root_pos_w - env.scene.env_origins
    ball_in_goal = (ball_pos_rel[:, 0] - cx) ** 2 + (ball_pos_rel[:, 1] - cy) ** 2 <= r**2

    # Check if ball is slower than max_vel_ball
    ball_slow = asset.data.root_lin_vel_w[:, 0] ** 2 + asset.data.root_lin_vel_w[:, 1] ** 2 <= max_ball_vel**2

    if weight is None:
        ball_in_goal = ball_in_goal * ball_slow
    else:
        ball_in_goal = weight * ball_in_goal * ball_slow

    return ball_in_goal


def peg_in_defense_line_with_rebounds(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    opponent_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    weight: float | None = None,
) -> torch.Tensor:
    """
    Rewards the player for blocking direct and rebound shot lines from opponent to ball.
    Includes reflections off the four walls.
    """
    # Positions
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    opponent_pos = body_xy_pos_w(env, opponent_cfg)  # (N, 2)
    player_pos = body_xy_pos_w(env, player_cfg)  # (N, 2)

    # Only consider when ball in opp half
    ball_in_opp_half = ball_pos[:, 1] > 0.0
    ball_is_close = distance_player_ball(env, player_cfg, ball_cfg) < 0.08

    # Reflect ball across 4 edges
    mirror_balls = [ball_pos]  # start with original ball

    mirror_opponents = [opponent_pos]
    # Reflect across x walls
    mirror_opponents.append(torch.stack([-0.32 - opponent_pos[:, 0], opponent_pos[:, 1]], dim=1))  # left wall
    mirror_opponents.append(torch.stack([0.32 - opponent_pos[:, 0], opponent_pos[:, 1]], dim=1))  # right wall

    mirror_balls.append(torch.stack([-0.32 - ball_pos[:, 0], ball_pos[:, 1]], dim=1))  # left wall
    mirror_balls.append(torch.stack([0.32 - ball_pos[:, 0], ball_pos[:, 1]], dim=1))  # right wall
    # Reflect across y walls ( this is unlikely )
    # mirror_balls.append(torch.stack([ball_pos[:, 0], -0.44 - ball_pos[:, 1]], dim=1))  # bottom wall
    # mirror_balls.append(torch.stack([ball_pos[:, 0], 0.44 - ball_pos[:, 1]], dim=1))   # top wall

    all_rewards = []

    for mirror_ball, mirror_opp in zip(mirror_balls, mirror_opponents):
        vec_ob = mirror_ball - mirror_opp
        vec_op = player_pos - mirror_opp

        # Dot product (batch)
        dot = torch.sum(vec_ob * vec_op, dim=1)
        norm_ob = torch.norm(vec_ob, dim=1)
        norm_op = torch.norm(vec_op, dim=1)

        # Angle (in radians)
        cos_theta = dot / (norm_ob * norm_op + 1e-6)
        angle = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))  # Clamp for numerical stability

        # Smaller angle = better blocking → higher reward
        reward = torch.exp(-1.0 * angle)  # Adjust 5.0 as needed
        all_rewards.append(reward)
    # Take maximum reward across all paths (best alignment)
    final_reward = torch.stack(all_rewards, dim=1).max(dim=1).values

    # Apply ball-in-own-half condition
    final_reward = final_reward * ball_is_close.float() * ball_in_opp_half.float()

    # Apply weight if needed
    if weight is not None:
        final_reward *= weight

    return final_reward


def root_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]


def root_lin_xy_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, :2]


def body_xy_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body position in the environment frame"""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_pos_w[:, asset_cfg.body_ids, :2].squeeze(dim=1) - env.scene.env_origins[:, :2]


def shot_over_middle(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, weight: float | None = None) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)  # shape: (num_envs, 2)
    ball_vel = root_lin_xy_vel_w(env, ball_cfg)  # shape: (num_envs, 2)

    # Detect near center line and moving forward in +y direction
    is_near_center = (ball_pos[:, 1] >= 0.0) & (ball_pos[:, 1] <= 0.02)
    is_moving_forward = ball_vel[:, 1] > 0.1
    if weight is None:
        return (torch.abs(ball_vel[:, 1]) ** 4 * is_near_center * is_moving_forward).float()
    return weight * (torch.abs(ball_vel[:, 1]) ** 4 * is_near_center * is_moving_forward).float()


def body_lin_xy_vel_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].squeeze(dim=1)


def opponent_goal_obs(env: ManagerBasedRLEnv, goal: tuple[float, float]) -> torch.Tensor:
    return torch.Tensor([*goal, 0.0, 0.0]).repeat(env.num_envs, 1)


def goal_position_obs(env: ManagerBasedRLEnv, goal: tuple[float, float]) -> torch.Tensor:
    """Returns the goal XY position as observation (constant for all envs).

    Args:
        env: The environment
        goal: Tuple of (x, y) goal position

    Returns:
        Tensor of shape (num_envs, 2) with goal XY position
    """
    return torch.tensor([goal[0], goal[1]], dtype=torch.float32, device=env.device).unsqueeze(0).repeat(env.num_envs, 1)


def direction_to_ball(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns the direction vector from player to ball (2D).

    This is the key observation for learning to move towards the ball.
    The agent just needs to learn: action ≈ k * direction_to_ball

    The raw displacement is returned (in meters). With observation normalization enabled,
    this will be normalized by the running mean/std during training.
    """
    ball_pos = root_xy_pos_w(env, ball_cfg)
    player_pos = body_xy_pos_w(env, player_cfg)
    return ball_pos - player_pos


def distance_player_ball(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    return torch.sqrt(torch.sum((root_xy_pos_w(env, ball_cfg) - body_xy_pos_w(env, player_cfg)) ** 2, dim=1))


def proximity_player_ball(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Proximity reward that incentivizes moving towards the ball.

    Uses exponential decay so the gradient is stronger at all distances.
    Returns higher values when closer to the ball.
    """
    dist = distance_player_ball(env, player_cfg, ball_cfg)
    return torch.exp(-10.0 * dist)


def speed(vel: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2)


def ball_speed(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    vel = root_lin_xy_vel_w(env, ball_cfg)
    return speed(vel)


def peg_speed(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg) -> torch.Tensor:
    vel = body_lin_xy_vel_w(env, player_cfg)
    return speed(vel)


def peg_speed_exp(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, sigma: float = 0.3) -> torch.Tensor:
    """Exponential saturation reward for peg speed. Returns ~1.0 when moving, ~0.0 when stationary."""
    vel = body_lin_xy_vel_w(env, player_cfg)
    spd = speed(vel)
    return 1.0 - torch.exp(-spd / sigma)


def difference_speed(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg):
    vel_ball = root_lin_xy_vel_w(env, ball_cfg)
    vel_player = body_lin_xy_vel_w(env, player_cfg)
    diff = vel_player - vel_ball
    return speed(diff)


def distance_ball_goal(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal: tuple[float, float, float]
) -> torch.Tensor:
    cx, cy, r = goal
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return torch.exp(
        -5 * torch.sqrt((ball_pos[:, 0] - cx) ** 2 + (ball_pos[:, 1] - cy) ** 2)
    )  # factor 5 because distances are really small


def distance_player_ball_own_half(
    env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    ball_in_own_half = ball_pos[:, 1] < 0.0
    return ball_in_own_half * (
        torch.exp(-5 * distance_player_ball(env, player_cfg, ball_cfg))
    )  # factor 5 because distances are really small


def ball_stationary(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, eps=5e-3) -> torch.Tensor:
    return ball_speed(env, ball_cfg) < eps


def collision_player_ball(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    eps=0.017,
) -> torch.Tensor:
    return (distance_player_ball(env, player_cfg, ball_cfg) < eps) * torch.exp(
        -0.1 / difference_speed(env, player_cfg, ball_cfg)
    )


def collision_player_ball_bool(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    eps: float = 0.017,
    min_relative_vel: float = 0.08,
    min_ball_speed: float = 0.01,
) -> torch.Tensor:
    """Returns True (bool) if player is colliding with ball.

    Requires both proximity AND relative velocity to detect actual hits,
    preventing false positives from "hovering near the ball".

    Args:
        env: The environment instance
        player_cfg: Configuration for the player entity
        ball_cfg: Configuration for the ball entity
        eps: Distance threshold for collision detection (default: 0.017)
        min_relative_vel: Minimum relative velocity to count as hit (default: 0.08 m/s)
        min_ball_speed: Minimum ball speed to count as hit (default: 0.01 m/s)
    Returns:
        Boolean tensor indicating collision
    """
    dist = distance_player_ball(env, player_cfg, ball_cfg)
    rel_vel = difference_speed(env, player_cfg, ball_cfg)
    ball_vel = ball_speed(env, ball_cfg)
    return (dist < eps) & (rel_vel > min_relative_vel) & (ball_vel > min_ball_speed)


def collision_player_ball_time_decay(
    env: ManagerBasedRLEnv,
    player_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    eps: float = 0.017,
    min_relative_vel: float = 0.08,
    min_ball_speed: float = 0.01,
    decay_type: str = "linear",
    decay_rate: float = 5.0,
) -> torch.Tensor:
    """Returns collision reward that decays with episode progress.

    Rewards faster ball contact - the earlier in the episode the collision happens,
    the higher the reward. Supports both linear and exponential decay.

    IMPORTANT: This reward is given ONLY ONCE per episode. After the first collision,
    subsequent collisions give zero reward. This prevents reward exploitation from
    repeated ball touches.

    Decay modes:
    - "linear": reward = collision * (1 - progress)
      - At t=0%: multiplier = 1.0
      - At t=50%: multiplier = 0.5
      - At t=100%: multiplier = 0.0

    - "exponential": reward = collision * exp(-decay_rate * progress)
      - Maintains higher rewards early, then drops quickly
      - With decay_rate=5.0:
        - At t=0%: multiplier = 1.0
        - At t=50%: multiplier = 0.08
        - At t=100%: multiplier = 0.007
      - Higher decay_rate = steeper drop

    Args:
        env: The environment instance
        player_cfg: Configuration for the player entity
        ball_cfg: Configuration for the ball entity
        eps: Distance threshold for collision detection (default: 0.017)
        min_relative_vel: Minimum relative velocity to count as hit (default: 0.08 m/s)
        min_ball_speed: Minimum ball speed to count as hit (default: 0.01 m/s)
        decay_type: Type of decay - "linear" or "exponential" (default: "linear")
        decay_rate: Rate of exponential decay (only used if decay_type="exponential", default: 5.0)

    Returns:
        Time-decayed collision reward (0.0 to 1.0 range before weight scaling)
        Returns 0.0 if collision already happened this episode.
    """
    # Initialize the tracking buffer if it doesn't exist
    if not hasattr(env, "ball_hit_this_episode"):
        env.ball_hit_this_episode = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Check if collision occurred (with improved detection)
    collision_now = collision_player_ball_bool(env, player_cfg, ball_cfg, eps, min_relative_vel, min_ball_speed)

    # Only reward if collision AND not yet rewarded this episode
    should_reward = collision_now & (~env.ball_hit_this_episode)

    # Calculate episode progress (0.0 at start, 1.0 at max_episode_length)
    progress = env.episode_length_buf.float() / env.max_episode_length

    # Apply decay based on type
    if decay_type == "linear":
        # Linear decay: reward is highest at start, zero at end
        time_multiplier = 1.0 - progress
    elif decay_type == "exponential":
        # Exponential decay: maintains higher rewards early, then drops quickly
        time_multiplier = torch.exp(-decay_rate * progress)
    else:
        raise ValueError(f"Invalid decay_type '{decay_type}'. Must be 'linear' or 'exponential'.")

    # Mark environments that had collision (for future timesteps)
    env.ball_hit_this_episode |= collision_now

    return should_reward.float() * time_multiplier


def termination_reward_time_decay(
    env: ManagerBasedRLEnv,
    termination_term: str,
    decay_type: str = "linear",
    decay_rate: float = 5.0,
) -> torch.Tensor:
    """Returns a time-decayed reward when a specific termination condition is triggered.

    This function provides DETERMINISTIC reward-termination coupling by reading the
    termination signal directly from the termination manager. This guarantees that:
    1. The reward is given if and only if the termination occurred
    2. No race conditions or floating-point discrepancies between reward and termination checks

    NOTE: In IsaacLab's step() function, terminations are computed BEFORE rewards,
    so env.termination_manager contains the current step's termination results.

    Decay modes:
    - "linear": reward = terminated * (1 - progress)
      - At t=0%: multiplier = 1.0
      - At t=50%: multiplier = 0.5
      - At t=100%: multiplier = 0.0

    - "exponential": reward = terminated * exp(-decay_rate * progress)
      - Maintains higher rewards early, then drops quickly

    Args:
        env: The environment instance
        termination_term: Name of the termination term to check (e.g., "ball_hit")
        decay_type: Type of decay - "linear" or "exponential" (default: "linear")
        decay_rate: Rate of exponential decay (only used if decay_type="exponential", default: 5.0)

    Returns:
        Time-decayed reward tensor (0.0 to 1.0 range before weight scaling)

    Raises:
        ValueError: If termination_term is not found in the termination manager
    """
    # Get the termination signal directly from the termination manager
    # This ensures perfect synchronization between termination and reward
    term_manager = env.termination_manager

    if termination_term not in term_manager.active_terms:
        available_terms = term_manager.active_terms
        raise ValueError(f"Termination term '{termination_term}' not found. Available terms: {available_terms}")

    # Get the boolean termination signal for this specific term using the proper API
    terminated = term_manager.get_term(termination_term)

    # Calculate episode progress (0.0 at start, 1.0 at max_episode_length)
    progress = env.episode_length_buf.float() / env.max_episode_length

    # Apply decay based on type
    if decay_type == "linear":
        time_multiplier = 1.0 - progress
    elif decay_type == "exponential":
        time_multiplier = torch.exp(-decay_rate * progress)
    else:
        raise ValueError(f"Invalid decay_type '{decay_type}'. Must be 'linear' or 'exponential'.")

    return terminated.float() * time_multiplier


def ball_in_own_half(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg):
    ball_pos = root_xy_pos_w(env, ball_cfg)
    return 1.0 * (ball_pos[:, 1] < 0.0)


def distance_to_wall(env: ManagerBasedRLEnv, player_cfg: SceneEntityCfg) -> torch.Tensor:
    player_pos = body_xy_pos_w(env, player_cfg)
    device = player_pos.device
    cost = torch.zeros(player_pos.shape[0], device=device)

    x_edge = player_pos[:, 0] < 0.03 + torch.tensor(KLASK_PARAMS["joint_x_pos_limit"][0])
    distance = player_pos[:, 0] - torch.tensor(KLASK_PARAMS["joint_x_pos_limit"][0])
    cost += x_edge * torch.exp(-5 * distance)

    x_edge = torch.tensor(KLASK_PARAMS["joint_x_pos_limit"][1]) - player_pos[:, 0] < 0.03
    distance = torch.tensor(KLASK_PARAMS["joint_x_pos_limit"][1]) - player_pos[:, 0]
    cost += x_edge * torch.exp(-5 * distance)

    y_edge = player_pos[:, 1] - torch.tensor(KLASK_PARAMS["joint_y1_pos_limit"][0]) < 0.03
    distance = player_pos[:, 1] - torch.tensor(KLASK_PARAMS["joint_y1_pos_limit"][0])
    cost += y_edge * torch.exp(-5 * distance)

    y_edge = torch.tensor(KLASK_PARAMS["joint_y1_pos_limit"][1]) - player_pos[:, 1] < 0.03
    distance = torch.tensor(KLASK_PARAMS["joint_y1_pos_limit"][1]) - player_pos[:, 1]
    cost += y_edge * torch.exp(-5 * distance)

    return 1.0 * (cost)


def angle_ball_goal(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    player_cfg: SceneEntityCfg,
    goal: tuple[float, float, float],
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    player_pos = body_xy_pos_w(env, player_cfg)
    cx, cy, r = goal
    goal_pos = torch.tensor([cx, cy], device=player_pos.device)  # shape (2,)

    # Vectors from player to goal and ball
    vec_to_goal = goal_pos - player_pos  # shape (N, 2)
    vec_to_ball = ball_pos - player_pos  # shape (N, 2)

    # Dot product between the vectors
    dot = torch.sum(vec_to_goal * vec_to_ball, dim=1)  # shape (N,)

    # Norms (magnitudes)
    norm_goal = torch.norm(vec_to_goal, dim=1)  # shape (N,)
    norm_ball = torch.norm(vec_to_ball, dim=1)  # shape (N,)

    # Cosine of angle
    cos_theta = dot / (norm_goal * norm_ball + 1e-8)  # avoid division by zero
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # numerical stability

    # Angle in radians
    angle_rad = torch.acos(cos_theta)
    return angle_rad.unsqueeze(-1)


def angle_ball_opp(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg,
    player_1_cfg: SceneEntityCfg,
    player_2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    ball_pos = root_xy_pos_w(env, ball_cfg)
    player_pos = body_xy_pos_w(env, player_1_cfg)
    opponent_pos = body_xy_pos_w(env, player_2_cfg)
    vec_opp_to_ball = ball_pos - player_pos  # (N, 2)

    # Vector from ball to player
    vec_ball_to_player = opponent_pos - player_pos  # (N, 2)

    # Dot product and norms
    dot = torch.sum(vec_opp_to_ball * vec_ball_to_player, dim=1)  # (N,)
    norm1 = torch.norm(vec_opp_to_ball, dim=1)  # (N,)
    norm2 = torch.norm(vec_ball_to_player, dim=1)  # (N,)

    # Cosine and angle
    cos_theta = dot / (norm1 * norm2 + 1e-8)  # Avoid division by zero
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Clamp for numerical stability
    angle_rad = torch.acos(cos_theta)  # (N,)

    return angle_rad.unsqueeze(-1)


def distance_to_goal(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, goal: tuple) -> torch.Tensor:
    """
    Compute Euclidean distance from ball to goal center.

    Args:
        env: simulation environment
        ball_cfg: config for ball entity
        goal: (cx, cy, r) goal center and radius

    Returns:
        torch.Tensor of shape (N,) with distances
    """
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    goal_pos = torch.tensor(goal[:2], device=ball_pos.device)  # (2,)
    dist = torch.norm(ball_pos - goal_pos, dim=1)  # (N,)
    return dist.unsqueeze(-1)


def distance_ball_to_player(
    env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg, player_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Compute Euclidean distance from ball to player.

    Args:
        env: simulation environment
        ball_cfg: config for ball entity
        player_cfg: config for player entity

    Returns:
        torch.Tensor of shape (N,) with distances
    """
    ball_pos = root_xy_pos_w(env, ball_cfg)  # (N, 2)
    player_pos = body_xy_pos_w(env, player_cfg)  # (N, 2)
    dist = torch.norm(ball_pos - player_pos, dim=1)  # (N,)
    return dist.unsqueeze(-1)
