# Board aspect ratio: 420mm x 320mm = 21:16 base ratio.
# Valid camera resolutions are multiples of (21, 16).
_BOARD_BASE_H = 21  # maps to the 420mm dimension
_BOARD_BASE_W = 16  # maps to the 320mm dimension


def _camera_params_from_size(image_size: int) -> tuple[int, int]:
    """Derive camera resolution from a square image size.

    Returns the largest (cam_height, cam_width) that is an exact multiple of
    the 21x16 board ratio and fits within image_size x image_size.
    """
    scale = min(image_size // _BOARD_BASE_H, image_size // _BOARD_BASE_W)
    return scale * _BOARD_BASE_H, scale * _BOARD_BASE_W


def apply_camera_size_to_env_cfg(env_cfg, size_cfg) -> None:
    """Apply camera resolution and observation padding derived from *size_cfg*.

    *size_cfg* is the ``env.size`` value from the Hydra config — either a
    two-element sequence ``[H, W]`` or a plain integer.  Only the first
    (height) element is used; images are assumed square.

    Mutates *env_cfg* in-place:
      - ``env_cfg.scene.camera.{width,height}``
      - ``target_h`` / ``target_w`` params on any ``image`` observation term
        found under ``env_cfg.observations.image`` and
        ``env_cfg.observations.opponent_image``.
    """
    if size_cfg is None:
        return
    image_size = int(size_cfg[0])
    cam_h, cam_w = _camera_params_from_size(image_size)
    env_cfg.scene.camera.width = cam_w
    env_cfg.scene.camera.height = cam_h
    for group_attr in ("image", "opponent_image"):
        group = getattr(env_cfg.observations, group_attr, None)
        if group is not None:
            image_term = getattr(group, "image", None)
            if image_term is not None and isinstance(getattr(image_term, "params", None), dict):
                image_term.params["target_h"] = image_size
                image_term.params["target_w"] = image_size
