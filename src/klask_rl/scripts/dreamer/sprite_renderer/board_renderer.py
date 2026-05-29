"""Sprite-based board renderer for Klask.

Composites pre-extracted sprites (left peg, right peg, ball) onto a background
image based on board state positions. Designed for maximum runtime speed by
precomputing all heavy work at init time.
"""

import glob
import json
import os

import cv2
import numpy as np


_AUG_AXES = ("brightness", "contrast", "gamma", "color_temp", "saturation", "hue")


def _axis_levels(axis_cfg: dict) -> np.ndarray:
    """Return the discrete levels for a single axis config dict.

    Uses ``np.linspace(lo, hi, levels)`` — ``levels`` evenly spaced values
    inclusive of both endpoints. ``levels=1`` yields the single value ``lo``.
    """
    lo, hi = axis_cfg["range"]
    n_levels = int(axis_cfg["levels"])
    if n_levels < 1:
        raise ValueError(f"Augmentation levels must be >= 1, got {n_levels}")
    return np.linspace(lo, hi, n_levels)


def num_augmentation_variants(aug_cfg: dict | None) -> int:
    """Compute total variants = 1 (identity) + product of enabled-axis level counts.

    Identity variant 0 is always present. When at least one axis is enabled,
    the Cartesian product of enabled-axis levels makes up variants 1..N.
    Validates the config first so malformed YAML produces a clear error.
    """
    if not aug_cfg:
        return 1
    _validate_aug_cfg(aug_cfg)
    enabled_count = 0
    total = 1
    for axis in _AUG_AXES:
        c = aug_cfg.get(axis, {})
        if c.get("enabled"):
            total *= len(_axis_levels(c))
            enabled_count += 1
    return 1 + total if enabled_count > 0 else 1


def freeze_aug_cfg(cfg: dict | None):
    """Convert a (possibly nested) augmentation_cfg dict into a hashable tuple."""
    if not cfg:
        return ()
    out = []
    for k in sorted(cfg.keys()):
        v = cfg[k]
        if isinstance(v, dict):
            out.append((k, freeze_aug_cfg(v)))
        elif isinstance(v, list):
            out.append((k, tuple(v)))
        else:
            out.append((k, v))
    return tuple(out)


def _validate_aug_cfg(cfg: dict) -> None:
    """Raise on unknown axis names or malformed entries."""
    allowed_top = set(_AUG_AXES) | {"hold_per_episode"}
    for k in cfg.keys():
        if k not in allowed_top:
            raise ValueError(
                f"Unknown augmentation key {k!r}. Allowed: {sorted(allowed_top)}"
            )
    for axis in _AUG_AXES:
        c = cfg.get(axis)
        if c is None:
            continue
        if not isinstance(c, dict):
            raise ValueError(f"augmentation.{axis} must be a dict, got {type(c).__name__}")
        if not c.get("enabled"):
            continue
        if "range" not in c or "levels" not in c:
            raise ValueError(f"augmentation.{axis} requires 'range' and 'levels' when enabled")
        if len(c["range"]) != 2 or c["range"][0] > c["range"][1]:
            raise ValueError(f"augmentation.{axis}.range must be [lo, hi] with lo <= hi")
        if not isinstance(c["levels"], int) or c["levels"] < 1:
            raise ValueError(f"augmentation.{axis}.levels must be a positive integer, got {c['levels']!r}")


def _build_color_lut(brightness: float, contrast: float, gamma: float, color_temp: float) -> np.ndarray:
    """Compose brightness/contrast/gamma/color_temp into a single 256x3 uint8 LUT.

    Operations applied in this order to each pixel value v in [0, 1]:
        v = v * brightness                      # brightness scale
        v = (v - 0.5) * contrast + 0.5          # contrast about mid-gray
        v = clip(v, 0, 1) ** (1/gamma)          # gamma curve (gamma>1 brightens midtones)
        v[B] *= (1 - color_temp); v[R] *= (1 + color_temp)   # warm = +ct (more R, less B)

    Returns a (256, 3) uint8 LUT applied in BGR channel order (cv2 convention).
    """
    base = np.arange(256, dtype=np.float32) / 255.0  # (256,)
    v = base * brightness
    v = (v - 0.5) * contrast + 0.5
    v = np.clip(v, 0.0, 1.0)
    if gamma != 1.0:
        v = np.power(v, 1.0 / gamma)
    # Build per-channel LUT (BGR order)
    lut = np.empty((256, 3), dtype=np.float32)
    lut[:, 0] = v * (1.0 - color_temp)  # B
    lut[:, 1] = v                        # G
    lut[:, 2] = v * (1.0 + color_temp)  # R
    return np.clip(lut * 255.0, 0, 255).astype(np.uint8)


def _apply_lut(bgr: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply a per-channel uint8 LUT of shape (256, 3) to a uint8 BGR image.

    Uses numpy advanced indexing rather than ``cv2.LUT`` because OpenCV's LUT
    only accepts a single 1D table; we need a different mapping per channel.
    """
    out = np.empty_like(bgr)
    out[..., 0] = lut[bgr[..., 0], 0]
    out[..., 1] = lut[bgr[..., 1], 1]
    out[..., 2] = lut[bgr[..., 2], 2]
    return out


def _apply_hsv(bgr: np.ndarray, sat_scale: float, hue_shift_deg: float) -> np.ndarray:
    """Apply saturation scale + hue shift via HSV roundtrip. Input/output uint8 BGR."""
    if sat_scale == 1.0 and hue_shift_deg == 0.0:
        return bgr
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    if hue_shift_deg != 0.0:
        # OpenCV H range is [0, 180) for 8-bit HSV
        hsv[..., 0] = (hsv[..., 0] + hue_shift_deg / 2.0) % 180.0
    if sat_scale != 1.0:
        hsv[..., 1] = np.clip(hsv[..., 1] * sat_scale, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


class BoardRenderer:
    """Renders a Klask board image from a board state (3 object positions).

    Parameters
    ----------
    sprite_dir : str
        Path to the sprite output directory containing ``peg/`` and ``ball/``
        subdirectories with ``*.png`` and ``*.json`` files.
    background_path : str
        Path to the background PNG image (e.g. ``median_background.png``).
    output_size : tuple[int, int]
        ``(width, height)`` of the output image.  For ``target_frame="sim"``
        a portrait size such as ``(96, 126)`` is typically appropriate.
    fast_mode : bool
        If *False* (default), composite at full source resolution (652x495)
        then downscale with ``INTER_LINEAR`` (~0.2 ms). Gives better position
        precision through sub-pixel interpolation.
        If *True*, pre-downscale everything to *output_size* at init (~0.02 ms
        per render) but sprites snap to coarser output-resolution pixels.
    board_width_m : float
        Physical board width in metres.
    board_height_m : float
        Physical board height in metres.
    target_frame : str
        Coordinate frame for ``render()`` inputs.
        ``"camera"`` (default) — origin at top-left, X right, Y down.
        Landscape output (player on left, opponent on right).
        ``"sim"`` — origin at board centre, X right, Y up.  Portrait output
        (player at bottom, opponent at top).  Background and sprites are
        rotated 90 deg CCW at init time.
    """

    # TODO: remove these hardcoded constants by reading from sprite metadata at init time

    # Grid dimensions per object type
    _PEG_COLS = 8
    _PEG_ROWS = 6
    _BALL_COLS = 4
    _BALL_ROWS = 3

    # Grid margin offsets (metres) from sprite_generator_params.yaml
    _LEFT_HALF_X_MIN = 0.02
    _LEFT_HALF_X_MAX_OFFSET = 0.04  # from board centre
    _RIGHT_HALF_X_MIN_OFFSET = 0.04  # from board centre
    _RIGHT_HALF_X_MAX = 0.02  # from board right edge
    _PEG_Y_MIN = 0.02
    _PEG_Y_MAX_OFFSET = 0.02  # from board bottom edge
    _BALL_MARGIN = 0.02

    def __init__(
        self,
        sprite_dir: str,
        background_path: str,
        output_size: tuple[int, int] = (126, 96),
        fast_mode: bool = False,
        board_width_m: float = 0.42,
        board_height_m: float = 0.32,
        target_frame: str = "camera",
        augmentation_cfg: dict | None = None,
    ):
        """Initialise the renderer by loading and precomputing all sprites and data.

        ``augmentation_cfg`` (optional) enables color/lighting augmentation. When
        provided, the renderer precomputes one (bg, sprite_grids) variant per
        Cartesian combination of enabled-axis levels. ``render()`` then takes an
        ``aug_id`` integer to look up which variant to composite. See
        :func:`num_augmentation_variants` and the module-level docstring.
        """
        if target_frame not in ("camera", "sim"):
            raise ValueError(f"target_frame must be 'camera' or 'sim', got {target_frame!r}")
        self._output_size = output_size
        self._fast_mode = fast_mode
        self._board_w = board_width_m
        self._board_h = board_height_m
        self._sim_frame = target_frame == "sim"
        self._half_w = board_width_m / 2.0
        self._half_h = board_height_m / 2.0

        # Load background
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(f"Cannot load background: {background_path}")
        if self._sim_frame:
            bg = cv2.rotate(bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self._src_h, self._src_w = bg.shape[:2]

        # Compute scale factors (source -> output)
        self._scale_x = output_size[0] / self._src_w
        self._scale_y = output_size[1] / self._src_h

        # Build grid centroids for each object type
        half_w = board_width_m / 2.0
        left_xs = np.linspace(self._LEFT_HALF_X_MIN, half_w - self._LEFT_HALF_X_MAX_OFFSET, self._PEG_COLS)
        right_xs = np.linspace(
            half_w + self._RIGHT_HALF_X_MIN_OFFSET, board_width_m - self._RIGHT_HALF_X_MAX, self._PEG_COLS
        )
        peg_ys = np.linspace(self._PEG_Y_MIN, board_height_m - self._PEG_Y_MAX_OFFSET, self._PEG_ROWS)
        ball_xs = np.linspace(self._BALL_MARGIN, board_width_m - self._BALL_MARGIN, self._BALL_COLS)
        ball_ys = np.linspace(self._BALL_MARGIN, board_height_m - self._BALL_MARGIN, self._BALL_ROWS)

        # Precompute searchsorted boundaries (midpoints between centroids)
        self._left_bx = _midpoints(left_xs)
        self._left_by = _midpoints(peg_ys)
        self._right_bx = _midpoints(right_xs)
        self._right_by = _midpoints(peg_ys)
        self._ball_bx = _midpoints(ball_xs)
        self._ball_by = _midpoints(ball_ys)

        # Load and organise sprites into 2D grids
        peg_dir = os.path.join(sprite_dir, "peg")
        ball_dir = os.path.join(sprite_dir, "ball")

        left_raw = _load_sprites(peg_dir, "left_peg")
        right_raw = _load_sprites(peg_dir, "right_peg")
        ball_raw = _load_sprites(ball_dir, "ball")

        left_grid = _assign_to_grid(left_raw, left_xs, peg_ys, self._PEG_COLS, self._PEG_ROWS)
        right_grid = _assign_to_grid(right_raw, right_xs, peg_ys, self._PEG_COLS, self._PEG_ROWS)
        ball_grid = _assign_to_grid(ball_raw, ball_xs, ball_ys, self._BALL_COLS, self._BALL_ROWS)

        # Physical dimensions along image axes (swapped for sim frame due to rotation)
        if self._sim_frame:
            phys_w, phys_h = board_height_m, board_width_m
        else:
            phys_w, phys_h = board_width_m, board_height_m

        # Metres-to-pixels conversion factors (at working resolution).
        # Prepare BGRA grids first so the augmentation pass can re-premultiply
        # them with each variant's color transform.
        rot = self._sim_frame
        if fast_mode:
            self._bg = cv2.resize(bg, output_size, interpolation=cv2.INTER_AREA)
            self._m2px_x = output_size[0] / phys_w
            self._m2px_y = output_size[1] / phys_h
            sx, sy = self._scale_x, self._scale_y
        else:
            self._bg = bg
            self._m2px_x = self._src_w / phys_w
            self._m2px_y = self._src_h / phys_h
            sx, sy = 1.0, 1.0

        left_bgra = _prepare_bgra_grid(left_grid, self._PEG_ROWS, self._PEG_COLS, sx, sy, rotate_ccw=rot)
        right_bgra = _prepare_bgra_grid(right_grid, self._PEG_ROWS, self._PEG_COLS, sx, sy, rotate_ccw=rot)
        ball_bgra = _prepare_bgra_grid(ball_grid, self._BALL_ROWS, self._BALL_COLS, sx, sy, rotate_ccw=rot)

        # Variant 0 = identity. Compute it and grab inv_alpha grids for sharing.
        self._left, left_inv_a = _premultiply_bgra_grid(left_bgra)
        self._right, right_inv_a = _premultiply_bgra_grid(right_bgra)
        self._ball, ball_inv_a = _premultiply_bgra_grid(ball_bgra)

        self._canvas_h, self._canvas_w = self._bg.shape[:2]

        # ------------------------------------------------------------------
        # Augmentation precompute (Cartesian product over enabled-axis levels)
        # ------------------------------------------------------------------
        self._augmentation_cfg = augmentation_cfg
        self._bg_variants: list[np.ndarray] = [self._bg]
        self._left_variants: list[list[list[tuple]]] = [self._left]
        self._right_variants: list[list[list[tuple]]] = [self._right]
        self._ball_variants: list[list[list[tuple]]] = [self._ball]

        if augmentation_cfg:
            _validate_aug_cfg(augmentation_cfg)
            enabled_axes: list[tuple[str, np.ndarray]] = []
            for axis in _AUG_AXES:
                c = augmentation_cfg.get(axis, {})
                if c.get("enabled"):
                    enabled_axes.append((axis, _axis_levels(c)))
            if enabled_axes:
                import itertools as _it

                from tqdm import tqdm

                level_lists = [levels for _, levels in enabled_axes]
                axis_names = [name for name, _ in enabled_axes]
                total = 1
                for lvls in level_lists:
                    total *= len(lvls)
                desc = f"Precomputing sprite augmentations ({'x'.join(str(len(l)) for l in level_lists)} = {total})"
                for combo in tqdm(_it.product(*level_lists), total=total, desc=desc, unit="variant"):
                    params = dict(zip(axis_names, combo))
                    brightness = float(params.get("brightness", 1.0))
                    contrast = float(params.get("contrast", 1.0))
                    gamma = float(params.get("gamma", 1.0))
                    color_temp = float(params.get("color_temp", 0.0))
                    sat = float(params.get("saturation", 1.0))
                    hue = float(params.get("hue", 0.0))

                    lut = _build_color_lut(brightness, contrast, gamma, color_temp)

                    def _transform(bgr: np.ndarray, _lut=lut, _sat=sat, _hue=hue) -> np.ndarray:
                        out = _apply_lut(bgr, _lut)
                        out = _apply_hsv(out, _sat, _hue)
                        return out

                    self._bg_variants.append(_transform(self._bg))
                    left_v, _ = _premultiply_bgra_grid(left_bgra, _transform, shared_inv_alpha=left_inv_a)
                    right_v, _ = _premultiply_bgra_grid(right_bgra, _transform, shared_inv_alpha=right_inv_a)
                    ball_v, _ = _premultiply_bgra_grid(ball_bgra, _transform, shared_inv_alpha=ball_inv_a)
                    self._left_variants.append(left_v)
                    self._right_variants.append(right_v)
                    self._ball_variants.append(ball_v)

        self._num_variants = len(self._bg_variants)

    @property
    def num_variants(self) -> int:
        """Total number of precomputed augmentation variants (including identity)."""
        return self._num_variants

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, left_peg_m, right_peg_m, ball_m, aug_id: int = 0) -> np.ndarray:
        """Render a board image for the given object positions.

        Parameters
        ----------
        left_peg_m, right_peg_m, ball_m : array-like of length 2
            ``(x, y)`` position of each object in metres.  When
            ``target_frame="camera"`` these are camera-frame coordinates
            (origin top-left, X right, Y down).  When ``target_frame="sim"``
            these are sim-frame coordinates (origin at board centre, X right,
            Y up).
        aug_id : int
            Index into the precomputed augmentation variants. ``0`` is the
            identity (no augmentation). Ignored when ``augmentation_cfg`` was
            not provided.

        Returns:
        -------
        np.ndarray
            ``uint8`` BGR image of shape ``(out_h, out_w, 3)``.
        """
        if aug_id < 0 or aug_id >= self._num_variants:
            raise IndexError(
                f"aug_id={aug_id} out of range; renderer has {self._num_variants} variants"
            )
        bg_variant = self._bg_variants[aug_id]
        left_variant = self._left_variants[aug_id]
        right_variant = self._right_variants[aug_id]
        ball_variant = self._ball_variants[aug_id]
        canvas = bg_variant.copy()

        # Convert input coordinates to camera frame for grid lookup
        if self._sim_frame:
            half_w = self._half_w
            half_h = self._half_h
            l_cx, l_cy = left_peg_m[1] + half_w, left_peg_m[0] + half_h
            r_cx, r_cy = right_peg_m[1] + half_w, right_peg_m[0] + half_h
            b_cx, b_cy = ball_m[1] + half_w, ball_m[0] + half_h
        else:
            l_cx, l_cy = left_peg_m[0], left_peg_m[1]
            r_cx, r_cy = right_peg_m[0], right_peg_m[1]
            b_cx, b_cy = ball_m[0], ball_m[1]

        # Look up sprite grid indices (camera-frame boundaries)
        lix = int(np.searchsorted(self._left_bx, l_cx).clip(0, self._PEG_COLS - 1))
        liy = int(np.searchsorted(self._left_by, l_cy).clip(0, self._PEG_ROWS - 1))

        rix = int(np.searchsorted(self._right_bx, r_cx).clip(0, self._PEG_COLS - 1))
        riy = int(np.searchsorted(self._right_by, r_cy).clip(0, self._PEG_ROWS - 1))

        bix = int(np.searchsorted(self._ball_bx, b_cx).clip(0, self._BALL_COLS - 1))
        biy = int(np.searchsorted(self._ball_by, b_cy).clip(0, self._BALL_ROWS - 1))

        # Composite sprites at the *requested* pixel positions
        m2px_x = self._m2px_x
        m2px_y = self._m2px_y
        cw = self._canvas_w
        ch = self._canvas_h

        if self._sim_frame:
            # Sim -> rotated-image pixel coords
            _composite(
                canvas,
                left_variant[liy][lix],
                (left_peg_m[0] + half_h) * m2px_x,
                (half_w - left_peg_m[1]) * m2px_y,
                cw,
                ch,
            )
            _composite(
                canvas,
                right_variant[riy][rix],
                (right_peg_m[0] + half_h) * m2px_x,
                (half_w - right_peg_m[1]) * m2px_y,
                cw,
                ch,
            )
            _composite(
                canvas, ball_variant[biy][bix], (ball_m[0] + half_h) * m2px_x, (half_w - ball_m[1]) * m2px_y, cw, ch
            )
        else:
            _composite(canvas, left_variant[liy][lix], l_cx * m2px_x, l_cy * m2px_y, cw, ch)
            _composite(canvas, right_variant[riy][rix], r_cx * m2px_x, r_cy * m2px_y, cw, ch)
            _composite(canvas, ball_variant[biy][bix], b_cx * m2px_x, b_cy * m2px_y, cw, ch)

        # Downscale if full-res mode
        if not self._fast_mode:
            canvas = cv2.resize(canvas, self._output_size, interpolation=cv2.INTER_LINEAR)

        return canvas


# ======================================================================
# Private helpers
# ======================================================================


def _midpoints(centroids: np.ndarray) -> np.ndarray:
    """Compute decision-boundary midpoints between sorted centroids."""
    return (centroids[:-1] + centroids[1:]) / 2.0


def _load_sprites(directory: str, prefix: str) -> list[dict]:
    """Load all sprite PNGs and JSON metadata with the given prefix."""
    json_files = sorted(glob.glob(os.path.join(directory, f"{prefix}_*.json")))
    sprites = []
    for jf in json_files:
        with open(jf) as f:
            meta = json.load(f)
        png_path = jf.replace(".json", ".png")
        img = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)  # BGRA
        if img is None:
            raise FileNotFoundError(f"Cannot load sprite image: {png_path}")
        sprites.append({"meta": meta, "img": img})
    return sprites


def _assign_to_grid(
    sprites: list[dict],
    grid_xs: np.ndarray,
    grid_ys: np.ndarray,
    n_cols: int,
    n_rows: int,
) -> list[list[dict | None]]:
    """Assign each sprite to its nearest grid cell.

    Returns a ``[row][col]`` 2D list.
    """
    grid = [[None] * n_cols for _ in range(n_rows)]
    bx = _midpoints(grid_xs)
    by = _midpoints(grid_ys)

    for sp in sprites:
        pos = sp["meta"]["position_m"]
        ix = int(np.searchsorted(bx, pos[0]).clip(0, n_cols - 1))
        iy = int(np.searchsorted(by, pos[1]).clip(0, n_rows - 1))
        if grid[iy][ix] is not None:
            # Keep the one closest to the grid centroid
            old_pos = grid[iy][ix]["meta"]["position_m"]
            old_dist = (old_pos[0] - grid_xs[ix]) ** 2 + (old_pos[1] - grid_ys[iy]) ** 2
            new_dist = (pos[0] - grid_xs[ix]) ** 2 + (pos[1] - grid_ys[iy]) ** 2
            if new_dist >= old_dist:
                continue
        grid[iy][ix] = sp

    # Validate completeness
    for iy in range(n_rows):
        for ix in range(n_cols):
            if grid[iy][ix] is None:
                raise ValueError(
                    f"No sprite assigned to grid cell ({ix}, {iy}). "
                    f"Expected {n_cols}x{n_rows} = {n_cols * n_rows} sprites, "
                    f"got {len(sprites)}."
                )
    return grid


def _prepare_bgra_grid(
    grid: list[list[dict]],
    n_rows: int,
    n_cols: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotate_ccw: bool = False,
) -> list[list[tuple]]:
    """Rotate and resize the per-cell BGRA sprites to working resolution.

    Returns ``[row][col]`` of ``(off_x, off_y, bgra_uint8)``. This is the
    pre-premultiplication form needed by the augmentation pass.
    """
    result = [[None] * n_cols for _ in range(n_rows)]
    for iy in range(n_rows):
        for ix in range(n_cols):
            sp = grid[iy][ix]
            meta = sp["meta"]
            rgba = sp["img"]  # BGRA uint8

            offset_px = list(meta["offset_px"])

            if rotate_ccw:
                orig_w = rgba.shape[1]
                rgba = cv2.rotate(rgba, cv2.ROTATE_90_COUNTERCLOCKWISE)
                offset_px = [offset_px[1], orig_w - offset_px[0]]

            if scale_x != 1.0 or scale_y != 1.0:
                new_w = max(1, round(rgba.shape[1] * scale_x))
                new_h = max(1, round(rgba.shape[0] * scale_y))
                rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
                off_x = offset_px[0] * scale_x
                off_y = offset_px[1] * scale_y
            else:
                off_x = offset_px[0]
                off_y = offset_px[1]

            result[iy][ix] = (off_x, off_y, rgba)
    return result


def _premultiply_bgra_grid(
    bgra_grid: list[list[tuple]],
    bgr_transform=None,
    shared_inv_alpha: list[list[np.ndarray]] | None = None,
) -> tuple[list[list[tuple]], list[list[np.ndarray]]]:
    """Premultiply each cell's BGRA into ``(off_x, off_y, premul_bgr, inv_alpha)``.

    If ``bgr_transform`` is given, it is applied to the uint8 BGR channels of
    each cell before premultiplication. If ``shared_inv_alpha`` is given, those
    inv_alpha tensors are reused (so a per-axis color shift doesn't allocate
    new alpha tensors per variant — alpha is invariant under color ops).

    Returns ``(premul_grid, inv_alpha_grid)``. The second value is the
    inv_alpha-by-cell grid you can pass back in as ``shared_inv_alpha`` for the
    next variant.
    """
    n_rows = len(bgra_grid)
    n_cols = len(bgra_grid[0])
    result = [[None] * n_cols for _ in range(n_rows)]
    inv_alpha_grid: list[list[np.ndarray]] = [[None] * n_cols for _ in range(n_rows)]
    for iy in range(n_rows):
        for ix in range(n_cols):
            off_x, off_y, rgba = bgra_grid[iy][ix]
            bgr = rgba[:, :, :3]
            if bgr_transform is not None:
                bgr = bgr_transform(bgr)
            alpha_u8 = rgba[:, :, 3:4]
            if shared_inv_alpha is not None:
                inv_alpha = shared_inv_alpha[iy][ix]
                alpha_f = 1.0 - inv_alpha
            else:
                alpha_f = alpha_u8.astype(np.float32) / 255.0
                inv_alpha = 1.0 - alpha_f
            premul_bgr = np.clip(bgr.astype(np.float32), 0.0, 255.0) * alpha_f
            result[iy][ix] = (off_x, off_y, premul_bgr, inv_alpha)
            inv_alpha_grid[iy][ix] = inv_alpha
    return result, inv_alpha_grid


def _precompute_grid(
    grid: list[list[dict]],
    n_rows: int,
    n_cols: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    rotate_ccw: bool = False,
) -> list[list[tuple]]:
    """Precompute premultiplied-alpha compositing data for every grid cell.

    Returns ``[row][col]`` of ``(offset_x, offset_y, premul_bgr, inv_alpha)``.
    Convenience wrapper that combines ``_prepare_bgra_grid`` and
    ``_premultiply_bgra_grid`` for callers that don't need augmentation.
    """
    bgra_grid = _prepare_bgra_grid(grid, n_rows, n_cols, scale_x, scale_y, rotate_ccw)
    premul_grid, _ = _premultiply_bgra_grid(bgra_grid)
    return premul_grid


def _composite(
    canvas: np.ndarray,
    sprite_data: tuple,
    centre_px_x: float,
    centre_px_y: float,
    canvas_w: int,
    canvas_h: int,
) -> None:
    """Alpha-composite a precomputed sprite onto the canvas in-place.

    *centre_px_x/y* is the requested object centre in working-resolution pixels.
    The sprite's precomputed offset converts this to the top-left placement.
    """
    off_x, off_y, premul, inv_a = sprite_data
    sh, sw = premul.shape[:2]

    px = round(centre_px_x - off_x)
    py = round(centre_px_y - off_y)

    # Compute canvas region (clip to bounds)
    cx0 = max(px, 0)
    cy0 = max(py, 0)
    cx1 = min(px + sw, canvas_w)
    cy1 = min(py + sh, canvas_h)

    if cx0 >= cx1 or cy0 >= cy1:
        return  # fully outside canvas

    # Corresponding sprite region
    sx0 = cx0 - px
    sy0 = cy0 - py
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    roi = canvas[cy0:cy1, cx0:cx1]
    canvas[cy0:cy1, cx0:cx1] = (premul[sy0:sy1, sx0:sx1] + roi.astype(np.float32) * inv_a[sy0:sy1, sx0:sx1]).astype(
        np.uint8
    )
