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
        ``(width, height)`` of the output image.
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
    """

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
    ):
        """Initialise the renderer by loading and precomputing all sprites and data."""
        self._output_size = output_size
        self._fast_mode = fast_mode
        self._board_w = board_width_m
        self._board_h = board_height_m

        # Load background
        bg = cv2.imread(background_path, cv2.IMREAD_COLOR)
        if bg is None:
            raise FileNotFoundError(f"Cannot load background: {background_path}")
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

        # Metres-to-pixels conversion factors (at working resolution)
        if fast_mode:
            self._bg = cv2.resize(bg, output_size, interpolation=cv2.INTER_AREA)
            self._m2px_x = output_size[0] / board_width_m
            self._m2px_y = output_size[1] / board_height_m
            self._left = _precompute_grid(left_grid, self._PEG_ROWS, self._PEG_COLS, self._scale_x, self._scale_y)
            self._right = _precompute_grid(right_grid, self._PEG_ROWS, self._PEG_COLS, self._scale_x, self._scale_y)
            self._ball = _precompute_grid(ball_grid, self._BALL_ROWS, self._BALL_COLS, self._scale_x, self._scale_y)
        else:
            self._bg = bg
            self._m2px_x = self._src_w / board_width_m
            self._m2px_y = self._src_h / board_height_m
            self._left = _precompute_grid(left_grid, self._PEG_ROWS, self._PEG_COLS)
            self._right = _precompute_grid(right_grid, self._PEG_ROWS, self._PEG_COLS)
            self._ball = _precompute_grid(ball_grid, self._BALL_ROWS, self._BALL_COLS)

        self._canvas_h, self._canvas_w = self._bg.shape[:2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, left_peg_m, right_peg_m, ball_m) -> np.ndarray:
        """Render a board image for the given object positions.

        Parameters
        ----------
        left_peg_m, right_peg_m, ball_m : array-like of length 2
            ``(x, y)`` position of each object in metres.

        Returns:
        -------
        np.ndarray
            ``uint8`` BGR image of shape ``(out_h, out_w, 3)``.
        """
        canvas = self._bg.copy()

        # Look up sprite grid indices
        lix = int(np.searchsorted(self._left_bx, left_peg_m[0]).clip(0, self._PEG_COLS - 1))
        liy = int(np.searchsorted(self._left_by, left_peg_m[1]).clip(0, self._PEG_ROWS - 1))

        rix = int(np.searchsorted(self._right_bx, right_peg_m[0]).clip(0, self._PEG_COLS - 1))
        riy = int(np.searchsorted(self._right_by, right_peg_m[1]).clip(0, self._PEG_ROWS - 1))

        bix = int(np.searchsorted(self._ball_bx, ball_m[0]).clip(0, self._BALL_COLS - 1))
        biy = int(np.searchsorted(self._ball_by, ball_m[1]).clip(0, self._BALL_ROWS - 1))

        # Composite sprites at the *requested* pixel positions
        m2px_x = self._m2px_x
        m2px_y = self._m2px_y
        cw = self._canvas_w
        ch = self._canvas_h

        _composite(canvas, self._left[liy][lix], left_peg_m[0] * m2px_x, left_peg_m[1] * m2px_y, cw, ch)
        _composite(canvas, self._right[riy][rix], right_peg_m[0] * m2px_x, right_peg_m[1] * m2px_y, cw, ch)
        _composite(canvas, self._ball[biy][bix], ball_m[0] * m2px_x, ball_m[1] * m2px_y, cw, ch)

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


def _precompute_grid(
    grid: list[list[dict]],
    n_rows: int,
    n_cols: int,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> list[list[tuple]]:
    """Precompute premultiplied-alpha compositing data for every grid cell.

    Returns ``[row][col]`` of ``(offset_x, offset_y, premul_bgr, inv_alpha)``.
    ``offset_x/y`` is the sprite's centre offset in working-resolution pixels,
    used at render time to convert a requested centre position to a top-left
    placement coordinate.
    """
    result = [[None] * n_cols for _ in range(n_rows)]
    for iy in range(n_rows):
        for ix in range(n_cols):
            sp = grid[iy][ix]
            meta = sp["meta"]
            rgba = sp["img"]  # BGRA uint8

            offset_px = meta["offset_px"]

            if scale_x != 1.0 or scale_y != 1.0:
                # Downscale sprite
                new_w = max(1, round(rgba.shape[1] * scale_x))
                new_h = max(1, round(rgba.shape[0] * scale_y))
                rgba = cv2.resize(rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)
                off_x = offset_px[0] * scale_x
                off_y = offset_px[1] * scale_y
            else:
                off_x = offset_px[0]
                off_y = offset_px[1]

            # Premultiplied alpha
            alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
            premul_bgr = rgba[:, :, :3].astype(np.float32) * alpha
            inv_alpha = 1.0 - alpha

            result[iy][ix] = (off_x, off_y, premul_bgr, inv_alpha)
    return result


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
