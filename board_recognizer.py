"""
Screen / Image Board Recognizer for Mahjong Soul (雀魂).

Features:
- Bundles & uses exact templates from assets/templates.npz:
  * Original theme (mjpface_default) & Cat theme (mjpface_2308event), both clean & corner-indexed
  * ALL 62 tile backs (mjpai) in both 2D hand/UI and 3D table views
- Exact Unity 3D camera homography (scene_mj) -> 1280x1280 bird's-eye top-down table rectification
  so all 4 player orientations (bottom, right, top, left) + rotated Riichi/meld tiles are matched uniformly
- Exact Unity 2D orthographic hand geometry (scene_mjhandpai) with vertical hover/lift tolerance
- Multi-frame temporal glint suppression + gradient/chroma ZNCC for Dora shimmer calibration
- Automatic Ankan (暗杠: 2 tile backs + 2 faces -> 4 copies) detection in meld zones
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image, ImageDraw

from mahjong_engine import BoardState


def resource_path(rel_path: str) -> str:
    base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel_path)


# Unity scene_mj camera constants (from scene_mj.prefab)
CAM_POS = np.array([0.0, 0.846, 0.882], dtype=np.float64)
CAM_QUAT = (0.0, 0.9171, -0.3987, 0.0)
CAM_FOV_Y_DEG = 28.0
REF_ASPECT = 16.0 / 9.0
TABLE_Y = 0.020

# Bird's-eye table mapping: world X in [+0.56, -0.56] -> x in [0, 1280], Z in [-0.56, +0.56] -> y in [0, 1280]
BEV_SIZE = 1280
BEV_HALF_SPAN = 0.56
BEV_PX_PER_M = BEV_SIZE / (2.0 * BEV_HALF_SPAN)  # 1142.857 px/m
TILE_W_M = 0.0301
TILE_H_M = 0.03995
BEV_TILE_W = TILE_W_M * BEV_PX_PER_M   # ~34.4 px
BEV_TILE_H = TILE_H_M * BEV_PX_PER_M   # ~45.66 px

# World anchors for hai_1..4 and babei_1..4 from scene_mj.prefab
HAI_ANCHORS = [
    (0.0834, 0.1330),   # 0: self (bottom)
    (-0.1240, 0.0884),  # 1: right (shimocha)
    (-0.0808, -0.1120), # 2: top (toimen)
    (0.1280, -0.0740),  # 3: left (kamicha)
]

BABEI_ANCHORS = [
    (-0.2357, 0.2770),   # 0: self (bottom, k_ccw=0)
    (-0.2832, -0.1630),  # 1: right (shimocha, k_ccw=3)
    (0.2750, -0.2580),   # 2: top (3P kamicha / 4P toimen, k_ccw=2)
    (0.2951, 0.2560),    # 3: left (k_ccw=1)
]

# Tiles whose face looks similar when rotated 90 degrees (require extra geometric checks for Riichi)
ROT_SYMMETRIC_TILES = {'1p', '2p', '4p', '5p', '0p', '8p', '9p', '5z'}

# Tiles absent from 3-player mahjong (2m..8m, 0m)
SANMA_EXCLUDED_TILES = {'2m', '3m', '4m', '5m', '6m', '7m', '8m', '0m'}


def _effective_height(w: int, h: int) -> float:
    """Effective 16:9 reference height for Unity Canvas/Camera scaling at arbitrary aspect ratios."""
    return min(float(h), float(w) * (9.0 / 16.0))


def _quat_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    return np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)


_CAM_R = _quat_to_mat(*CAM_QUAT)
_CAM_FWD = _CAM_R @ np.array([0.0, 0.0, 1.0])
_CAM_UP = _CAM_R @ np.array([0.0, 1.0, 0.0])
_CAM_RIGHT = _CAM_R @ np.array([1.0, 0.0, 0.0])
_TAN_HALF_Y = math.tan(math.radians(CAM_FOV_Y_DEG) / 2.0)


def project_world_to_screen(pt3: np.ndarray, width: int, height: int) -> Tuple[float, float]:
    d = pt3 - CAM_POS
    cx = float(np.dot(d, _CAM_RIGHT))
    cy = float(np.dot(d, _CAM_UP))
    cz = float(np.dot(d, _CAM_FWD))
    asp = float(width) / max(1.0, float(height))
    if asp < REF_ASPECT:
        tan_half_x = _TAN_HALF_Y * REF_ASPECT
        tan_half_y = tan_half_x / asp
    else:
        tan_half_y = _TAN_HALF_Y
        tan_half_x = _TAN_HALF_Y * asp
    nx = cx / (cz * tan_half_x)
    ny = cy / (cz * tan_half_y)
    return (nx + 1.0) * 0.5 * width, (1.0 - ny) * 0.5 * height


def world_to_bev(x_w: float, z_w: float) -> Tuple[float, float]:
    xb = (BEV_HALF_SPAN - x_w) * BEV_PX_PER_M
    yb = (z_w + BEV_HALF_SPAN) * BEV_PX_PER_M
    return xb, yb


def bev_to_world(xb: float, yb: float) -> Tuple[float, float]:
    x_w = BEV_HALF_SPAN - xb / BEV_PX_PER_M
    z_w = yb / BEV_PX_PER_M - BEV_HALF_SPAN
    return x_w, z_w


def _world_to_bev_roi_local(
    wx: float,
    wz: float,
    bev_roi: Tuple[int, int, int, int],
    k_ccw: int,
) -> Tuple[float, float]:
    """Map a world (x, z) anchor into rotated local ROI coordinates matching np.rot90(..., k=k_ccw)."""
    xb, yb = world_to_bev(wx, wz)
    bx0, by0, bx1, by1 = bev_roi
    rw, rh = float(bx1 - bx0), float(by1 - by0)
    sx, sy = xb - bx0, yb - by0
    if k_ccw == 0:
        return sx, sy
    elif k_ccw == 1:
        return sy, rw - sx
    elif k_ccw == 2:
        return rw - sx, rh - sy
    else:
        return rh - sy, sx


def _solve_perspective_coeffs(dst_pts: List[Tuple[float, float]], src_pts: List[Tuple[float, float]]) -> Tuple[float, ...]:
    """Compute 8 PIL PERSPECTIVE coefficients mapping dst (x,y) -> src (u,v)."""
    matrix = []
    rhs = []
    for (x, y), (u, v) in zip(dst_pts, src_pts):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rhs.append(u)
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        rhs.append(v)
    coeffs = np.linalg.solve(np.asarray(matrix, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    return tuple(float(c) for c in coeffs)


def warp_table_to_bev(screen_img: Image.Image) -> Tuple[np.ndarray, Tuple[float, ...]]:
    w, h = screen_img.size
    world_corners = [
        np.array([+BEV_HALF_SPAN, TABLE_Y, -BEV_HALF_SPAN]),
        np.array([-BEV_HALF_SPAN, TABLE_Y, -BEV_HALF_SPAN]),
        np.array([-BEV_HALF_SPAN, TABLE_Y, +BEV_HALF_SPAN]),
        np.array([+BEV_HALF_SPAN, TABLE_Y, +BEV_HALF_SPAN]),
    ]
    src_pts = [project_world_to_screen(p, w, h) for p in world_corners]
    dst_pts = [(0.0, 0.0), (float(BEV_SIZE), 0.0), (float(BEV_SIZE), float(BEV_SIZE)), (0.0, float(BEV_SIZE))]
    coeffs = _solve_perspective_coeffs(dst_pts, src_pts)
    bev_img = screen_img.transform((BEV_SIZE, BEV_SIZE), Image.Transform.PERSPECTIVE, coeffs, Image.Resampling.BILINEAR)
    return np.asarray(bev_img, dtype=np.uint8), coeffs


def _compute_features(patches_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized feature extraction for (N, H, W, 3) uint8 patches.
    Returns (gray_z, grad_z, chroma_z, mean_rgb, std_gray, chroma_std).
    """
    f = patches_rgb.astype(np.float32)
    n, h, w, _ = f.shape
    hw = h * w

    # Luminance gray
    gray = f[..., 0] * np.float32(0.299)
    gray += f[..., 1] * np.float32(0.587)
    gray += f[..., 2] * np.float32(0.114)

    # Zero-mean unit-norm gray (also yields std_gray without a second pass)
    g_c = gray.reshape(n, hw).copy()
    g_c -= g_c.mean(axis=1, keepdims=True)
    g_ss = np.einsum('ij,ij->i', g_c, g_c)
    std_gray = np.sqrt(g_ss / np.float32(hw))
    g_norm = np.sqrt(g_ss)[:, None] + np.float32(1e-5)
    g_c /= g_norm
    gray_z = g_c

    # Edge/gradient magnitude (glint-resistant)
    grad = np.zeros((n, h, w), dtype=np.float32)
    grad[:, :, 1:] = np.abs(gray[:, :, 1:] - gray[:, :, :-1])
    grad[:, 1:, :] += np.abs(gray[:, 1:, :] - gray[:, :-1, :])
    gr_c = grad.reshape(n, hw)
    gr_c -= gr_c.mean(axis=1, keepdims=True)
    gr_norm = np.sqrt(np.einsum('ij,ij->i', gr_c, gr_c))[:, None] + np.float32(1e-5)
    gr_c /= gr_norm
    grad_z = gr_c

    # Color chroma (R-G, G-B) zero-mean unit-norm + spatial chroma std
    chroma = np.empty((n, 2 * hw), dtype=np.float32)
    c1 = chroma[:, :hw]
    c2 = chroma[:, hw:]
    np.subtract(f[..., 0].reshape(n, hw), f[..., 1].reshape(n, hw), out=c1)
    np.subtract(f[..., 1].reshape(n, hw), f[..., 2].reshape(n, hw), out=c2)
    c1_m = c1.mean(axis=1)
    c2_m = c2.mean(axis=1)
    c1_ss = np.einsum('ij,ij->i', c1, c1) - np.float32(hw) * (c1_m * c1_m)
    c2_ss = np.einsum('ij,ij->i', c2, c2) - np.float32(hw) * (c2_m * c2_m)
    chroma_std = np.sqrt(np.maximum(np.float32(0.0), (c1_ss + c2_ss) / np.float32(hw)))

    chroma -= chroma.mean(axis=1, keepdims=True)
    ch_norm = np.sqrt(np.einsum('ij,ij->i', chroma, chroma))[:, None] + np.float32(1e-5)
    chroma /= ch_norm

    mean_rgb = f.reshape(n, hw, 3).mean(axis=1)
    return gray_z, grad_z, chroma, mean_rgb, std_gray, chroma_std


def _box_sum_integral(img_2d: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Compute exact sliding-window sums of shape (H-bh+1, W-bw+1) via 2D prefix sums."""
    h, w = img_2d.shape
    if h < bh or w < bw:
        return np.empty((0, 0), dtype=img_2d.dtype)
    cs = np.pad(img_2d, ((1, 0), (1, 0)), mode='constant').cumsum(axis=0).cumsum(axis=1)
    return cs[bh:, bw:] - cs[:-bh, bw:] - cs[bh:, :-bw] + cs[:-bh, :-bw]


def _bright_mask_grid(rgb: np.ndarray, bh: int, bw: int, thr: int, frac: float = 0.25) -> np.ndarray:
    """Vectorized boolean mask of windows where at least `frac` of channel values >= `thr`."""
    cnt = (rgb >= thr).sum(axis=2, dtype=np.int32)
    sums = _box_sum_integral(cnt, bh, bw)
    min_cnt = int(frac * 3 * bh * bw)
    return sums >= min_cnt


def _gray_std_grid(rgb: np.ndarray, bh: int, bw: int) -> np.ndarray:
    """Vectorized exact spatial luminance standard deviation over (bh, bw) patches."""
    f = rgb.astype(np.float64)
    gray = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
    n = float(bh * bw)
    sum1 = _box_sum_integral(gray, bh, bw)
    sum2 = _box_sum_integral(gray * gray, bh, bw)
    var = np.maximum(0.0, sum2 / n - (sum1 / n) ** 2)
    return np.sqrt(var)


def _make_sliding_view(
    roi: np.ndarray,
    crop_w: int,
    crop_h: int,
    out_w: int,
    out_h: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Resize `roi` once to template scale and return (sliding_windows, scale_x, scale_y).
    sliding_windows has shape (Hs - out_h + 1, Ws - out_w + 1, out_h, out_w, 3).
    """
    rh, rw = roi.shape[:2]
    scaled_w = max(out_w, int(round(rw * (out_w / float(crop_w)))))
    scaled_h = max(out_h, int(round(rh * (out_h / float(crop_h)))))
    scaled = np.asarray(
        Image.fromarray(roi).resize((scaled_w, scaled_h), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    windows = sliding_window_view(scaled, (out_h, out_w, 3))[:, :, 0]
    return windows, scaled_w / float(rw), scaled_h / float(rh)


def _sample_sliding_view(
    windows: np.ndarray,
    sx: float,
    sy: float,
    coords_xy: List[Tuple[int, int]],
) -> np.ndarray:
    """Gather (M, out_h, out_w, 3) contiguous patches from a pre-resized sliding window view."""
    max_py = windows.shape[0] - 1
    max_px = windows.shape[1] - 1
    xs = np.array([c[0] for c in coords_xy], dtype=np.float32)
    ys = np.array([c[1] for c in coords_xy], dtype=np.float32)
    px = np.clip(np.rint(xs * sx).astype(np.intp), 0, max_px)
    py = np.clip(np.rint(ys * sy).astype(np.intp), 0, max_py)
    return np.ascontiguousarray(windows[py, px])


def _downsample_patches(patches: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize (N, H, W, 3) template array to (N, out_h, out_w, 3) using PIL BILINEAR."""
    n = len(patches)
    out = np.empty((n, out_h, out_w, 3), dtype=np.uint8)
    for i in range(n):
        out[i] = np.asarray(Image.fromarray(patches[i]).resize((out_w, out_h), Image.Resampling.BILINEAR), dtype=np.uint8)
    return out


class FeatureBank:
    def __init__(self, patches_rgb: np.ndarray, labels: List[str], is_back: np.ndarray, is_blank_5z: np.ndarray):
        self.h, self.w = patches_rgb.shape[1], patches_rgb.shape[2]
        self.labels = labels
        self.is_back = is_back
        self.is_blank_5z = is_blank_5z
        self.has_blank_5z = bool(np.any(is_blank_5z))
        (
            self.gray_z,
            self.grad_z,
            self.chroma_z,
            self.mean_rgb,
            self.std_gray,
            self.chroma_std,
        ) = _compute_features(patches_rgb)
        self.gray_z_T = np.ascontiguousarray(self.gray_z.T)
        self.grad_z_T = np.ascontiguousarray(self.grad_z.T)
        self.chroma_z_T = np.ascontiguousarray(self.chroma_z.T)
        self.is_low_var_back = is_back & (self.std_gray < 14.0)
        self.has_low_var_back = bool(np.any(self.is_low_var_back))
        self.t_r = np.ascontiguousarray(self.mean_rgb[:, 0])
        self.t_g = np.ascontiguousarray(self.mean_rgb[:, 1])
        self.t_b = np.ascontiguousarray(self.mean_rgb[:, 2])
        self.t_rg = np.ascontiguousarray(self.t_r - self.t_g)
        self.t_gb = np.ascontiguousarray(self.t_g - self.t_b)
        self.cd_back_mask = is_back[None, :]

    def match_features(
        self,
        feats: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        cand_rgb: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Match precomputed features from `_compute_features(cand_rgb)` against this bank."""
        pg, pgr, pch, pm, pstd, pcstd = feats
        m = len(pg)
        sim_gray = pg @ self.gray_z_T
        sim_grad = pgr @ self.grad_z_T
        sim_ch = pch @ self.chroma_z_T

        # Low-allocation color & chroma mean distances
        color_dist = np.abs(pm[:, 0:1] - self.t_r[None, :])
        color_dist += np.abs(pm[:, 1:2] - self.t_g[None, :])
        color_dist += np.abs(pm[:, 2:3] - self.t_b[None, :])
        color_dist *= np.float32(1.0 / 3.0)

        p_rg = pm[:, 0] - pm[:, 1]
        p_gb = pm[:, 1] - pm[:, 2]
        chroma_mean_dist = np.abs(p_rg[:, None] - self.t_rg[None, :])
        chroma_mean_dist += np.abs(p_gb[:, None] - self.t_gb[None, :])
        chroma_mean_dist *= np.float32(0.5)

        # Adaptive chroma weight: ignore chroma correlation noise on monochrome/neutral tiles
        max_cstd = np.maximum(pcstd[:, None], self.chroma_std[None, :])
        w_ch = np.clip((max_cstd - np.float32(5.0)) * np.float32(1.0 / 22.0), np.float32(0.0), np.float32(1.0)) * np.float32(0.24)
        rem_w = np.float32(0.24) - w_ch
        w_g = np.float32(0.48) + rem_w * np.float32(0.55)
        w_gr = np.float32(0.28) + rem_w * np.float32(0.45)

        cd_pen = np.where(
            self.cd_back_mask,
            np.float32(0.0022) * color_dist,
            np.float32(0.0008) * color_dist + np.float32(0.0014) * chroma_mean_dist,
        )
        scores = w_g * sim_gray
        scores += w_gr * sim_grad
        scores += w_ch * sim_ch
        scores -= cd_pen

        # Handle low-variance uniform tile backs (e.g. solid orange/green/yellow mjpai)
        if self.has_low_var_back:
            low_rows = pstd < 20.0
            if np.any(low_rows):
                low_var_mask = low_rows[:, None] & self.is_low_var_back[None, :]
                uniform_back_score = np.float32(0.85) - np.float32(0.0080) * color_dist
                scores = np.where(low_var_mask, np.maximum(scores, uniform_back_score), scores)

        # Handle blank White Dragon 5z (only compute saturation on low/moderate-std bright ivory candidates)
        if self.has_blank_5z and cand_rgb is not None:
            bright_ivory = (pstd < 35.0) & (pm.min(axis=1) > 150.0) & (pm.mean(axis=1) > 165.0)
            if np.any(bright_ivory):
                b_idx = np.flatnonzero(bright_ivory)
                sub = cand_rgb[b_idx]
                sub_f = sub.astype(np.float32)
                psat_sub = (sub_f.max(axis=3) - sub_f.min(axis=3)).reshape(len(b_idx), -1).mean(axis=1)
                is_blank_sub = (pstd[b_idx] < 13.0) & (psat_sub < 26.0)
                if sub.shape[1] >= 8 and sub.shape[2] >= 8:
                    pad_y = 2 if sub.shape[1] >= 16 else 1
                    pad_x = 2 if sub.shape[2] >= 16 else 1
                    inn = sub_f[:, pad_y:-pad_y, pad_x:-pad_x]
                    ig = (inn[..., 0] * 0.299 + inn[..., 1] * 0.587 + inn[..., 2] * 0.114).reshape(len(b_idx), -1)
                    istd = ig.std(axis=1)
                    im = inn.reshape(len(b_idx), -1, 3).mean(axis=1)
                    isat = (inn.max(axis=3) - inn.min(axis=3)).reshape(len(b_idx), -1).mean(axis=1)
                    is_blank_sub = is_blank_sub | ((istd < 13.0) & (im.min(axis=1) > 150.0) & (im.mean(axis=1) > 165.0) & (isat < 26.0))
                if np.any(is_blank_sub):
                    blank_cand = np.zeros(m, dtype=bool)
                    blank_cand[b_idx[is_blank_sub]] = True
                    blank_mask = blank_cand[:, None] & self.is_blank_5z[None, :]
                    blank_score = np.float32(0.88) - np.float32(0.0015) * color_dist
                    scores = np.where(blank_mask, np.maximum(scores, blank_score), scores)

        best_idx = np.argmax(scores, axis=1)
        best_scores = scores[np.arange(m), best_idx]
        best_labels = [self.labels[i] for i in best_idx]
        best_backs = self.is_back[best_idx]
        return best_scores, best_labels, best_backs

    def match_batch(self, cand_rgb: np.ndarray) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """
        Match (M, H, W, 3) patches against this bank.
        Returns (best_scores, best_labels, best_is_back).
        """
        feats = _compute_features(cand_rgb)
        return self.match_features(feats, cand_rgb=cand_rgb)


@dataclass
class DetectedBox:
    label: str
    score: float
    box: Tuple[int, int, int, int]  # (x0, y0, x1, y1) in screen pixels
    is_back: bool = False
    is_rotated: bool = False


@dataclass
class RecognitionResult:
    board: BoardState
    detected_theme: str
    hand_boxes: List[DetectedBox] = field(default_factory=list)
    dora_boxes: List[DetectedBox] = field(default_factory=list)
    river_boxes: List[List[DetectedBox]] = field(default_factory=lambda: [[], [], [], []])
    meld_boxes: List[List[DetectedBox]] = field(default_factory=lambda: [[], [], [], []])
    kita_boxes: List[List[DetectedBox]] = field(default_factory=lambda: [[], [], [], []])
    avg_confidence: float = 0.0
    notes: List[str] = field(default_factory=list)


class BoardRecognizer:
    def __init__(self, templates_path: Optional[str] = None):
        path = templates_path or resource_path(os.path.join('assets', 'templates.npz'))
        data = np.load(path, allow_pickle=False)
        self.tile_names: List[str] = [str(x) for x in data['tile_names']]
        self.hand_variant_names: List[str] = [str(x) for x in data['hand_variant_names']]
        self.mjp_variant_names: List[str] = [str(x) for x in data['mjp_variant_names']]
        self.back_names: List[str] = [str(x) for x in data['back_names']]
        self.ui_icons: Dict[str, Image.Image] = {
            self.tile_names[i]: Image.fromarray(data['ui_icons'][i])
            for i in range(len(self.tile_names))
        }

        hand_tpls = data['hand_templates']  # (4, 37, H_h, W_h, 3)
        mjp_tpls = data['mjp_templates']    # (4, 37, H_m, W_m, 3)
        back_hand = data['back_hand']       # (62, H_h, W_h, 3)
        back_mjp = data['back_mjp']         # (62, H_m, W_m, 3)

        self.hand_h, self.hand_w = hand_tpls.shape[2], hand_tpls.shape[3]
        self.mjp_h, self.mjp_w = mjp_tpls.shape[2], mjp_tpls.shape[3]
        self.hand_cw, self.hand_ch = self.hand_w // 2, self.hand_h // 2
        self.mjp_cw, self.mjp_ch = self.mjp_w // 2, self.mjp_h // 2
        self._pool = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))

        # Build hand banks per theme ('default' and 'cat'), each including clean + indexed + 62 backs
        self.hand_banks: Dict[str, FeatureBank] = {}
        self.hand_face_banks: Dict[str, FeatureBank] = {}
        self.hand_face_coarse_banks: Dict[str, FeatureBank] = {}
        for theme in ('default', 'cat'):
            idxs = [i for i, vn in enumerate(self.hand_variant_names) if vn.startswith(theme + '_')]
            face_patches = np.concatenate([hand_tpls[i] for i in idxs], axis=0)
            face_labels = self.tile_names * len(idxs)
            face_blank = np.array([lbl == '5z' for lbl in face_labels], dtype=bool)
            self.hand_face_banks[theme] = FeatureBank(
                face_patches, face_labels,
                np.zeros(len(face_labels), dtype=bool),
                face_blank,
            )
            self.hand_face_coarse_banks[theme] = FeatureBank(
                _downsample_patches(face_patches, self.hand_cw, self.hand_ch),
                face_labels,
                np.zeros(len(face_labels), dtype=bool),
                face_blank,
            )

            all_patches = np.concatenate([face_patches, back_hand], axis=0)
            all_labels = face_labels + [f'BACK:{b}' for b in self.back_names]
            is_back = np.array([False] * len(face_labels) + [True] * len(self.back_names), dtype=bool)
            is_blank = np.concatenate([face_blank, np.zeros(len(self.back_names), dtype=bool)])
            self.hand_banks[theme] = FeatureBank(all_patches, all_labels, is_back, is_blank)

        # Build 3D table banks per theme ('default' and 'cat'):
        # 1) Upright portrait bank (0 deg faces + 0 & 180 deg backs)
        # 2) Sideways landscape bank (90 deg CCW and 90 deg CW for Riichi & called tiles!)
        self.mjp_upright_banks: Dict[str, FeatureBank] = {}
        self.mjp_side_banks: Dict[str, FeatureBank] = {}
        self.mjp_upright_coarse_banks: Dict[str, FeatureBank] = {}
        self.mjp_side_coarse_banks: Dict[str, FeatureBank] = {}
        for theme in ('default', 'cat'):
            idxs = [i for i, vn in enumerate(self.mjp_variant_names) if vn.startswith(theme + '_')]
            face_p = np.concatenate([mjp_tpls[i] for i in idxs], axis=0)
            face_lbl = self.tile_names * len(idxs)
            face_blank = np.array([lbl == '5z' for lbl in face_lbl], dtype=bool)

            # Upright (include 0 deg faces + 62 backs at 0 & 180 deg)
            back_180 = np.rot90(back_mjp, k=2, axes=(1, 2))
            up_patches = np.concatenate([face_p, back_mjp, back_180], axis=0)
            up_labels = face_lbl + [f'BACK:{b}' for b in self.back_names] * 2
            up_back = np.array([False] * len(face_lbl) + [True] * (2 * len(self.back_names)), dtype=bool)
            up_blank = np.concatenate([face_blank, np.zeros(2 * len(self.back_names), dtype=bool)])
            self.mjp_upright_banks[theme] = FeatureBank(up_patches, up_labels, up_back, up_blank)
            self.mjp_upright_coarse_banks[theme] = FeatureBank(
                _downsample_patches(up_patches, self.mjp_cw, self.mjp_ch),
                up_labels, up_back, up_blank,
            )

            # Sideways (k=1 CCW and k=3 CW for both faces and backs)
            face_ccw = np.rot90(face_p, k=1, axes=(1, 2))
            face_cw = np.rot90(face_p, k=3, axes=(1, 2))
            back_ccw = np.rot90(back_mjp, k=1, axes=(1, 2))
            back_cw = np.rot90(back_mjp, k=3, axes=(1, 2))
            side_patches = np.concatenate([face_ccw, face_cw, back_ccw, back_cw], axis=0)
            side_labels = face_lbl * 2 + [f'BACK:{b}' for b in self.back_names] * 2
            side_back = np.array([False] * (2 * len(face_lbl)) + [True] * (2 * len(self.back_names)), dtype=bool)
            side_blank = np.concatenate([face_blank, face_blank, np.zeros(2 * len(self.back_names), dtype=bool)])
            self.mjp_side_banks[theme] = FeatureBank(side_patches, side_labels, side_back, side_blank)
            self.mjp_side_coarse_banks[theme] = FeatureBank(
                _downsample_patches(side_patches, self.mjp_ch, self.mjp_cw),
                side_labels, side_back, side_blank,
            )

    @staticmethod
    def _suppress_glint_roi(frames: List[np.ndarray], y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """Take pixel-wise minimum across frames when ROI is stationary to strip dora_shine glint."""
        if len(frames) == 1:
            return frames[0][y0:y1, x0:x1]
        rois = [f[y0:y1, x0:x1] for f in frames]
        diff = np.mean(np.abs(rois[-1].astype(np.int16) - rois[0].astype(np.int16)))
        if diff < 16.0:
            out = rois[0]
            for r in rois[1:]:
                out = np.minimum(out, r)
            return out
        return rois[-1]

    @staticmethod
    def _resize_patches(crops: List[np.ndarray], out_w: int, out_h: int) -> np.ndarray:
        arr = np.empty((len(crops), out_h, out_w, 3), dtype=np.uint8)
        for i, c in enumerate(crops):
            arr[i] = np.asarray(Image.fromarray(c).resize((out_w, out_h), Image.Resampling.BILINEAR), dtype=np.uint8)
        return arr

    @staticmethod
    def _normalize_frames(frames: List[np.ndarray]) -> Tuple[List[np.ndarray], Tuple[int, int, float, float]]:
        """
        Strip pure-black letterbox/pillarbox margins and downscale >1200p captures to ~1080p
        for multi-resolution invariance and fast inference on 1440p/4K displays.
        Returns (normalized_frames, (crop_x0, crop_y0, scale_x, scale_y)).
        """
        latest = frames[-1]
        orig_h, orig_w = latest.shape[:2]
        rs = latest[:, ::4]
        cs = latest[::4, :]
        row_black = ((rs[:, :, 0] <= 6) & (rs[:, :, 1] <= 6) & (rs[:, :, 2] <= 6)).mean(axis=1) >= 0.98
        col_black = ((cs[:, :, 0] <= 6) & (cs[:, :, 1] <= 6) & (cs[:, :, 2] <= 6)).mean(axis=0) >= 0.98

        cy0 = 0
        while cy0 < orig_h and row_black[cy0]:
            cy0 += 1
        cy1 = orig_h
        while cy1 > cy0 and row_black[cy1 - 1]:
            cy1 -= 1
        cx0 = 0
        while cx0 < orig_w and col_black[cx0]:
            cx0 += 1
        cx1 = orig_w
        while cx1 > cx0 and col_black[cx1 - 1]:
            cx1 -= 1

        if (cx1 - cx0) < 320 or (cy1 - cy0) < 240:
            cx0, cy0, cx1, cy1 = 0, 0, orig_w, orig_h

        cropped = [f[cy0:cy1, cx0:cx1] for f in frames] if (cx0 > 0 or cy0 > 0 or cx1 < orig_w or cy1 < orig_h) else list(frames)
        cw, ch = cx1 - cx0, cy1 - cy0
        h_eff = _effective_height(cw, ch)

        if h_eff > 1200.0:
            target_scale = 1080.0 / h_eff
            nw = max(320, int(round(cw * target_scale)))
            nh = max(240, int(round(ch * target_scale)))
            sx = nw / float(cw)
            sy = nh / float(ch)
            norm = [
                np.asarray(Image.fromarray(f).resize((nw, nh), Image.Resampling.BILINEAR), dtype=np.uint8)
                for f in cropped
            ]
            return norm, (cx0, cy0, sx, sy)
        return cropped, (cx0, cy0, 1.0, 1.0)

    @staticmethod
    def _remap_box(
        box: Tuple[int, int, int, int],
        cx0: int,
        cy0: int,
        sx: float,
        sy: float,
        orig_w: int,
        orig_h: int,
    ) -> Tuple[int, int, int, int]:
        return (
            max(0, min(orig_w, int(round(box[0] / sx)) + cx0)),
            max(0, min(orig_h, int(round(box[1] / sy)) + cy0)),
            max(0, min(orig_w, int(round(box[2] / sx)) + cx0)),
            max(0, min(orig_h, int(round(box[3] / sy)) + cy0)),
        )

    def _remap_boxes(
        self,
        boxes: List[DetectedBox],
        cx0: int,
        cy0: int,
        sx: float,
        sy: float,
        orig_w: int,
        orig_h: int,
    ) -> List[DetectedBox]:
        if cx0 == 0 and cy0 == 0 and sx == 1.0 and sy == 1.0:
            return boxes
        return [
            DetectedBox(
                label=b.label,
                score=b.score,
                box=self._remap_box(b.box, cx0, cy0, sx, sy, orig_w, orig_h),
                is_back=b.is_back,
                is_rotated=b.is_rotated,
            )
            for b in boxes
        ]

    def recognize_hand_quick(self, frame: np.ndarray, theme_pref: str = 'auto') -> Tuple[str, List[DetectedBox]]:
        """Fast single-frame self-hand recognition (~40-80ms) for continuous draw monitoring."""
        orig_h, orig_w = frame.shape[:2]
        norm_frames, (cx0, cy0, sx, sy) = self._normalize_frames([frame])
        theme, hand_boxes = self._recognize_hand_and_theme(norm_frames, theme_pref)
        return theme, self._remap_boxes(hand_boxes, cx0, cy0, sx, sy, orig_w, orig_h)

    def _recognize_hand_and_theme(
        self,
        frames: List[np.ndarray],
        theme_pref: str,
    ) -> Tuple[str, List[DetectedBox]]:
        latest = frames[-1]
        h, w = latest.shape[:2]
        h_eff = _effective_height(w, h)

        # Expected hand tile dimensions from scene_mjhandpai (orthoSize=9.0, tile_h=2.54, scale_x=0.62)
        base_tile_h = 0.14111 * h_eff
        base_tile_w = 0.08749 * h_eff
        y_center = 0.5 * h + 0.4228 * h_eff
        ref_r = max(2, int(round(3.0 * h_eff / 1080.0)))

        themes_to_try = ['default', 'cat'] if theme_pref == 'auto' else [theme_pref]

        def _scan_at_scale(scale: float) -> Tuple[str, float, List[DetectedBox]]:
            tile_h = base_tile_h * scale
            tile_w = base_tile_w * scale
            # Inner-face crop ratios matching slice_hand_tiles (x: 4..76 of 80, y: 15..125 of 129)
            inner_w = max(12, int(round(tile_w * (72.0 / 80.0))))
            inner_h = max(18, int(round(tile_h * (110.0 / 129.0))))
            y_top_est = y_center - 0.5 * tile_h
            y_band_top = max(0, int(round(y_top_est - 0.065 * h_eff)))
            y_band_bot = min(h, int(round(y_center + 0.5 * tile_h + 0.03 * h_eff)))
            if y_band_bot - y_band_top < inner_h:
                return themes_to_try[0], -1e9, []

            roi_clean = self._suppress_glint_roi(frames, y_band_top, y_band_bot, 0, w)
            roi_latest = latest[y_band_top:y_band_bot, 0:w]
            use_dual = (len(frames) > 1) and (
                float(np.mean(np.abs(roi_latest.astype(np.int16) - roi_clean.astype(np.int16)))) > 1.5
            )

            roi_h = y_band_bot - y_band_top
            y_base = int(round((y_top_est - y_band_top) + tile_h * (15.0 / 129.0)))
            y_step = max(3, int(round(0.006 * h_eff)))
            y_offsets = sorted(set(
                max(0, min(roi_h - inner_h, y_base + dy))
                for dy in range(-int(0.045 * h_eff), int(0.025 * h_eff) + 1, y_step)
            ))
            step_x = max(3, int(round(inner_w * 0.07)))
            x_start = max(0, int(round(0.02 * w)))
            x_end = min(w - inner_w, int(round(0.98 * w)) - inner_w)
            xs = list(range(x_start, x_end + 1, step_x))

            # Quick y-band narrowing via prefix sum: pick top-2 y_offsets with brightest white tile density + neighbors
            if len(y_offsets) > 4:
                sub_l = roi_latest[:, x_start:x_end + inner_w]
                bright_px = (sub_l[:, :, 0] > 165) & (sub_l[:, :, 1] > 165) & (sub_l[:, :, 2] > 165)
                row_cs = np.pad(bright_px.sum(axis=1, dtype=np.int32), (1, 0)).cumsum()
                y_scores = [(int(row_cs[y + inner_h] - row_cs[y]), y) for y in y_offsets]
                y_scores.sort(key=lambda z: -z[0])
                top_ys = sorted(set(y for _, y in y_scores[:2]))
                peak_y = y_scores[0][1]
                y_offsets_use = sorted(set(
                    y for y in y_offsets
                    if (y in top_ys) or abs(y - peak_y) <= y_step
                ))
            else:
                y_offsets_use = y_offsets

            br_grid = _bright_mask_grid(roi_latest, inner_h, inner_w, thr=145, frac=0.25)
            cand_coords: List[Tuple[int, int]] = [
                (x, y)
                for y in y_offsets_use
                for x in xs
                if br_grid[y, x]
            ]

            if not cand_coords:
                return themes_to_try[0], -1e9, []

            # Coarse scan using pre-resized ROI sliding window view & half-resolution templates (compute features ONCE)
            win_c, sx_c, sy_c = _make_sliding_view(roi_clean, inner_w, inner_h, self.hand_cw, self.hand_ch)
            batch_clean = _sample_sliding_view(win_c, sx_c, sy_c, cand_coords)
            feats_clean = _compute_features(batch_clean)
            if use_dual:
                win_l, sx_l, sy_l = _make_sliding_view(roi_latest, inner_w, inner_h, self.hand_cw, self.hand_ch)
                batch_latest = _sample_sliding_view(win_l, sx_l, sy_l, cand_coords)
                feats_latest = _compute_features(batch_latest)
            else:
                batch_latest = None
                feats_latest = None

            coarse_results: List[Tuple[str, float, List[Tuple[int, int, float, str]]]] = []
            min_dx = tile_w * 0.78
            for theme in themes_to_try:
                cbank = self.hand_face_coarse_banks[theme]
                sc_c, lbl_c, _ = cbank.match_features(feats_clean, cand_rgb=batch_clean)
                if feats_latest is not None:
                    sc_l, lbl_l, _ = cbank.match_features(feats_latest, cand_rgb=batch_latest)
                    use_c = sc_c >= sc_l
                    scores = np.where(use_c, sc_c, sc_l)
                    labels = [lbl_c[i] if use_c[i] else lbl_l[i] for i in range(len(cand_coords))]
                else:
                    scores = sc_c
                    labels = lbl_c

                order = np.argsort(-scores)
                kept: List[Tuple[int, int, float, str]] = []
                for idx in order:
                    sc = float(scores[idx])
                    if sc < 0.38:
                        break
                    cx, cy = cand_coords[idx]
                    if any(abs(cx - kx) < min_dx for kx, _, _, _ in kept):
                        continue
                    kept.append((cx, cy, sc, labels[idx]))
                    if len(kept) >= 18:
                        break
                if kept:
                    coarse_results.append((theme, sum(k[2] for k in kept), kept))

            if not coarse_results:
                return themes_to_try[0], -1e9, []

            # If one theme clearly wins at coarse stage with >= 4 tiles, skip full-res work on the loser
            if len(coarse_results) > 1:
                coarse_results.sort(key=lambda r: -r[1])
                t0_name, tot0, kept0 = coarse_results[0]
                t1_name, tot1, kept1 = coarse_results[1]
                if len(kept0) >= 4 and tot0 > tot1 + 0.10 * max(len(kept0), len(kept1)):
                    coarse_results = [(t0_name, tot0, kept0)]

            sc_best_theme = themes_to_try[0]
            sc_best_total = -1e9
            sc_best_dets: List[DetectedBox] = []

            dx_vals = sorted(set((-ref_r - 1, -2, 0, 2, ref_r + 1)))
            dy_vals = sorted(set((-ref_r, 0, ref_r)))
            fwin_c, fsx_c, fsy_c = _make_sliding_view(roi_clean, inner_w, inner_h, self.hand_w, self.hand_h)
            if use_dual:
                fwin_l, fsx_l, fsy_l = _make_sliding_view(roi_latest, inner_w, inner_h, self.hand_w, self.hand_h)
            else:
                fwin_l = fsx_l = fsy_l = None

            for theme, _, kept in coarse_results:
                bank = self.hand_face_banks[theme]
                # Mark regular vs gap-fill candidates: (cx, cy, is_gap)
                tile_slots: List[Tuple[int, int, bool]] = [(cx, cy, False) for cx, cy, _, _ in kept]

                # Gap-fill check: if low-res missed a tile between two adjacent tiles (gap ~ 2*pitch), probe it
                kept_sorted = sorted(kept, key=lambda k: k[0])
                if len(kept_sorted) >= 3:
                    adj_dxs = [
                        kept_sorted[i + 1][0] - kept_sorted[i][0]
                        for i in range(len(kept_sorted) - 1)
                        if 0.85 * tile_w <= (kept_sorted[i + 1][0] - kept_sorted[i][0]) <= 1.25 * tile_w
                    ]
                    est_pitch = float(np.median(adj_dxs)) if adj_dxs else (tile_w * 1.005)
                    gap_coords: List[Tuple[int, int]] = []
                    gap_c: List[np.ndarray] = []
                    gap_l: List[np.ndarray] = []
                    gap_counts: List[int] = []
                    for i in range(len(kept_sorted) - 1):
                        gap = kept_sorted[i + 1][0] - kept_sorted[i][0]
                        if 1.65 * est_pitch <= gap <= 2.35 * est_pitch:
                            gx_est = int(round(0.5 * (kept_sorted[i][0] + kept_sorted[i + 1][0])))
                            gy_est = int(round(0.5 * (kept_sorted[i][1] + kept_sorted[i + 1][1])))
                            cnt_g = 0
                            for dy in range(-3, 4, 2):
                                for dx in range(-4, 5):
                                    nx = max(0, min(w - inner_w, gx_est + dx))
                                    ny = max(0, min(roi_h - inner_h, gy_est + dy))
                                    gap_coords.append((nx, ny))
                                    gap_c.append(roi_clean[ny:ny + inner_h, nx:nx + inner_w])
                                    if use_dual:
                                        gap_l.append(roi_latest[ny:ny + inner_h, nx:nx + inner_w])
                                    cnt_g += 1
                            gap_counts.append(cnt_g)
                    if gap_counts:
                        gbc = self._resize_patches(gap_c, self.hand_w, self.hand_h)
                        gsc_c, _, _ = bank.match_batch(gbc)
                        if use_dual:
                            gbl = self._resize_patches(gap_l, self.hand_w, self.hand_h)
                            gsc_l, _, _ = bank.match_batch(gbl)
                            gsc_all = np.maximum(gsc_c, gsc_l)
                        else:
                            gsc_all = gsc_c
                        off = 0
                        for cnt_g in gap_counts:
                            sub_sc = gsc_all[off:off + cnt_g]
                            gi = int(np.argmax(sub_sc))
                            if float(sub_sc[gi]) >= 0.45:
                                gx, gy = gap_coords[off + gi]
                                tile_slots.append((gx, gy, True))
                            off += cnt_g

                # Batched full-resolution refinement across all kept tiles via pre-resized sliding view
                ref_coords: List[Tuple[int, int]] = []
                slot_slices: List[Tuple[int, int, bool]] = []
                for cx, cy, is_gap in tile_slots:
                    s_start = len(ref_coords)
                    for dy in dy_vals:
                        ny = max(0, min(roi_h - inner_h, cy + dy))
                        for dx in dx_vals:
                            nx = max(0, min(w - inner_w, cx + dx))
                            ref_coords.append((nx, ny))
                    slot_slices.append((s_start, len(ref_coords), is_gap))

                rb_c = _sample_sliding_view(fwin_c, fsx_c, fsy_c, ref_coords)
                rsc_c, rlbl_c, _ = bank.match_batch(rb_c)
                if fwin_l is not None:
                    rb_l = _sample_sliding_view(fwin_l, fsx_l, fsy_l, ref_coords)
                    rsc_l, rlbl_l, _ = bank.match_batch(rb_l)
                else:
                    rsc_l, rlbl_l = None, None

                refined: List[DetectedBox] = []
                for s_start, s_end, is_gap in slot_slices:
                    sub_c = rsc_c[s_start:s_end]
                    i_c = s_start + int(np.argmax(sub_c))
                    if rsc_l is not None:
                        sub_l = rsc_l[s_start:s_end]
                        i_l = s_start + int(np.argmax(sub_l))
                        if rsc_c[i_c] >= rsc_l[i_l]:
                            f_sc, f_lbl, (fx, fy) = float(rsc_c[i_c]), rlbl_c[i_c], ref_coords[i_c]
                        else:
                            f_sc, f_lbl, (fx, fy) = float(rsc_l[i_l]), rlbl_l[i_l], ref_coords[i_l]
                    else:
                        f_sc, f_lbl, (fx, fy) = float(rsc_c[i_c]), rlbl_c[i_c], ref_coords[i_c]

                    min_thr = 0.45 if is_gap else 0.48
                    if f_sc >= min_thr:
                        refined.append(DetectedBox(
                            label=f_lbl,
                            score=f_sc,
                            box=(fx, y_band_top + fy, fx + inner_w, y_band_top + fy + inner_h),
                        ))

                if not refined:
                    continue
                refined.sort(key=lambda d: d.box[0])
                if len(refined) > 14:
                    refined = sorted(refined, key=lambda d: -d.score)[:14]
                    refined.sort(key=lambda d: d.box[0])

                total_sc = sum(d.score for d in refined)
                if total_sc > sc_best_total:
                    sc_best_total = total_sc
                    sc_best_theme = theme
                    sc_best_dets = refined

            return sc_best_theme, sc_best_total, sc_best_dets

        best_theme, best_total, best_detections = _scan_at_scale(1.0)
        mean_sc = (best_total / len(best_detections)) if best_detections else 0.0
        if len(best_detections) < 4 or mean_sc < 0.62:
            for alt_scale in (0.93, 1.07, 0.86, 1.14):
                t_alt, tot_alt, dets_alt = _scan_at_scale(alt_scale)
                if tot_alt > best_total:
                    best_theme, best_total, best_detections = t_alt, tot_alt, dets_alt

        if not best_detections:
            return ('default' if theme_pref == 'auto' else theme_pref), []
        return best_theme, best_detections

    def _recognize_dora_ui(self, frames: List[np.ndarray], theme: str) -> List[DetectedBox]:
        """Locate and classify the 1..5 dora indicators in the top-left 2D UI panel."""
        latest = frames[-1]
        h, w = latest.shape[:2]
        h_eff = _effective_height(w, h)
        x_canvas_left = max(0.0, (w - h_eff * (16.0 / 9.0)) * 0.5)
        y_canvas_top = max(0.0, (h - h_eff) * 0.5)
        roi_x1 = min(w, int(round(max(0.32 * w, x_canvas_left + 0.58 * h_eff))))
        roi_y1 = min(h, int(round(max(0.22 * h, y_canvas_top + 0.25 * h_eff))))
        roi_clean = self._suppress_glint_roi(frames, 0, roi_y1, 0, roi_x1)
        roi_latest = latest[0:roi_y1, 0:roi_x1]
        use_dual = (len(frames) > 1) and (
            float(np.mean(np.abs(roi_latest.astype(np.int16) - roi_clean.astype(np.int16)))) > 1.5
        )

        # Expected dora UI tile size from UI_DesktopInfo (80x120 at scale 0.7 on 1080p -> 56x84)
        base_tile_h = (84.0 / 1080.0) * h_eff
        base_tile_w = base_tile_h * (80.0 / 120.0)
        inner_w = max(10, int(round(base_tile_w * (72.0 / 80.0))))
        inner_h = max(14, int(round(base_tile_h * (110.0 / 129.0))))

        coarse_bank = self.hand_face_coarse_banks[theme]
        face_bank = self.hand_face_banks[theme]
        full_bank = self.hand_banks[theme]

        # Search for slot 0 (first dora indicator), trying 1.0 first for speed
        best_slot0 = None
        best_sc = -1e9
        best_dims = (inner_w, inner_h, base_tile_w)

        for scale in (1.0, 0.90, 1.10):
            cw = max(10, int(round(inner_w * scale)))
            ch = max(14, int(round(inner_h * scale)))
            step = max(2, int(round(cw * 0.08)))
            x_search_max = min(roi_x1 - cw, max(int(roi_x1 * 0.55) - cw, int(round(x_canvas_left + 0.40 * h_eff)) - cw))
            br_grid = _bright_mask_grid(roi_latest, ch, cw, thr=140, frac=0.25)
            coords = [
                (x, y)
                for y in range(2, max(3, roi_y1 - ch), step)
                for x in range(2, max(3, x_search_max), step)
                if br_grid[y, x]
            ]
            if not coords:
                continue

            win_c, sx_c, sy_c = _make_sliding_view(roi_clean, cw, ch, self.hand_cw, self.hand_ch)
            bc = _sample_sliding_view(win_c, sx_c, sy_c, coords)
            sc_c, _, _ = coarse_bank.match_batch(bc)
            if use_dual and float(sc_c.max()) < 0.68:
                win_l, sx_l, sy_l = _make_sliding_view(roi_latest, cw, ch, self.hand_cw, self.hand_ch)
                bl = _sample_sliding_view(win_l, sx_l, sy_l, coords)
                sc_l, _, _ = coarse_bank.match_batch(bl)
                sc = np.maximum(sc_c, sc_l)
            else:
                sc = sc_c

            # Pick top 3 spatially separated coarse peaks and verify at full resolution
            order = np.argsort(-sc)
            top_peaks: List[Tuple[int, int]] = []
            for idx in order:
                if float(sc[idx]) < 0.35 and top_peaks:
                    break
                px, py = coords[idx]
                if any(abs(px - tx) < cw * 0.6 and abs(py - ty) < ch * 0.6 for tx, ty in top_peaks):
                    continue
                top_peaks.append((px, py))
                if len(top_peaks) >= 3:
                    break
            if not top_peaks:
                continue

            # Expand top coarse peak by ±2px (3x3) plus centers of peaks 1..2 so grid misalignment never hides >=0.72
            ver_coords: List[Tuple[int, int]] = []
            p0x, p0y = top_peaks[0]
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    ver_coords.append((max(0, min(roi_x1 - cw, p0x + dx)), max(0, min(roi_y1 - ch, p0y + dy))))
            for px, py in top_peaks[1:]:
                ver_coords.append((px, py))

            pk_c = [roi_clean[py:py + ch, px:px + cw] for px, py in ver_coords]
            fsc_c, flbl_c, _ = face_bank.match_batch(self._resize_patches(pk_c, self.hand_w, self.hand_h))
            if use_dual:
                pk_l = [roi_latest[py:py + ch, px:px + cw] for px, py in ver_coords]
                fsc_l, flbl_l, _ = face_bank.match_batch(self._resize_patches(pk_l, self.hand_w, self.hand_h))
                use_fc = fsc_c >= fsc_l
                fsc = np.where(use_fc, fsc_c, fsc_l)
                flbl = [flbl_c[i] if use_fc[i] else flbl_l[i] for i in range(len(ver_coords))]
            else:
                fsc, flbl = fsc_c, flbl_c

            idx = int(np.argmax(fsc))
            if float(fsc[idx]) > best_sc:
                best_sc = float(fsc[idx])
                best_slot0 = (ver_coords[idx][0], ver_coords[idx][1], flbl[idx])
                best_dims = (cw, ch, base_tile_w * scale)
            if scale == 1.0 and best_sc >= 0.72:
                break

        if best_slot0 is None or best_sc < 0.45:
            return []

        x0, y0, _ = best_slot0
        cw, ch, step_slot = best_dims

        # Refine slot 0 within ±3px, then walk leftwards in case slot 1+ had slightly higher peak score
        ref_coords, ref_c, ref_l = [], [], []
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                nx = max(0, min(roi_x1 - cw, x0 + dx))
                ny = max(0, min(roi_y1 - ch, y0 + dy))
                ref_coords.append((nx, ny))
                ref_c.append(roi_clean[ny:ny + ch, nx:nx + cw])
                if use_dual:
                    ref_l.append(roi_latest[ny:ny + ch, nx:nx + cw])
        rb_c = self._resize_patches(ref_c, self.hand_w, self.hand_h)
        rsc = face_bank.match_batch(rb_c)[0]
        if use_dual:
            rb_l = self._resize_patches(ref_l, self.hand_w, self.hand_h)
            rsc = np.maximum(rsc, face_bank.match_batch(rb_l)[0])
        i_best = int(np.argmax(rsc))
        x0, y0 = ref_coords[i_best]

        # Walk leftwards to find true slot 0 if we initially locked onto slot 1..4
        for _ in range(4):
            lx = int(round(x0 - step_slot))
            if lx < 0:
                break
            l_coords, l_c, l_l = [], [], []
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    nx = max(0, min(roi_x1 - cw, lx + dx))
                    ny = max(0, min(roi_y1 - ch, y0 + dy))
                    p_l = roi_latest[ny:ny + ch, nx:nx + cw]
                    if float(np.percentile(p_l, 75)) < 135.0:
                        continue
                    l_coords.append((nx, ny))
                    l_c.append(roi_clean[ny:ny + ch, nx:nx + cw])
                    if use_dual:
                        l_l.append(p_l)
            if not l_coords:
                break
            lbc = self._resize_patches(l_c, self.hand_w, self.hand_h)
            lsc = face_bank.match_batch(lbc)[0]
            if use_dual:
                lbl_arr = self._resize_patches(l_l, self.hand_w, self.hand_h)
                lsc = np.maximum(lsc, face_bank.match_batch(lbl_arr)[0])
            li = int(np.argmax(lsc))
            if float(lsc[li]) >= 0.48:
                x0, y0 = l_coords[li]
            else:
                break

        # Classify all 5 dora indicator slots in one batch (face vs all 62 tile backs)
        all_coords: List[Tuple[int, int]] = []
        all_c: List[np.ndarray] = []
        all_l: List[np.ndarray] = []
        slot_slices: List[Tuple[int, int, int]] = []
        for k in range(5):
            sx = int(round(x0 + k * step_slot))
            if sx + cw > roi_x1:
                break
            s_start = len(all_coords)
            for dy in (-2, 0, 2):
                for dx in (-2, 0, 2):
                    nx = max(0, min(roi_x1 - cw, sx + dx))
                    ny = max(0, min(roi_y1 - ch, y0 + dy))
                    all_coords.append((nx, ny))
                    all_c.append(roi_clean[ny:ny + ch, nx:nx + cw])
                    if use_dual:
                        all_l.append(roi_latest[ny:ny + ch, nx:nx + cw])
            slot_slices.append((k, s_start, len(all_coords)))

        dora_boxes: List[DetectedBox] = []
        if not slot_slices:
            return dora_boxes

        bc = self._resize_patches(all_c, self.hand_w, self.hand_h)
        sc_c, lbl_c, bk_c = full_bank.match_batch(bc)
        if use_dual:
            bl = self._resize_patches(all_l, self.hand_w, self.hand_h)
            sc_l, lbl_l, bk_l = full_bank.match_batch(bl)
        else:
            sc_l, lbl_l, bk_l = None, None, None

        for k, s_start, s_end in slot_slices:
            if sc_l is not None:
                sc_sub = np.maximum(sc_c[s_start:s_end], sc_l[s_start:s_end])
                ib = s_start + int(np.argmax(sc_sub))
                if sc_c[ib] >= sc_l[ib]:
                    lbl, is_bk, fsc = lbl_c[ib], bool(bk_c[ib]), float(sc_c[ib])
                else:
                    lbl, is_bk, fsc = lbl_l[ib], bool(bk_l[ib]), float(sc_l[ib])
            else:
                ib = s_start + int(np.argmax(sc_c[s_start:s_end]))
                lbl, is_bk, fsc = lbl_c[ib], bool(bk_c[ib]), float(sc_c[ib])

            fx, fy = all_coords[ib]
            min_thr = 0.45 if k == 0 else 0.50
            if fsc < min_thr or is_bk:
                break
            if k >= 1 and float(np.percentile(roi_latest[fy:fy + ch, fx:fx + cw], 75)) < 135.0:
                break
            dora_boxes.append(DetectedBox(label=lbl, score=fsc, box=(fx, fy, fx + cw, fy + ch), is_back=False))
        return dora_boxes

    @staticmethod
    def _bev_box_to_screen(x0: float, y0: float, x1: float, y1: float, coeffs_fwd: np.ndarray, w: int, h: int) -> Tuple[int, int, int, int]:
        xs, ys = [], []
        for bx, by in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            wx, wz = bev_to_world(bx, by)
            sx, sy = project_world_to_screen(np.array([wx, TABLE_Y, wz]), w, h)
            xs.append(sx)
            ys.append(sy)
        return (
            max(0, int(round(min(xs)))),
            max(0, int(round(min(ys)))),
            min(w, int(round(max(xs)))),
            min(h, int(round(max(ys)))),
        )

    @staticmethod
    def _rotate_roi_box_back(rx0: int, ry0: int, rx1: int, ry1: int, roi_box: Tuple[int, int, int, int], k_ccw: int) -> Tuple[int, int, int, int]:
        """Map a box inside np.rot90(bev[y0:y1, x0:x1], k=k_ccw) back to BEV coordinates."""
        bx0, by0, bx1, by1 = roi_box
        rw, rh = bx1 - bx0, by1 - by0
        # k_ccw is the k passed to np.rot90 to make the player's tiles upright
        if k_ccw == 0:
            return bx0 + rx0, by0 + ry0, bx0 + rx1, by0 + ry1
        elif k_ccw == 1:  # 90 CCW: new(x,y) -> old(rw - 1 - y, x)
            return bx0 + (rw - ry1), by0 + rx0, bx0 + (rw - ry0), by0 + rx1
        elif k_ccw == 2:  # 180: new(x,y) -> old(rw - x, rh - y)
            return bx0 + (rw - rx1), by0 + (rh - ry1), bx0 + (rw - rx0), by0 + (rh - ry0)
        else:             # 270 CCW (90 CW): new(x,y) -> old(y, rh - x)
            return bx0 + ry0, by0 + (rh - rx1), bx0 + ry1, by0 + (rh - rx0)

    @staticmethod
    def _top_overhang_is_empty(local_rgb: np.ndarray, cx_probe: float, cy_upright: float, tw: float, th: float) -> bool:
        """
        Check whether the upper overhang of a tall upright tile slot (above a sideways Riichi tile)
        is empty table mat rather than tile face. Prevents upright circular tiles like 1p from
        falsely triggering sideways Riichi detection.
        """
        rh, rw = local_rgb.shape[:2]
        x0 = max(0, int(round(cx_probe - 8)))
        x1 = min(rw, int(round(cx_probe + 9)))
        oy0 = max(0, int(round(cy_upright - th * 0.5 + 2)))
        oy1 = min(rh, int(round(cy_upright - th * 0.5 + 10)))
        by0 = max(0, int(round(cy_upright + 2)))
        by1 = min(rh, int(round(cy_upright + th * 0.5 - 5)))
        if x1 - x0 < 4 or oy1 - oy0 < 3 or by1 - by0 < 4:
            return False

        ov = local_rgb[oy0:oy1, x0:x1].astype(np.float32)
        bd = local_rgb[by0:by1, x0:x1].astype(np.float32)

        ov_gray = 0.299 * ov[..., 0] + 0.587 * ov[..., 1] + 0.114 * ov[..., 2]
        bd_gray = 0.299 * bd[..., 0] + 0.587 * bd[..., 1] + 0.114 * bd[..., 2]
        ov_sat = ov.max(axis=2) - ov.min(axis=2)
        bd_sat = bd.max(axis=2) - bd.min(axis=2)

        # Upright tile face is bright/neutral (even when darkened for tsumogiri); table mat is darker or more saturated
        ov_white_frac = float(((ov_gray > 105.0) & (ov_sat < 48.0)).mean())
        bd_gray_mean = float(bd_gray.mean())
        ov_gray_mean = float(ov_gray.mean())
        ov_sat_mean = float(ov_sat.mean())
        bd_sat_mean = float(bd_sat.mean())

        return (
            ov_white_frac < 0.35
            or (bd_gray_mean - ov_gray_mean > 28.0 and ov_white_frac < 0.55)
            or (ov_sat_mean - bd_sat_mean > 22.0)
        )

    def _scan_river_local(
        self,
        local_rgb: np.ndarray,
        theme: str,
        bev_roi: Tuple[int, int, int, int],
        k_ccw: int,
        w: int,
        h: int,
        expected_origin: Tuple[float, float],
    ) -> Tuple[List[str], bool, List[DetectedBox]]:
        """
        Scan a player-local upright river ROI (tiles flow left-to-right, top-to-bottom).
        Uses inner-face matching crops + world-anchor guidance + geometric overhang/joint-score checks
        to support upright & Riichi tiles while preventing false sideways triggers on 1p/5z/etc.
        """
        tw = BEV_TILE_W
        th = BEV_TILE_H
        default_pitch = 36.8
        crop_uw = max(14, int(round(tw * 0.84)))
        crop_uh = max(18, int(round(th * 0.85)))
        crop_sw, crop_sh = crop_uh, crop_uw

        up_bank = self.mjp_upright_banks[theme]
        up_coarse_bank = self.mjp_upright_coarse_banks[theme]
        side_bank = self.mjp_side_banks[theme]
        rh, rw = local_rgb.shape[:2]

        up_bright = _bright_mask_grid(local_rgb, crop_uh, crop_uw, thr=95, frac=0.25)
        side_bright = _bright_mask_grid(local_rgb, crop_sh, crop_sw, thr=95, frac=0.25)
        up_win_coarse, up_sx_c, up_sy_c = _make_sliding_view(local_rgb, crop_uw, crop_uh, self.mjp_cw, self.mjp_ch)
        up_win_full, up_sx_f, up_sy_f = _make_sliding_view(local_rgb, crop_uw, crop_uh, self.mjp_w, self.mjp_h)
        side_win_full_cache: Optional[Tuple[np.ndarray, float, float]] = None

        def _get_side_win():
            nonlocal side_win_full_cache
            if side_win_full_cache is None:
                side_win_full_cache = _make_sliding_view(local_rgb, crop_sw, crop_sh, self.mjp_h, self.mjp_w)
            return side_win_full_cache

        def _eval_upright(cx_c: float, cy_c: float, r_dx: int = 5, r_dy: int = 6, step: int = 2, try_tight: bool = False):
            x_base = int(round(cx_c - crop_uw * 0.5))
            y_base = int(round(cy_c - crop_uh * 0.5))
            max_x = rw - crop_uw
            max_y = rh - crop_uh
            if try_tight:
                t_coords, t_win_xy = [], []
                for dy in (-1, 0, 1):
                    ny = max(0, min(max_y, y_base + dy))
                    for dx in (-1, 0, 1):
                        nx = max(0, min(max_x, x_base + dx))
                        if up_bright[ny, nx]:
                            t_coords.append((nx + crop_uw * 0.5, ny + crop_uh * 0.5))
                            t_win_xy.append((nx, ny))
                if t_win_xy:
                    ub = _sample_sliding_view(up_win_full, up_sx_f, up_sy_f, t_win_xy)
                    usc, ulbl, ubk = up_bank.match_batch(ub)
                    ui = int(np.argmax(usc))
                    if (not ubk[ui]) and float(usc[ui]) >= 0.62:
                        return (float(usc[ui]), ulbl[ui], False, t_coords[ui])

            rx_s = (r_dx // step) * step
            ry_s = (r_dy // step) * step
            coords, win_xy = [], []
            for dy in range(-ry_s, ry_s + 1, step):
                ny = max(0, min(max_y, y_base + dy))
                for dx in range(-rx_s, rx_s + 1, step):
                    nx = max(0, min(max_x, x_base + dx))
                    if up_bright[ny, nx]:
                        coords.append((nx + crop_uw * 0.5, ny + crop_uh * 0.5))
                        win_xy.append((nx, ny))
            if not win_xy:
                return (-1e9, '', True, (cx_c, cy_c))
            ub = _sample_sliding_view(up_win_full, up_sx_f, up_sy_f, win_xy)
            usc, ulbl, ubk = up_bank.match_batch(ub)
            ui = int(np.argmax(usc))
            best_sc, best_lbl, best_bk, best_c = float(usc[ui]), ulbl[ui], bool(ubk[ui]), coords[ui]
            if step > 1:
                px, py = win_xy[ui]
                p_coords, p_xy = [], []
                for dy in (-1, 0, 1):
                    ny = max(0, min(max_y, py + dy))
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx = max(0, min(max_x, px + dx))
                        if up_bright[ny, nx]:
                            p_coords.append((nx + crop_uw * 0.5, ny + crop_uh * 0.5))
                            p_xy.append((nx, ny))
                if p_xy:
                    pub = _sample_sliding_view(up_win_full, up_sx_f, up_sy_f, p_xy)
                    pusc, pulbl, pubk = up_bank.match_batch(pub)
                    pi = int(np.argmax(pusc))
                    if float(pusc[pi]) > best_sc:
                        best_sc, best_lbl, best_bk, best_c = float(pusc[pi]), pulbl[pi], bool(pubk[pi]), p_coords[pi]
            return (best_sc, best_lbl, best_bk, best_c)

        def _eval_sideways(left_edge_x: float, cy_c: float):
            if left_edge_x + th > rw or cy_c + crop_sh * 0.5 > rh:
                return (-1e9, '', True, (left_edge_x + th * 0.5, cy_c))
            sx_c = left_edge_x + th * 0.5
            sy_c = cy_c + (th - tw) * 0.5
            x_base = int(round(sx_c - crop_sw * 0.5))
            y_base = int(round(sy_c - crop_sh * 0.5))
            max_x = rw - crop_sw
            max_y = rh - crop_sh
            coords, win_xy = [], []
            for dy in range(-4, 5, 2):
                ny = max(0, min(max_y, y_base + dy))
                for dx in range(-4, 5, 2):
                    nx = max(0, min(max_x, x_base + dx))
                    if side_bright[ny, nx]:
                        coords.append((nx + crop_sw * 0.5, ny + crop_sh * 0.5))
                        win_xy.append((nx, ny))
            if not win_xy:
                return (-1e9, '', True, (sx_c, sy_c))
            swin, ssx, ssy = _get_side_win()
            sb = _sample_sliding_view(swin, ssx, ssy, win_xy)
            ssc, slbl, sbk = side_bank.match_batch(sb)
            si = int(np.argmax(ssc))
            return (float(ssc[si]), slbl[si], bool(sbk[si]), coords[si])

        def _emit_box(cx_c: float, cy_c: float, bw: float, bh: float, lbl: str, sc: float, is_rot: bool) -> DetectedBox:
            fx0 = int(round(cx_c - bw * 0.5))
            fy0 = int(round(cy_c - bh * 0.5))
            fx1 = int(round(cx_c + bw * 0.5))
            fy1 = int(round(cy_c + bh * 0.5))
            bx0, by0, bx1, by1 = self._rotate_roi_box_back(fx0, fy0, fx1, fy1, bev_roi, k_ccw)
            sbox = self._bev_box_to_screen(bx0, by0, bx1, by1, None, w, h)
            return DetectedBox(label=lbl, score=sc, box=sbox, is_rotated=is_rot)

        # Anchor-guided origin calibration around expected row 0, col 0 center (coarse-to-fine)
        exp_cx, exp_cy = expected_origin
        best_up_r0 = None
        best_up_r0_sc = -1e9

        up_win_xy0 = []
        x0_base = int(round(exp_cx - crop_uw * 0.5))
        y0_base = int(round(exp_cy - crop_uh * 0.5))
        for dy in range(-20, 21, 3):
            ny = max(0, min(rh - crop_uh, y0_base + dy))
            for dx in range(-24, 25, 3):
                nx = max(0, min(rw - crop_uw, x0_base + dx))
                if up_bright[ny, nx]:
                    up_win_xy0.append((nx, ny))
        if up_win_xy0:
            ub0_c = _sample_sliding_view(up_win_coarse, up_sx_c, up_sy_c, up_win_xy0)
            usc0, _, ubk0 = up_coarse_bank.match_batch(ub0_c)
            face_idxs = [i for i in range(len(usc0)) if not ubk0[i]]
            face_idxs.sort(key=lambda i: -float(usc0[i]))
            peaks0: List[Tuple[int, int]] = []
            for i in face_idxs:
                nx, ny = up_win_xy0[i]
                if any(abs(nx - px) < 18 and abs(ny - py) < 18 for px, py in peaks0):
                    continue
                peaks0.append((nx, ny))
                if len(peaks0) >= 3:
                    break
            if peaks0:
                ref_xy0 = []
                for px, py in peaks0:
                    for dy in range(-2, 3):
                        ny = max(0, min(rh - crop_uh, py + dy))
                        for dx in range(-2, 3):
                            nx = max(0, min(rw - crop_uw, px + dx))
                            if up_bright[ny, nx]:
                                ref_xy0.append((nx, ny))
                if ref_xy0:
                    ub0_f = _sample_sliding_view(up_win_full, up_sx_f, up_sy_f, ref_xy0)
                    usc0_f, _, ubk0_f = up_bank.match_batch(ub0_f)
                    for i in range(len(usc0_f)):
                        if not ubk0_f[i] and float(usc0_f[i]) > best_up_r0_sc:
                            best_up_r0_sc = float(usc0_f[i])
                            nx, ny = ref_xy0[i]
                            best_up_r0 = (nx + crop_uw * 0.5, ny + crop_uh * 0.5)

        best_r0 = best_up_r0
        best_r0_sc = best_up_r0_sc

        # Only check sideways col=0 origin if upright col=0 is weak/missing
        if best_up_r0 is None or best_up_r0_sc < 0.46:
            side_coords0, side_win_xy0 = [], []
            sx0_base = int(round(exp_cx + (th - tw) * 0.5 - crop_sw * 0.5))
            sy0_base = int(round(exp_cy + (th - tw) * 0.5 - crop_sh * 0.5))
            for dy in range(-16, 17, 3):
                ny = max(0, min(rh - crop_sh, sy0_base + dy))
                for dx in range(-20, 21, 3):
                    nx = max(0, min(rw - crop_sw, sx0_base + dx))
                    if side_bright[ny, nx]:
                        cy_up = (ny + crop_sh * 0.5) - (th - tw) * 0.5
                        cx_up = (nx + crop_sw * 0.5) - (th - tw) * 0.5
                        side_coords0.append((cx_up, cy_up, nx + crop_sw * 0.5))
                        side_win_xy0.append((nx, ny))
            if side_win_xy0:
                swin, ssx, ssy = _get_side_win()
                sb0 = _sample_sliding_view(swin, ssx, ssy, side_win_xy0)
                ssc0, _, sbk0 = side_bank.match_batch(sb0)
                best_side_idx = -1
                best_side_sc = -1e9
                for i in range(len(ssc0)):
                    if not sbk0[i] and float(ssc0[i]) > best_side_sc:
                        best_side_sc = float(ssc0[i])
                        best_side_idx = i
                if best_side_idx >= 0 and best_side_sc >= 0.60:
                    cx_u, cy_u, fx_s = side_coords0[best_side_idx]
                    if self._top_overhang_is_empty(local_rgb, fx_s, cy_u, tw, th):
                        cy_u = max(exp_cy - 12.0, min(exp_cy + 12.0, cy_u))
                        best_r0 = (cx_u, cy_u)
                        best_r0_sc = best_side_sc

        # Fallback: if col=0 is obscured, probe col=1 at exp_cx + default_pitch
        if best_r0 is None or best_r0_sc < 0.48:
            c1_coords, c1_win_xy = [], []
            x1_base = int(round(exp_cx + default_pitch - crop_uw * 0.5))
            for dy in range(-18, 19, 3):
                ny = max(0, min(rh - crop_uh, y0_base + dy))
                for dx in range(-20, 21, 3):
                    nx = max(0, min(rw - crop_uw, x1_base + dx))
                    if up_bright[ny, nx]:
                        c1_coords.append((nx + crop_uw * 0.5 - default_pitch, ny + crop_uh * 0.5))
                        c1_win_xy.append((nx, ny))
            if c1_win_xy:
                ub1 = _sample_sliding_view(up_win_full, up_sx_f, up_sy_f, c1_win_xy)
                usc1, _, ubk1 = up_bank.match_batch(ub1)
                for i in range(len(usc1)):
                    if not ubk1[i] and float(usc1[i]) >= 0.54 and float(usc1[i]) > best_r0_sc:
                        best_r0_sc = float(usc1[i])
                        best_r0 = c1_coords[i]

        if best_r0 is None or best_r0_sc < 0.48:
            return [], False, []

        x_origin, y_origin = best_r0
        tiles: List[str] = []
        riichi = False
        boxes: List[DetectedBox] = []

        # Up to 4 rows in river (rows 0..1 max 6 tiles, row 2+ overflow up to 12)
        for row in range(4):
            row_base_y = y_origin + row * th
            cy = float(row_base_y)
            if cy - crop_uh * 0.5 < 0 or cy + crop_uh * 0.5 > rh:
                break
            cx = float(x_origin)
            pitch_x = default_pitch
            prev_fx: Optional[float] = None
            max_cols = 6 if row < 2 else 12
            row_found_any = False
            just_after_riichi = False

            col = 0
            while col < max_cols:
                if cx - crop_uw * 0.5 < 0 or cx + crop_uw * 0.5 > rw:
                    break

                r_dx = 8 if just_after_riichi else 5
                r_dy = 7 if just_after_riichi else 6
                best_up = _eval_upright(
                    cx, cy, r_dx=r_dx, r_dy=r_dy,
                    try_tight=(col > 0 and not just_after_riichi),
                )
                up_valid = (not best_up[2]) and (best_up[0] >= 0.48)

                # Evaluate sideways Riichi candidate only if not already Riichi and upright isn't overwhelmingly clear
                side_valid = False
                best_side = (-1e9, '', True, (cx, cy))
                skip_side = False
                if up_valid and best_up[0] >= 0.72 and (best_up[1] not in ROT_SYMMETRIC_TILES):
                    if best_up[0] >= 0.80 or not self._top_overhang_is_empty(local_rgb, best_up[3][0], cy, tw, th):
                        skip_side = True
                if not riichi and not skip_side:
                    left_edge = cx - tw * 0.5
                    best_side = _eval_sideways(left_edge, cy)
                    if (not best_side[2]) and (best_side[0] >= 0.56):
                        fx_s = best_side[3][0]
                        next_cx_up = cx + pitch_x
                        next_cx_side = fx_s + th * 0.5 + tw * 0.5 + 1.0
                        next_up = _eval_upright(next_cx_up, cy, r_dx=6, r_dy=6)
                        next_side = _eval_upright(next_cx_side, cy, r_dx=7, r_dy=6)
                        nu_ok = (not next_up[2]) and (next_up[0] >= 0.48)
                        ns_ok = (not next_side[2]) and (next_side[0] >= 0.48)
                        overhang_empty = self._top_overhang_is_empty(local_rgb, 0.5 * (cx + fx_s), cy, tw, th)

                        if nu_ok or ns_ok:
                            cur_up_sc = max(0.0, best_up[0]) if not best_up[2] else 0.0
                            joint_up = cur_up_sc + (next_up[0] if nu_ok else 0.0)
                            joint_side = best_side[0] + (next_side[0] if ns_ok else 0.0)
                            if (joint_side > joint_up + 0.12 and best_side[0] > best_up[0] + 0.04) or (
                                ns_ok and not nu_ok and overhang_empty and best_side[0] >= 0.58
                            ):
                                side_valid = True
                        else:
                            # Last tile in row: require stricter geometric + margin checks
                            if best_side[1] in ROT_SYMMETRIC_TILES:
                                side_valid = overhang_empty and (best_side[0] >= 0.62) and (best_side[0] > best_up[0] + 0.10)
                            else:
                                side_valid = (
                                    best_side[0] >= 0.58
                                    and best_side[0] > best_up[0] + 0.10
                                    and (overhang_empty or best_side[0] > best_up[0] + 0.18)
                                )

                if side_valid:
                    sc, lbl, _, (fx_c, fy_c) = best_side
                    tiles.append(lbl)
                    riichi = True
                    row_found_any = True
                    just_after_riichi = True
                    boxes.append(_emit_box(fx_c, fy_c, th, tw, lbl, sc, True))
                    cx = fx_c + th * 0.5 + tw * 0.5 + 1.0
                    prev_fx = None
                    col += 1
                elif up_valid:
                    sc, lbl, _, (fx_c, fy_c) = best_up
                    tiles.append(lbl)
                    row_found_any = True
                    just_after_riichi = False
                    boxes.append(_emit_box(fx_c, fy_c, tw, th, lbl, sc, False))
                    cy = max(row_base_y - 6.0, min(row_base_y + 6.0, 0.7 * cy + 0.3 * fy_c))
                    if prev_fx is not None:
                        measured = fx_c - prev_fx
                        if 33.5 <= measured <= 40.0:
                            pitch_x = 0.5 * pitch_x + 0.5 * measured
                    prev_fx = fx_c
                    cx = fx_c + pitch_x
                    col += 1
                else:
                    # Single-tile gap resilience: peek at col+1
                    if col + 1 < max_cols and (cx + pitch_x + crop_uw * 0.5) <= rw:
                        next_up = _eval_upright(cx + pitch_x, cy, r_dx=6, r_dy=6)
                        if (not next_up[2]) and (next_up[0] >= 0.56):
                            if (not best_up[2]) and (best_up[0] >= 0.34):
                                sc, lbl, _, (fx_c, fy_c) = best_up
                                tiles.append(lbl)
                                boxes.append(_emit_box(fx_c, fy_c, tw, th, lbl, sc, False))
                            row_found_any = True
                            just_after_riichi = False
                            prev_fx = None
                            cx = cx + pitch_x
                            col += 1
                            continue
                    break

            if not row_found_any:
                break

        return tiles, riichi, boxes

    def _scan_meld_local(
        self,
        local_rgb: np.ndarray,
        theme: str,
        bev_roi: Tuple[int, int, int, int],
        k_ccw: int,
        w: int,
        h: int,
    ) -> Tuple[List[List[str]], List[DetectedBox]]:
        """
        Scan a player-local meld strip (upright + sideways + ALL 62 tile backs).
        Automatically resolves Ankan (暗杠: 2 backs + 2 faces -> 4 copies of that tile).
        """
        uw = max(14, int(round(BEV_TILE_W * 0.85)))
        uh = max(18, int(round(BEV_TILE_H * 0.85)))
        sw, sh = uh, uw

        up_bank = self.mjp_upright_banks[theme]
        up_coarse_bank = self.mjp_upright_coarse_banks[theme]
        side_bank = self.mjp_side_banks[theme]
        side_coarse_bank = self.mjp_side_coarse_banks[theme]
        rh, rw = local_rgb.shape[:2]

        # Coarse-to-fine scan across the meld strip (vectorized std+bright prefilter -> half-res coarse -> batched full-res ±2px)
        cands: List[Tuple[float, str, bool, bool, int, int, int, int]] = []
        step = 4

        gstd_u = _gray_std_grid(local_rgb, uh, uw)
        br_u = _bright_mask_grid(local_rgb, uh, uw, thr=115, frac=0.25)
        br_u_wh = _bright_mask_grid(local_rgb, uh, uw, thr=180, frac=0.60)
        up_coords = [
            (x, y)
            for y in range(0, max(1, rh - uh + 1), step)
            for x in range(0, max(1, rw - uw + 1), step)
            if (gstd_u[y, x] > 5.0 and br_u[y, x]) or (gstd_u[y, x] > 16.0) or br_u_wh[y, x]
        ]
        if up_coords:
            uwin_c, usx_c, usy_c = _make_sliding_view(local_rgb, uw, uh, self.mjp_cw, self.mjp_ch)
            ub_c = _sample_sliding_view(uwin_c, usx_c, usy_c, up_coords)
            usc, _, ubk = up_coarse_bank.match_batch(ub_c)
            coarse_u = []
            for i, sc in enumerate(usc):
                thr = 0.50 if ubk[i] else 0.36
                if sc >= thr:
                    x, y = up_coords[i]
                    if (not ubk[i]) and (not br_u[y, x]):
                        continue
                    coarse_u.append((float(sc), x, y))
            coarse_u.sort(key=lambda z: -z[0])
            kept_u = []
            for s, cx, cy in coarse_u:
                if any(abs(cx - kx) < uw * 0.55 and abs(cy - ky) < uh * 0.55 for _, kx, ky in kept_u):
                    continue
                kept_u.append((s, cx, cy))
                if len(kept_u) >= 24:
                    break
            if kept_u:
                uwin_f, usx_f, usy_f = _make_sliding_view(local_rgb, uw, uh, self.mjp_w, self.mjp_h)
                r_coords: List[Tuple[int, int]] = []
                for _, cx, cy in kept_u:
                    for dy in range(-2, 3):
                        ny = max(0, min(rh - uh, cy + dy))
                        for dx in range(-2, 3):
                            nx = max(0, min(rw - uw, cx + dx))
                            r_coords.append((nx, ny))
                rb = _sample_sliding_view(uwin_f, usx_f, usy_f, r_coords)
                rsc, rlbl, rbk = up_bank.match_batch(rb)
                for k_idx in range(len(kept_u)):
                    off = k_idx * 25
                    bi = off + int(np.argmax(rsc[off:off + 25]))
                    f_thr = 0.66 if rbk[bi] else 0.52
                    if float(rsc[bi]) >= f_thr:
                        fx, fy = r_coords[bi]
                        if (not rbk[bi]) and float(np.percentile(local_rgb[fy:fy + uh, fx:fx + uw], 75)) < 125.0:
                            continue
                        cands.append((float(rsc[bi]), rlbl[bi], bool(rbk[bi]), False, fx, fy, fx + uw, fy + uh))

        gstd_s = _gray_std_grid(local_rgb, sh, sw)
        br_s = _bright_mask_grid(local_rgb, sh, sw, thr=115, frac=0.25)
        br_s_wh = _bright_mask_grid(local_rgb, sh, sw, thr=180, frac=0.60)
        side_coords = [
            (x, y)
            for y in range(0, max(1, rh - sh + 1), step)
            for x in range(0, max(1, rw - sw + 1), step)
            if (gstd_s[y, x] > 5.0 and br_s[y, x]) or (gstd_s[y, x] > 16.0) or br_s_wh[y, x]
        ]
        if side_coords:
            swin_c, ssx_c, ssy_c = _make_sliding_view(local_rgb, sw, sh, self.mjp_ch, self.mjp_cw)
            sb_c = _sample_sliding_view(swin_c, ssx_c, ssy_c, side_coords)
            ssc, _, sbk = side_coarse_bank.match_batch(sb_c)
            coarse_s = []
            for i, sc in enumerate(ssc):
                thr = 0.50 if sbk[i] else 0.37
                if sc >= thr:
                    x, y = side_coords[i]
                    if (not sbk[i]) and (not br_s[y, x]):
                        continue
                    coarse_s.append((float(sc), x, y))
            coarse_s.sort(key=lambda z: -z[0])
            kept_s = []
            for s, cx, cy in coarse_s:
                if any(abs(cx - kx) < sw * 0.55 and abs(cy - ky) < sh * 0.55 for _, kx, ky in kept_s):
                    continue
                kept_s.append((s, cx, cy))
                if len(kept_s) >= 24:
                    break
            if kept_s:
                swin_f, ssx_f, ssy_f = _make_sliding_view(local_rgb, sw, sh, self.mjp_h, self.mjp_w)
                r_coords = []
                for _, cx, cy in kept_s:
                    for dy in range(-2, 3):
                        ny = max(0, min(rh - sh, cy + dy))
                        for dx in range(-2, 3):
                            nx = max(0, min(rw - sw, cx + dx))
                            r_coords.append((nx, ny))
                rb = _sample_sliding_view(swin_f, ssx_f, ssy_f, r_coords)
                rsc, rlbl, rbk = side_bank.match_batch(rb)
                for k_idx in range(len(kept_s)):
                    off = k_idx * 25
                    bi = off + int(np.argmax(rsc[off:off + 25]))
                    f_thr = 0.66 if rbk[bi] else 0.54
                    if float(rsc[bi]) >= f_thr:
                        fx, fy = r_coords[bi]
                        if (not rbk[bi]) and float(np.percentile(local_rgb[fy:fy + sh, fx:fx + sw], 75)) < 125.0:
                            continue
                        cands.append((float(rsc[bi]), rlbl[bi], bool(rbk[bi]), True, fx, fy, fx + sw, fy + sh))

        if not cands:
            return [], []

        # 2D NMS
        cands.sort(key=lambda c: -c[0])
        kept = []
        for c in cands:
            _, _, _, _, x0, y0, x1, y1 = c
            area_c = (x1 - x0) * (y1 - y0)
            overlap = False
            for k in kept:
                _, _, _, _, kx0, ky0, kx1, ky1 = k
                ix0, iy0 = max(x0, kx0), max(y0, ky0)
                ix1, iy1 = min(x1, kx1), min(y1, ky1)
                if ix1 > ix0 and iy1 > iy0:
                    inter = (ix1 - ix0) * (iy1 - iy0)
                    area_k = (kx1 - kx0) * (ky1 - ky0)
                    if inter / min(area_c, area_k) > 0.25:
                        overlap = True
                        break
            if not overlap:
                kept.append(c)

        # Sort left-to-right and cluster into meld groups
        kept.sort(key=lambda c: c[4])
        groups: List[List[Tuple[float, str, bool, bool, int, int, int, int]]] = []
        for c in kept:
            if not groups or (c[4] - groups[-1][-1][6]) > 18:
                groups.append([c])
            else:
                groups[-1].append(c)

        meld_groups: List[List[str]] = []
        boxes: List[DetectedBox] = []
        for grp in groups:
            if len(grp) < 3:
                continue  # Ignore stray isolated noise
            backs = [g for g in grp if g[2]]
            faces = [g for g in grp if not g[2]]
            if not faces:
                continue  # Never emit a meld group with zero face tiles
            for sc, lbl, is_bk, is_rot, x0, y0, x1, y1 in grp:
                bx0, by0, bx1, by1 = self._rotate_roi_box_back(x0, y0, x1, y1, bev_roi, k_ccw)
                sbox = self._bev_box_to_screen(bx0, by0, bx1, by1, None, w, h)
                boxes.append(DetectedBox(label=lbl, score=sc, box=sbox, is_back=is_bk, is_rotated=is_rot))

            # Ankan (暗杠): 2 tile backs + 2 face-up tiles (or 1+ back + 2 identical faces) -> 4 copies
            if (len(backs) >= 2) or (len(backs) >= 1 and len(faces) == 2 and faces[0][1] == faces[1][1]):
                face_code = faces[0][1]
                meld_groups.append([face_code] * 4)
            else:
                meld_groups.append([f[1] for f in faces])

        return meld_groups, boxes

    @staticmethod
    def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
        ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
        if ix1 <= ix0 or iy1 <= iy0:
            return 0.0
        inter = float((ix1 - ix0) * (iy1 - iy0))
        area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
        area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
        return inter / min(area_a, area_b)

    def _scan_kita_local(
        self,
        bev: np.ndarray,
        p_idx: int,
        theme: str,
        w: int,
        h: int,
        exclude_boxes: List[DetectedBox],
    ) -> List[DetectedBox]:
        """Scan the babei_p anchor zone in BEV for peeled North (4z) Kita bonus tiles."""
        wx, wz = BABEI_ANCHORS[p_idx]
        k_ccw = [0, 3, 2, 1][p_idx]
        bx, by = world_to_bev(wx, wz)
        rx0 = max(0, int(round(bx - 180)))
        ry0 = max(0, int(round(by - 180)))
        rx1 = min(BEV_SIZE, int(round(bx + 180)))
        ry1 = min(BEV_SIZE, int(round(by + 180)))
        if rx1 - rx0 < 40 or ry1 - ry0 < 40:
            return []

        bev_roi = (rx0, ry0, rx1, ry1)
        local_rgb = np.rot90(bev[ry0:ry1, rx0:rx1], k=k_ccw)
        rh, rw = local_rgb.shape[:2]
        ax_l, ay_l = _world_to_bev_roi_local(wx, wz, bev_roi, k_ccw)

        tw, th = BEV_TILE_W, BEV_TILE_H
        uw = max(14, int(round(tw * 0.84)))
        uh = max(18, int(round(th * 0.85)))
        up_bank = self.mjp_upright_banks[theme]
        up_coarse_bank = self.mjp_upright_coarse_banks[theme]

        x_min = max(0, int(round(ax_l - 165 - uw * 0.5)))
        x_max = min(rw - uw, int(round(ax_l + 55 - uw * 0.5)))
        y_min = max(0, int(round(ay_l - 38 - uh * 0.5)))
        y_max = min(rh - uh, int(round(ay_l + 42 - uh * 0.5)))
        if x_max < x_min or y_max < y_min:
            return []

        br_k = _bright_mask_grid(local_rgb, uh, uw, thr=110, frac=0.25)
        coords = [
            (x, y)
            for y in range(y_min, y_max + 1, 3)
            for x in range(x_min, x_max + 1, 3)
            if br_k[y, x]
        ]
        if not coords:
            return []

        uwin_c, usx_c, usy_c = _make_sliding_view(local_rgb, uw, uh, self.mjp_cw, self.mjp_ch)
        ub_c = _sample_sliding_view(uwin_c, usx_c, usy_c, coords)
        sc, lbl, bk = up_coarse_bank.match_batch(ub_c)
        hits = [
            (float(sc[i]), coords[i][0], coords[i][1])
            for i in range(len(sc))
            if (not bk[i]) and lbl[i] == '4z' and float(sc[i]) >= 0.45
        ]
        if not hits:
            return []

        hits.sort(key=lambda z: -z[0])
        peaks: List[Tuple[float, int, int]] = []
        for s, x, y in hits:
            cx, cy = x + uw * 0.5, y + uh * 0.5
            if any(abs(cx - (px + uw * 0.5)) < 24.0 and abs(cy - (py + uh * 0.5)) < 24.0 for _, px, py in peaks):
                continue
            peaks.append((s, x, y))

        uwin_f, usx_f, usy_f = _make_sliding_view(local_rgb, uw, uh, self.mjp_w, self.mjp_h)
        r_coords: List[Tuple[int, int]] = []
        for _, px, py in peaks:
            for dy in range(-2, 3):
                ny = max(0, min(rh - uh, py + dy))
                for dx in range(-2, 3):
                    nx = max(0, min(rw - uw, px + dx))
                    r_coords.append((nx, ny))
        rb = _sample_sliding_view(uwin_f, usx_f, usy_f, r_coords)
        rsc, rlbl, rbk = up_bank.match_batch(rb)

        refined: List[Tuple[float, float, float, DetectedBox]] = []
        for p_i in range(len(peaks)):
            off = p_i * 25
            bi = off + int(np.argmax(rsc[off:off + 25]))
            if rbk[bi] or rlbl[bi] != '4z' or float(rsc[bi]) < 0.56:
                continue
            fx, fy = r_coords[bi]
            cx, cy = fx + uw * 0.5, fy + uh * 0.5
            fx0 = int(round(cx - tw * 0.5))
            fy0 = int(round(cy - th * 0.5))
            fx1 = int(round(cx + tw * 0.5))
            fy1 = int(round(cy + th * 0.5))
            bx0, by0, bx1, by1 = self._rotate_roi_box_back(fx0, fy0, fx1, fy1, bev_roi, k_ccw)
            sbox = self._bev_box_to_screen(bx0, by0, bx1, by1, None, w, h)
            if any(self._box_iou(sbox, eb.box) > 0.25 for eb in exclude_boxes):
                continue
            refined.append((float(rsc[bi]), cx, cy, DetectedBox(label='4z', score=float(rsc[bi]), box=sbox)))

        if not refined:
            return []
        if len(refined) > 1:
            med_cy = float(np.median([r[2] for r in refined]))
            refined = [r for r in refined if abs(r[2] - med_cy) <= 16.0]
        refined.sort(key=lambda r: -r[1])  # right-to-left (anchor grows leftward)
        return [r[3] for r in refined[:4]]

    @staticmethod
    def _compass_left_is_empty(bev: np.ndarray) -> Tuple[bool, int]:
        """Check whether the left player's center compass score display is empty (3P layout signal)."""
        def _score_bright(y0: int, y1: int, x0: int, x1: int) -> int:
            p = bev[y0:y1, x0:x1]
            return int(((p.max(axis=2) > 145) & (p[:, :, 0] > 115)).sum())

        b_cnt = _score_bright(675, 725, 580, 700)
        r_cnt = _score_bright(580, 700, 675, 725)
        t_cnt = _score_bright(555, 605, 580, 700)
        l_cnt = _score_bright(580, 700, 555, 605)
        med_other = float(np.median([b_cnt, r_cnt, t_cnt]))
        compass_left_empty = (med_other >= 40.0) and (l_cnt < 0.18 * med_other)
        return compass_left_empty, l_cnt

    @classmethod
    def _detect_player_count(
        cls,
        bev: np.ndarray,
        hand_boxes: List[DetectedBox],
        dora_boxes: List[DetectedBox],
        river_boxes: List[List[DetectedBox]],
        meld_boxes: List[List[DetectedBox]],
        kita_boxes: List[List[DetectedBox]],
    ) -> int:
        """Auto-detect 3-player (三麻) vs 4-player (四麻) table layout from tiles and BEV geometry."""
        # Hard 4P signal: any confident 2m..8m/0m tile anywhere on board
        all_tiles = list(hand_boxes) + list(dora_boxes)
        for bl in river_boxes + meld_boxes:
            all_tiles.extend(bl)
        if any(b.label in SANMA_EXCLUDED_TILES and b.score >= 0.62 for b in all_tiles):
            return 4
        if len(river_boxes[3]) > 0 or len(meld_boxes[3]) > 0:
            return 4

        compass_left_empty, l_cnt = cls._compass_left_is_empty(bev)

        # Left vs Right standing hand wall in BEV
        r_wall = bev[320:960, 1080:1240]
        l_wall = bev[320:960, 40:200]
        r_white = float(((r_wall[:, :, 0] > 175) & (r_wall[:, :, 1] > 175) & (r_wall[:, :, 2] > 175)).mean())
        l_white = float(((l_wall[:, :, 0] > 175) & (l_wall[:, :, 1] > 175) & (l_wall[:, :, 2] > 175)).mean())
        wall_left_empty = (r_white >= 0.008) and (l_white < 0.25 * r_white)

        has_kita = any(b.score >= 0.60 for kb in kita_boxes for b in kb)
        if (compass_left_empty and wall_left_empty) or (compass_left_empty and l_cnt <= 10) or (compass_left_empty and has_kita):
            return 3
        return 4

    def recognize_frames(
        self,
        frames: List[np.ndarray],
        theme_pref: str = 'auto',
        mode_pref: str = 'auto',
        hint_theme: Optional[str] = None,
    ) -> RecognitionResult:
        """
        Recognize the full Mahjong Soul board from 1..N RGB frames (latest frame + temporal glint calibration).
        Supports arbitrary resolutions, 3-player / 4-player auto-detection, and Kita (拔北宝牌) detection.
        """
        assert len(frames) >= 1
        orig_h, orig_w = frames[-1].shape[:2]
        norm_frames, (cx0, cy0, sx, sy) = self._normalize_frames(frames)
        latest = norm_frames[-1]
        h, w = latest.shape[:2]

        # 1. Self hand + theme detection (use cached hint_theme when available, falling back to 'auto' if weak)
        if theme_pref == 'auto' and hint_theme in ('default', 'cat'):
            theme, hand_boxes = self._recognize_hand_and_theme(norm_frames, hint_theme)
            mean_h = (sum(b.score for b in hand_boxes) / len(hand_boxes)) if hand_boxes else 0.0
            if len(hand_boxes) < 4 or mean_h < 0.58:
                theme, hand_boxes = self._recognize_hand_and_theme(norm_frames, 'auto')
        else:
            theme, hand_boxes = self._recognize_hand_and_theme(norm_frames, theme_pref)

        # 2. Run 2D dora recognition concurrently with single BEV warp (temporal min before perspective warp)
        dora_fut = self._pool.submit(self._recognize_dora_ui, norm_frames, theme)

        if len(norm_frames) > 1:
            y_cut = max(1, int(h * 0.85))
            diff = float(np.mean(np.abs(
                norm_frames[-1][:y_cut].astype(np.int16) - norm_frames[0][:y_cut].astype(np.int16)
            )))
            if diff < 16.0:
                table_frame = norm_frames[0]
                for nf in norm_frames[1:]:
                    table_frame = np.minimum(table_frame, nf)
            else:
                table_frame = norm_frames[-1]
        else:
            table_frame = norm_frames[0]

        if hand_boxes:
            f_masked = table_frame.copy()
            for hb in hand_boxes:
                hx0 = max(0, hb.box[0] - 10)
                hy0 = max(0, hb.box[1] - 10)
                hx1 = min(w, hb.box[2] + 10)
                hy1 = min(h, hb.box[3] + 10)
                f_masked[hy0:hy1, hx0:hx1] = (28, 72, 54)
            span_x0 = max(0, min(hb.box[0] for hb in hand_boxes) - 12)
            span_x1 = min(w, max(hb.box[2] for hb in hand_boxes) + 12)
            span_y0 = max(0, min(hb.box[1] for hb in hand_boxes) - 12)
            f_masked[span_y0:h, span_x0:span_x1] = (28, 72, 54)
            bev, _ = warp_table_to_bev(Image.fromarray(f_masked))
        else:
            bev, _ = warp_table_to_bev(Image.fromarray(table_frame))

        # Define BEV ROIs for all 4 rivers (hai_1..4) and 4 meld zones (ming_1..4)
        river_rois = [
            # (x0, y0, x1, y1, k_rot90_ccw)
            (490, 735, 900, 1005, 0),   # 0: self (bottom, upright)
            (730, 415, 1005, 800, 3),   # 1: right/shimocha (rotate 90 CW / k=3 to make upright)
            (395, 290, 785, 565, 2),    # 2: top/toimen (rotate 180 / k=2)
            (280, 500, 555, 895, 1),    # 3: left/kamicha (rotate 90 CCW / k=1)
        ]

        meld_rois = [
            (620, 1015, 1130, 1150, 0),  # 0: self melds (bottom-right)
            (1015, 210, 1145, 720, 3),   # 1: right melds (top-right)
            (150, 195, 660, 325, 2),     # 2: top melds (top-left)
            (140, 560, 275, 1090, 1),    # 3: left melds (bottom-left)
        ]

        # 3. Scan all 4 rivers and active melds in parallel across CPU cores
        river_futs = []
        for p_idx, (roi_x0, roi_y0, roi_x1, roi_y1, k_ccw) in enumerate(river_rois):
            bev_roi = (roi_x0, roi_y0, roi_x1, roi_y1)
            local = np.ascontiguousarray(np.rot90(bev[roi_y0:roi_y1, roi_x0:roi_x1], k=k_ccw))
            ax, az = HAI_ANCHORS[p_idx]
            exp_origin = _world_to_bev_roi_local(ax, az, bev_roi, k_ccw)
            river_futs.append(self._pool.submit(
                self._scan_river_local, local, theme, bev_roi, k_ccw, w, h, exp_origin
            ))

        max_self_melds = max(0, (14 - len(hand_boxes)) // 3)
        meld_futs = [None] * 4
        for p_idx, (roi_x0, roi_y0, roi_x1, roi_y1, k_ccw) in enumerate(meld_rois):
            if p_idx == 0 and max_self_melds == 0:
                continue
            local = np.ascontiguousarray(np.rot90(bev[roi_y0:roi_y1, roi_x0:roi_x1], k=k_ccw))
            meld_futs[p_idx] = self._pool.submit(
                self._scan_meld_local, local, theme, (roi_x0, roi_y0, roi_x1, roi_y1), k_ccw, w, h
            )

        dora_boxes = dora_fut.result()

        rivers: List[List[str]] = []
        riichi: List[bool] = []
        river_boxes: List[List[DetectedBox]] = []
        for fut in river_futs:
            t_list, is_r, b_list = fut.result()
            rivers.append(t_list)
            riichi.append(is_r)
            river_boxes.append(b_list)

        melds: List[List[List[str]]] = []
        meld_boxes: List[List[DetectedBox]] = []
        for p_idx, fut in enumerate(meld_futs):
            if fut is None:
                melds.append([])
                meld_boxes.append([])
                continue
            m_grps, b_list = fut.result()
            if p_idx == 0 and len(m_grps) > max_self_melds:
                m_grps = m_grps[:max_self_melds]
                b_list = b_list[:max_self_melds * 4]
            melds.append(m_grps)
            meld_boxes.append(b_list)

        # 4. Skip Kita scanning early when layout is guaranteed 4-player
        skip_kita = (mode_pref == '4p')
        if not skip_kita and mode_pref == 'auto':
            all_tiles = list(hand_boxes) + list(dora_boxes)
            for bl in river_boxes + meld_boxes:
                all_tiles.extend(bl)
            if (
                any(b.label in SANMA_EXCLUDED_TILES and b.score >= 0.62 for b in all_tiles)
                or len(river_boxes[3]) > 0
                or len(meld_boxes[3]) > 0
                or not self._compass_left_is_empty(bev)[0]
            ):
                skip_kita = True

        kita_boxes: List[List[DetectedBox]] = [[], [], [], []]
        if not skip_kita:
            kita_futs = {}
            for p_idx in (0, 1, 2):
                excl = river_boxes[p_idx] + meld_boxes[p_idx]
                kita_futs[p_idx] = self._pool.submit(self._scan_kita_local, bev, p_idx, theme, w, h, excl)
            for p_idx, fut in kita_futs.items():
                kita_boxes[p_idx] = fut.result()
            # Globally cap total Kita tiles across all players at <= 4
            all_k = [(b.score, p_idx, idx, b) for p_idx in (0, 1, 2) for idx, b in enumerate(kita_boxes[p_idx])]
            if len(all_k) > 4:
                all_k.sort(key=lambda z: -z[0])
                keep_set = {(z[1], z[2]) for z in all_k[:4]}
                for p_idx in (0, 1, 2):
                    kita_boxes[p_idx] = [b for idx, b in enumerate(kita_boxes[p_idx]) if (p_idx, idx) in keep_set]

        # 5. Determine 3-player vs 4-player mode
        if mode_pref == '3p':
            player_count = 3
        elif mode_pref == '4p':
            player_count = 4
        else:
            player_count = self._detect_player_count(bev, hand_boxes, dora_boxes, river_boxes, meld_boxes, kita_boxes)

        if player_count == 3:
            rivers[3] = []
            riichi[3] = False
            river_boxes[3] = []
            melds[3] = []
            meld_boxes[3] = []
            kita_counts = [len(kb) for kb in kita_boxes]
        else:
            kita_boxes = [[], [], [], []]
            kita_counts = [0, 0, 0, 0]

        # Remap all bounding boxes back to original capture coordinates
        hand_boxes = self._remap_boxes(hand_boxes, cx0, cy0, sx, sy, orig_w, orig_h)
        dora_boxes = self._remap_boxes(dora_boxes, cx0, cy0, sx, sy, orig_w, orig_h)
        river_boxes = [self._remap_boxes(bl, cx0, cy0, sx, sy, orig_w, orig_h) for bl in river_boxes]
        meld_boxes = [self._remap_boxes(bl, cx0, cy0, sx, sy, orig_w, orig_h) for bl in meld_boxes]
        kita_boxes = [self._remap_boxes(bl, cx0, cy0, sx, sy, orig_w, orig_h) for bl in kita_boxes]

        board = BoardState(
            hand=[b.label for b in hand_boxes],
            dora_indicators=[b.label for b in dora_boxes],
            rivers=rivers,
            melds=melds,
            riichi=riichi,
            theme=theme,
            player_count=player_count,
            kita=kita_counts,
        )

        all_scores = [b.score for b in hand_boxes] + [b.score for b in dora_boxes]
        for bl in river_boxes + meld_boxes + kita_boxes:
            all_scores.extend(b.score for b in bl)
        avg_conf = float(np.mean(all_scores)) if all_scores else 0.0

        pnames = ['自家', '下家', '上家', ''] if player_count == 3 else ['自家', '下家', '对家', '上家']
        active_indices = (0, 1, 2) if player_count == 3 else (0, 1, 2, 3)
        notes = []
        for p in active_indices:
            if riichi[p]:
                notes.append(f'{pnames[p]}已立直')
            for grp in melds[p]:
                if len(grp) == 4 and len(set(grp)) == 1:
                    notes.append(f'{pnames[p]}副露/暗杠 {grp[0]}')
            if player_count == 3 and kita_counts[p] > 0:
                notes.append(f'{pnames[p]}已拔北×{kita_counts[p]}')

        return RecognitionResult(
            board=board,
            detected_theme=theme,
            hand_boxes=hand_boxes,
            dora_boxes=dora_boxes,
            river_boxes=river_boxes,
            meld_boxes=meld_boxes,
            kita_boxes=kita_boxes,
            avg_confidence=avg_conf,
            notes=notes,
        )


def draw_annotations(img: Image.Image, rec: RecognitionResult) -> Image.Image:
    """Draw detected bounding boxes & labels onto a copy of the capture for visual verification."""
    vis = img.copy()
    draw = ImageDraw.Draw(vis)

    def _draw_list(boxes: List[DetectedBox], color: Tuple[int, int, int], suffix: str = ''):
        for b in boxes:
            x0, y0, x1, y1 = b.box
            draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
            txt = 'BK' if b.is_back else (b.label + ('*' if b.is_rotated else '') + suffix)
            draw.rectangle([x0, max(0, y0 - 14), x0 + max(36, len(txt) * 7 + 4), y0], fill=color)
            draw.text((x0 + 2, max(0, y0 - 13)), txt, fill=(0, 0, 0))

    _draw_list(rec.hand_boxes, (0, 230, 120))
    _draw_list(rec.dora_boxes, (255, 200, 0))
    for rb in rec.river_boxes:
        _draw_list(rb, (80, 180, 255))
    for mb in rec.meld_boxes:
        _draw_list(mb, (255, 110, 180))
    for kb in rec.kita_boxes:
        _draw_list(kb, (190, 120, 255), suffix='K')
    return vis
