import os
import math
from glob import glob
import numpy as np
import cv2
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd

# ------------ User Configuration ------------
IMG_DIR = "img/"
MASK_DIR = "mask/"

OUT_DIR = os.path.join('.', 'cobb10_amace_cobb_results')
VIS_DIR = os.path.join(OUT_DIR)
os.makedirs(VIS_DIR, exist_ok=True)

NAME_CLASSES = ["_background_", "L1", "L2", "L3", "L4", "L5",
                "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]
TARGET_VERTEBRAE = ["L1", "L2", "L3", "L4", "L5", "S1"]

ROI_PAD_Y = 60
ROI_PAD_X = 80
MIN_VERTEBRA_PIXELS = 50
CURV_SAMPLES = 1200
CURV_PEAK_REL_H = 0.12
CURV_BOUND_FRAC = 0.2
CURV_OFFSET = 40
CURV_SCALE = 200.0
SAVE_DPI = 300

CURV_SMOOTH_WINDOW = 7
PEAK_PROMINENCE_RATIO = 0.15
MAX_BEND_REGIONS = 3
PCA_WEIGHT_EDGE_FACTOR = 2.5

COLOR_MAP = [
    (0, 0, 0), (200, 80, 20), (30, 160, 40), (200, 200, 40), (30, 30, 160), (200, 30, 160),
    (30, 160, 160), (160, 160, 160), (120, 40, 40), (240, 40, 40), (240, 180, 40), (120, 180, 40)
]

BEND_REGION_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 10


# ------------------------------------------------------
# Utility Functions
# ------------------------------------------------------

def read_image(path):
    """
    Read an image file and convert from BGR to RGB.
    OpenCV reads in BGR by default, but matplotlib expects RGB.
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find or read image file: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_index(path):
    """
    Read a mask image and convert to integer index format.
    If mask is 3-channel and all channels are identical, take the first channel;
    otherwise convert to grayscale.
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Cannot find or read mask file: {path}")
    if m.ndim == 3:
        if np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 1] == m[..., 2]):
            m = m[..., 0]
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    return m.astype(np.int32)


def largest_component(mask_bool):
    """Extract the largest connected component from a boolean mask."""
    labels = measure.label(mask_bool, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    props = measure.regionprops(labels)
    lab = props[np.argmax([p.area for p in props])].label
    return labels == lab


def find_edge_points(coords_xy):
    """
    Find edge points from a set of points.
    Prefer convex hull; fall back to bounding box if hull fails.
    """
    if len(coords_xy) < 10:
        return []
    try:
        hull = ConvexHull(coords_xy)
        return coords_xy[hull.vertices]
    except:
        x_min, x_max = coords_xy[:, 0].min(), coords_xy[:, 0].max()
        y_min, y_max = coords_xy[:, 1].min(), coords_xy[:, 1].max()
        edge_mask = (
                (coords_xy[:, 0] == x_min) | (coords_xy[:, 0] == x_max) |
                (coords_xy[:, 1] == y_min) | (coords_xy[:, 1] == y_max)
        )
        return coords_xy[edge_mask]


def improved_pca_axis_from_coords(coords_xy, min_points=10):
    """
    Improved PCA to estimate the main axis of a vertebra from its coordinate points.
    Improvements:
    1. Remove outliers.
    2. Assign higher weights to edge points.
    """
    if coords_xy.shape[0] < min_points:
        return None, None

    # 1. Remove outliers
    centroid = coords_xy.mean(axis=0)
    distances = cdist([centroid], coords_xy)[0]
    valid_mask = distances < np.percentile(distances, 90)
    filtered_coords = coords_xy[valid_mask]

    if filtered_coords.shape[0] < min_points:
        filtered_coords = coords_xy

    # 2. Weighted PCA - give higher weight to edge points
    edge_points = find_edge_points(filtered_coords)
    weights = np.ones(len(filtered_coords))
    if len(edge_points) > 0:
        for i, point in enumerate(filtered_coords):
            if any(np.all(point == edge_point) for edge_point in edge_points):
                weights[i] = PCA_WEIGHT_EDGE_FACTOR

    # 3. Perform weighted PCA
    centroid = np.average(filtered_coords, axis=0, weights=weights)
    centered = filtered_coords - centroid
    weighted_cov = np.cov(centered.T, aweights=weights)

    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]

    return centroid, main_axis


def angle_between_vectors_deg(v1, v2):
    """Compute the angle (in degrees) between two vectors."""
    v1 = np.array(v1)
    v2 = np.array(v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    ang = math.degrees(math.acos(cosang))
    return ang


def normalize_vector(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def calculate_endplate_angle_improved(results):
    """
    Improved endplate angle calculation.
    Ensure consistency of endplate direction (perpendicular to vertebra main axis)
    and compute the acute angle.
    """
    if 'L1' not in results or 'L5' not in results:
        return None

    top_axis = results['L1']['axis']
    bot_axis = results['L5']['axis']

    if top_axis is None or bot_axis is None:
        return None

    # Endplate direction is approximately perpendicular to the main axis
    end_top = normalize_vector(np.array([-top_axis[1], top_axis[0]]))
    end_bot = normalize_vector(np.array([-bot_axis[1], bot_axis[0]]))

    # Force endplate direction upward (negative y) for stable angle calculation
    if end_top[1] > 0:
        end_top = -end_top
    if end_bot[1] > 0:
        end_bot = -end_bot

    angle = angle_between_vectors_deg(end_top, end_bot)

    if angle > 90.0:
        angle = 180.0 - angle

    return angle


def fit_spine_spline(centers):
    """
    Fit a smooth cubic spline through a sequence of vertebral centers.
    Returns spline functions for x and y, and the normalized arc-length parameter.
    """
    pts = np.array(centers)
    if pts.shape[0] < 2:
        return None, None, None

    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    t = np.concatenate(([0.0], np.cumsum(seglen)))

    if t[-1] == 0:
        t = np.linspace(0, 1, len(pts))
    else:
        t = t / t[-1]

    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])
    return cs_x, cs_y, t


def compute_curvature(cs_x, cs_y, num_samples=CURV_SAMPLES):
    """Compute curvature along the fitted spline."""
    s_vals = np.linspace(0.0, 1.0, num_samples)

    dx = cs_x.derivative(1)(s_vals)
    dy = cs_y.derivative(1)(s_vals)
    ddx = cs_x.derivative(2)(s_vals)
    ddy = cs_y.derivative(2)(s_vals)

    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5

    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.nan_to_num(num / den)

    speed = np.sqrt(dx * dx + dy * dy)

    return s_vals, dx, dy, ddx, ddy, k, speed


def smooth_curvature(k, window_size=CURV_SMOOTH_WINDOW):
    """Smooth curvature using Savitzky-Golay filter, fallback to moving average."""
    try:
        return savgol_filter(k, window_size, 2)
    except:
        return np.convolve(k, np.ones(window_size) / window_size, mode='same')


def detect_multiple_peaks(k, s_vals, min_prominence=None):
    """Detect multiple peaks on smoothed curvature using prominence."""
    if min_prominence is None:
        min_prominence = PEAK_PROMINENCE_RATIO * np.max(k) if np.max(k) > 0 else 0.1

    peaks, properties = find_peaks(k, prominence=min_prominence)

    peaks_info = []
    for i, peak_idx in enumerate(peaks):
        peaks_info.append({
            'index': peak_idx,
            's_value': s_vals[peak_idx],
            'curvature': k[peak_idx],
            'prominence': properties['prominences'][i] if 'prominences' in properties else 0
        })

    peaks_info.sort(key=lambda x: x['curvature'], reverse=True)
    return peaks_info


def detect_main_bend(k, s_vals, rel_peak_h=CURV_PEAK_REL_H, bound_frac=CURV_BOUND_FRAC, peak_idx=None):
    """
    Given a peak index, determine the left and right boundaries of its bend region.
    Boundaries are where curvature drops to a fraction of the peak value.
    """
    if peak_idx is None:
        peak_idx = int(np.argmax(k))

    maxk = np.max(k)
    if maxk <= 0:
        return s_vals[max(peak_idx - 1, 0)], s_vals[min(peak_idx + 1, len(s_vals) - 1)], peak_idx

    peak_k = k[peak_idx]
    thr = max(peak_k * bound_frac, 1e-9)

    left = peak_idx
    while left > 0 and k[left] > thr:
        left -= 1

    right = peak_idx
    while right < len(k) - 1 and k[right] > thr:
        right += 1

    left = max(left - 1, 0)
    right = min(right + 1, len(k) - 1)

    return s_vals[left], s_vals[right], peak_idx


def identify_main_bend_regions(peaks_info, k, s_vals, num_regions=MAX_BEND_REGIONS):
    """Identify main bend regions from detected peaks."""
    regions = []

    for i, peak in enumerate(peaks_info[:num_regions]):
        peak_idx = peak['index']
        s_left, s_right, _ = detect_main_bend(k, s_vals, peak_idx=peak_idx)

        regions.append({
            'peak_s': s_vals[peak_idx],
            's_left': s_left,
            's_right': s_right,
            'peak_curvature': k[peak_idx],
            'region_index': i
        })

    return regions


def improved_curvature_analysis(cs_x, cs_y, num_samples=CURV_SAMPLES):
    """
    Integrate improved curvature analysis steps:
    1. Compute raw curvature.
    2. Smooth curvature curve.
    3. Detect multiple curvature peaks.
    4. Identify bend regions for each peak.
    """
    s_vals, dx, dy, ddx, ddy, k, speed = compute_curvature(cs_x, cs_y, num_samples)
    k_smooth = smooth_curvature(k)
    peaks_info = detect_multiple_peaks(k_smooth, s_vals)
    main_bend_regions = identify_main_bend_regions(peaks_info, k_smooth, s_vals)

    return s_vals, k_smooth, speed, main_bend_regions, peaks_info


def find_nearest_s_for_point(cs_x, cs_y, point, s_vals=np.linspace(0, 1, 1000)):
    """Find s value on the spline closest to the given point."""
    ptsx = cs_x(s_vals)
    ptsy = cs_y(s_vals)
    d2 = (ptsx - point[0]) ** 2 + (ptsy - point[1]) ** 2
    return s_vals[np.argmin(d2)]


def line_segment_within_bbox(pt_rel, v, w, h, eps=1e-6):
    """
    Compute intersections of a line through pt_rel along vector v with the bounding box [0,w]x[0,h].
    Returns the segment endpoints inside the box.
    """
    x0, y0 = float(pt_rel[0]), float(pt_rel[1])
    vx, vy = float(v[0]), float(v[1])
    candidates = []

    if abs(vx) > eps:
        t1 = (0 - x0) / vx
        y1 = y0 + t1 * vy
        if -eps <= y1 <= h + eps:
            candidates.append((x0 + t1 * vx, y1, t1))
        t2 = (w - x0) / vx
        y2 = y0 + t2 * vy
        if -eps <= y2 <= h + eps:
            candidates.append((x0 + t2 * vx, y2, t2))

    if abs(vy) > eps:
        t3 = (0 - y0) / vy
        x3 = x0 + t3 * vx
        if -eps <= x3 <= w + eps:
            candidates.append((x3, y0 + t3 * vy, t3))
        t4 = (h - y0) / vy
        x4 = x0 + t4 * vx
        if -eps <= x4 <= w + eps:
            candidates.append((x4, y0 + t4 * vy, t4))

    if len(candidates) >= 2:
        candidates_sorted = sorted(candidates, key=lambda it: it[2])
        p1 = (candidates_sorted[0][0], candidates_sorted[0][1])
        p2 = (candidates_sorted[-1][0], candidates_sorted[-1][1])
        return p1, p2
    else:
        # Fallback: create a short segment centered at pt_rel
        seg_half = min(w, h) * 0.25
        vnorm = np.array([vx, vy], dtype=float)
        nrm = np.linalg.norm(vnorm)
        if nrm < eps:
            p1 = (max(0, min(w, x0 - 2)), max(0, min(h, y0 - 2)))
            p2 = (max(0, min(w, x0 + 2)), max(0, min(h, y0 + 2)))
            return p1, p2
        vunit = vnorm / nrm
        p1 = (x0 - vunit[0] * seg_half, y0 - vunit[1] * seg_half)
        p2 = (x0 + vunit[0] * seg_half, y0 + vunit[1] * seg_half)
        p1 = (max(0, min(w, p1[0])), max(0, min(h, p1[1])))
        p2 = (max(0, min(w, p2[0])), max(0, min(h, p2[1])))
        return p1, p2


# -------- Enhanced Visualization Function --------
def enhanced_visualization(img_rgb, mask_idx, names_all, centers_all, axes_all, cs_x, cs_y,
                           s_vals, k, speed, main_bend_regions, peaks_info,
                           metrics, out_path):
    """
    Generate enhanced visualization:
    Left: original image with mask overlay, vertebral centers, PCA axes, bend regions, and key metrics.
    Right: curvature vs arc length plot highlighting bend regions.
    """
    H, W = mask_idx.shape
    rows, cols = np.where((mask_idx > 0) & (mask_idx <= 11))
    if len(rows) == 0:
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        core_y0 = int(rows.min())
        core_y1 = int(rows.max())
        core_x0 = int(cols.min())
        core_x1 = int(cols.max())
        y0 = max(core_y0 - ROI_PAD_Y, 0)
        y1 = min(core_y1 + ROI_PAD_Y, H)
        x0 = max(core_x0 - ROI_PAD_X, 0)
        x1 = min(core_x1 + ROI_PAD_X, W)

    img_crop = img_rgb[y0:y1, x0:x1].copy()
    mask_crop = mask_idx[y0:y1, x0:x1]
    h_crop, w_crop = img_crop.shape[:2]

    color_mask = np.zeros_like(img_crop, dtype=np.uint8)
    for i in range(1, len(COLOR_MAP)):
        color_mask[mask_crop == i] = COLOR_MAP[i]
    overlay = cv2.addWeighted(img_crop.astype(np.uint8), 0.85, color_mask, 0.35, 0)

    fig = plt.figure(figsize=(14, 10))
    ax_img = fig.add_subplot(1, 2, 1)
    ax_plot = fig.add_subplot(1, 2, 2)

    # High-resolution sampling for smooth curve and curvature colormap
    s_plot = np.linspace(0.0, 1.0, 2000)
    xs = cs_x(s_plot)
    ys = cs_y(s_plot)
    xs_rel = xs - x0
    ys_rel = ys - y0

    dxp = cs_x.derivative(1)(s_plot)
    dyp = cs_y.derivative(1)(s_plot)
    ddxp = cs_x.derivative(2)(s_plot)
    ddyp = cs_y.derivative(2)(s_plot)
    nump = np.abs(dxp * ddyp - dyp * ddxp)
    denp = (dxp * dxp + dyp * dyp) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        k_plot = np.nan_to_num(nump / denp)

    k_vis = k_plot
    if k_vis.max() > 0:
        k_norm = (k_vis - k_vis.min()) / (k_vis.max() - k_vis.min() + 1e-12)
    else:
        k_norm = k_vis

    ax_img.imshow(overlay, interpolation='nearest')
    sc = ax_img.scatter(xs_rel, ys_rel, c=k_norm, s=6, cmap='inferno', alpha=0.95, linewidths=0)

    # Mark vertebral centers and names
    centers_rel = [(c[0] - x0, c[1] - y0) for c in centers_all]
    for i, (name, (cx, cy)) in enumerate(zip(names_all, centers_rel)):
        ax_img.scatter(cx, cy, s=28, c='white', edgecolors='black', linewidth=0.7, zorder=5)
        ax_img.text(cx + 6, cy, name, color='cyan', fontsize=9, weight='bold', va='center')

    # Draw PCA main axes and endplate directions
    for i, (name, center, axis) in enumerate(zip(names_all, centers_all, axes_all)):
        if axis is None:
            continue
        center_rel = (center[0] - x0, center[1] - y0)
        scale_len_main = max(w_crop, h_crop) * 0.15
        main_axis_start = (center_rel[0] - axis[0] * scale_len_main, center_rel[1] - axis[1] * scale_len_main)
        main_axis_end = (center_rel[0] + axis[0] * scale_len_main, center_rel[1] + axis[1] * scale_len_main)

        endplate_axis = np.array([-axis[1], axis[0]])
        scale_len_endplate = max(w_crop, h_crop) * 0.12
        endplate_start = (center_rel[0] - endplate_axis[0] * scale_len_endplate,
                          center_rel[1] - endplate_axis[1] * scale_len_endplate)
        endplate_end = (center_rel[0] + endplate_axis[0] * scale_len_endplate,
                        center_rel[1] + endplate_axis[1] * scale_len_endplate)

        if i == 0:
            ax_img.plot([main_axis_start[0], main_axis_end[0]], [main_axis_start[1], main_axis_end[1]],
                        color='red', linewidth=1.0, alpha=0.7, label='PCA Main Axis')
            ax_img.plot([endplate_start[0], endplate_end[0]], [endplate_start[1], endplate_end[1]],
                        color='green', linewidth=1.0, alpha=0.7, label='PCA Endplate')
        else:
            ax_img.plot([main_axis_start[0], main_axis_end[0]], [main_axis_start[1], main_axis_end[1]],
                        color='red', linewidth=1.0, alpha=0.7)
            ax_img.plot([endplate_start[0], endplate_end[0]], [endplate_start[1], endplate_end[1]],
                        color='green', linewidth=1.0, alpha=0.7)

    # Show bend regions
    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break
        s_region = np.linspace(region['s_left'], region['s_right'], 100)
        x_region = cs_x(s_region) - x0
        y_region = cs_y(s_region) - y0
        ax_img.plot(x_region, y_region, '-',
                    color=BEND_REGION_COLORS[i], linewidth=2.5, alpha=0.7,
                    label=f'Bend Region {i + 1}')

    # Show curvature peak points
    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break
        peak_x = cs_x(region['peak_s']) - x0
        peak_y = cs_y(region['peak_s']) - y0
        ax_img.scatter([peak_x], [peak_y], s=80,
                       color=BEND_REGION_COLORS[i], edgecolors='white', linewidth=1.5,
                       zorder=10, marker='*')

    # Draw extended endplate lines for L1 and L5
    if 'L1' in names_all and 'L5' in names_all:
        idx_top = names_all.index('L1')
        idx_bot = names_all.index('L5')
        top_center = centers_all[idx_top]
        bot_center = centers_all[idx_bot]
        top_axis = axes_all[idx_top]
        bot_axis = axes_all[idx_bot]
        if top_axis is not None and bot_axis is not None:
            end_top = np.array([-top_axis[1], top_axis[0]], dtype=float)
            end_bot = np.array([-bot_axis[1], bot_axis[0]], dtype=float)

            top_center_rel = (top_center[0] - x0, top_center[1] - y0)
            bot_center_rel = (bot_center[0] - x0, bot_center[1] - y0)
            top_seg_p1, top_seg_p2 = line_segment_within_bbox(top_center_rel, end_top, w_crop, h_crop)
            bot_seg_p1, bot_seg_p2 = line_segment_within_bbox(bot_center_rel, end_bot, w_crop, h_crop)

            ax_img.plot([top_seg_p1[0], top_seg_p2[0]], [top_seg_p1[1], top_seg_p2[1]],
                        color='yellow', linewidth=1.8, label='L1 Endplate (extended)')
            ax_img.plot([bot_seg_p1[0], bot_seg_p2[0]], [bot_seg_p1[1], bot_seg_p2[1]],
                        color='orange', linewidth=1.8, label='L5 Endplate (extended)')

            ax_img.text(6, 18, f"Endplate angle (PCA) = {metrics['endplate_angle_deg']:.2f}°",
                        color='yellow', fontsize=9, bbox=dict(facecolor='black', alpha=0.6, pad=2))

    # Visualize tangents at boundaries of the primary bend region
    if main_bend_regions:
        primary_region = main_bend_regions[0]
        s_left = primary_region['s_left']
        s_right = primary_region['s_right']

        xl = cs_x(s_left)
        yl = cs_y(s_left)
        xr = cs_x(s_right)
        yr = cs_y(s_right)
        dxl = cs_x.derivative(1)(s_left)
        dyl = cs_y.derivative(1)(s_left)
        dxr = cs_x.derivative(1)(s_right)
        dyr = cs_y.derivative(1)(s_right)

        v_l_u = np.array([dxl, dyl])
        v_r_u = np.array([dxr, dyr])
        if np.linalg.norm(v_l_u) > 0:
            v_l_u = v_l_u / np.linalg.norm(v_l_u)
        if np.linalg.norm(v_r_u) > 0:
            v_r_u = v_r_u / np.linalg.norm(v_r_u)

        short_half = min(w_crop, h_crop) * 0.12
        pt_l = np.array([xl - x0, yl - y0], dtype=float)
        pt_r = np.array([xr - x0, yr - y0], dtype=float)
        p_l1 = (pt_l - v_l_u * short_half)
        p_l2 = (pt_l + v_l_u * short_half)
        p_r1 = (pt_r - v_r_u * short_half)
        p_r2 = (pt_r + v_r_u * short_half)

        def clamp_pt(p):
            return (float(np.clip(p[0], 0.0, float(w_crop))), float(np.clip(p[1], 0.0, float(h_crop))))

        seg_l_short_p1 = clamp_pt(p_l1)
        seg_l_short_p2 = clamp_pt(p_l2)
        seg_r_short_p1 = clamp_pt(p_r1)
        seg_r_short_p2 = clamp_pt(p_r2)

        ax_img.plot([seg_l_short_p1[0], seg_l_short_p2[0]], [seg_l_short_p1[1], seg_l_short_p2[1]],
                    color='magenta', linewidth=1.2, label='curve tangent left (short)')
        ax_img.plot([seg_r_short_p1[0], seg_r_short_p2[0]], [seg_r_short_p1[1], seg_r_short_p2[1]],
                    color='lime', linewidth=1.2, label='curve tangent right (short)')

        ax_img.scatter([cs_x(s_left) - x0], [cs_y(s_left) - y0], s=32, c='magenta', edgecolors='black', zorder=6)
        ax_img.scatter([cs_x(s_right) - x0], [cs_y(s_right) - y0], s=32, c='lime', edgecolors='black', zorder=6)

    # Display metrics on image
    txt_y = h_crop - 84
    ax_img.text(6, txt_y, f"Arc length L = {metrics['L_arc']:.1f} px", color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6))
    ax_img.text(6, txt_y + 16, f"Max curvature k_max = {metrics['k_max']:.5f} px^-1", color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6))
    ax_img.text(6, txt_y + 32, f"Curvature integral I = {metrics['k_integral']:.3f}", color='white', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6))
    if metrics['R'] is None or np.isnan(metrics['R']):
        ax_img.text(6, txt_y + 48, f"Bending radius R = inf", color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.6))
    else:
        ax_img.text(6, txt_y + 48, f"Bending radius R = {metrics['R']:.1f} px", color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.6))

    txt_y_offset = 64
    for i, region in enumerate(main_bend_regions[:2]):
        if i >= 2:
            break
        region_text = f"Region {i + 1}: k_max={region['peak_curvature']:.5f}"
        ax_img.text(6, txt_y + txt_y_offset, region_text,
                    color=BEND_REGION_COLORS[i], fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.6))
        txt_y_offset += 14

    ax_img.axis('off')
    ax_img.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)

    # ---- Right plot: curvature vs arc length ----
    if main_bend_regions:
        for i, region in enumerate(main_bend_regions):
            if i >= len(BEND_REGION_COLORS):
                break
            li = np.argmin(np.abs(s_vals - region['s_left']))
            ri = np.argmin(np.abs(s_vals - region['s_right']))
            if ri <= li:
                ri = min(li + 1, len(s_vals) - 1)

            l_region = np.zeros_like(s_vals[li:ri + 1])
            if len(l_region) > 1:
                l_region[1:] = np.cumsum(
                    0.5 * (speed[li:ri] + speed[li + 1:ri + 1]) * (s_vals[li + 1:ri + 1] - s_vals[li:ri]))

            k_region = k[li:ri + 1]
            ax_plot.plot(l_region, k_region, '-', color=BEND_REGION_COLORS[i], linewidth=1.5,
                         label=f'Region {i + 1}')
            ax_plot.fill_between(l_region, 0, k_region, color=BEND_REGION_COLORS[i], alpha=0.18)

            peak_idx_in_region = np.argmin(np.abs(s_vals[li:ri + 1] - region['peak_s']))
            if peak_idx_in_region < len(l_region):
                ax_plot.scatter([l_region[peak_idx_in_region]], [k_region[peak_idx_in_region]],
                                color=BEND_REGION_COLORS[i], s=40, zorder=5)

    ax_plot.set_xlabel('arc length (px)')
    ax_plot.set_ylabel('curvature k (px^-1)')
    ax_plot.set_title('Curvature vs arc length (all bend regions)')
    ax_plot.grid(True, linewidth=0.5, alpha=0.6)
    ax_plot.legend()

    R_value = f"{metrics['R']:.1f}" if (metrics['R'] is not None and not np.isnan(metrics['R'])) else "inf"
    ax_plot.text(0.02, 0.95,
                 f"Primary Region:\nL={metrics['L_arc']:.1f}px\nk_max={metrics['k_max']:.5f}\nI={metrics['k_integral']:.3f}\nR={R_value}",
                 transform=ax_plot.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    cbar = fig.colorbar(sc, ax=[ax_img, ax_plot], fraction=0.045, pad=0.02)
    cbar.set_label('relative curvature')

    plt.tight_layout()
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)


# ---------------- Improved Processing Pipeline ----------------
def improved_process_pair(img_path, mask_path):
    """
    Full processing pipeline for one image-mask pair:
    1. Load and preprocess data.
    2. For each vertebra, segment and analyze using improved PCA.
    3. Fit spinal spline.
    4. Perform improved curvature analysis to identify bend regions.
    5. Compute key metrics.
    6. Generate enhanced visualization.
    7. Return results.
    """
    img = read_image(img_path)
    mask_idx = read_mask_index(mask_path)

    name_to_idx = {name: i for i, name in enumerate(NAME_CLASSES)}
    results = {}
    for vname in ["L1", "L2", "L3", "L4", "L5", "S1"]:
        if vname not in name_to_idx:
            continue
        idx = name_to_idx[vname]
        m = (mask_idx == idx)

        if m.sum() < MIN_VERTEBRA_PIXELS:
            for nm in NAME_CLASSES:
                if '/' in nm and vname in nm:
                    m = m | (mask_idx == name_to_idx[nm])

        if m.sum() < MIN_VERTEBRA_PIXELS:
            continue

        mc = largest_component(m)
        coords = np.column_stack(np.nonzero(mc))
        coords_xy = np.column_stack((coords[:, 1].astype(float), coords[:, 0].astype(float)))

        centroid, axis = improved_pca_axis_from_coords(coords_xy)
        if centroid is None:
            continue

        results[vname] = {'centroid': centroid, 'axis': axis, 'mask': mc}

    present = [v for v in ["L1", "L2", "L3", "L4", "L5"] if v in results]
    if len(present) < 3:
        return None, {"reason": "not_enough_L1-L5", "detected": list(results.keys())}

    ordered = sorted([(v, results[v]['centroid']) for v in present], key=lambda kv: kv[1][1])
    names = [k for k, _ in ordered]
    centers = np.array([c for _, c in ordered])
    axes = np.array([results[n]['axis'] for n in names])

    cs_x, cs_y, t = fit_spine_spline(centers)
    if cs_x is None:
        return None, {"reason": "spline_failed"}

    s_vals, k_smooth, speed, main_bend_regions, peaks_info = improved_curvature_analysis(cs_x, cs_y)

    region_metrics = []
    for region in main_bend_regions:
        li = np.argmin(np.abs(s_vals - region['s_left']))
        ri = np.argmin(np.abs(s_vals - region['s_right']))

        if ri <= li:
            ri = min(li + 1, len(s_vals) - 1)

        s_region = s_vals[li:ri + 1]
        speed_region = speed[li:ri + 1]
        k_region = k_smooth[li:ri + 1]

        L_arc = float(np.trapz(speed_region, s_region))
        k_max = float(np.max(k_region))
        k_integral = float(np.trapz(np.abs(k_region) * speed_region, s_region))
        R = 1.0 / k_max if k_max > 1e-12 else None

        region_metrics.append({
            'L_arc': L_arc,
            'k_max': k_max,
            'k_integral': k_integral,
            'R': R
        })

    main_metrics = region_metrics[0] if region_metrics else {
        'L_arc': 0,
        'k_max': 0,
        'k_integral': 0,
        'R': None
    }

    endplate_angle = calculate_endplate_angle_improved(results)

    metrics = {
        'endplate_angle_deg': float(endplate_angle) if endplate_angle is not None else np.nan,
        'L_arc': main_metrics['L_arc'],
        'k_max': main_metrics['k_max'],
        'k_integral': main_metrics['k_integral'],
        'R': main_metrics['R'] if main_metrics['R'] is not None else np.nan,
        'num_bend_regions': len(main_bend_regions),
        'primary_peak_curvature': main_bend_regions[0]['peak_curvature'] if main_bend_regions else 0
    }

    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_vis = os.path.join(VIS_DIR, f"{basename}_improved_pca_curv.png")

    enhanced_visualization(img, mask_idx, names, centers, axes, cs_x, cs_y,
                           s_vals, k_smooth, speed, main_bend_regions, peaks_info,
                           metrics, out_vis)

    return {
               "filename": basename,
               "endplate_angle_deg": (float(endplate_angle) if endplate_angle is not None else ""),
               "L_arc_px": main_metrics['L_arc'],
               "k_max_px_inv": main_metrics['k_max'],
               "k_integral": main_metrics['k_integral'],
               "R_px": (main_metrics['R'] if main_metrics['R'] is not None else ""),
               "num_bend_regions": len(main_bend_regions),
               "primary_peak_curvature": (main_bend_regions[0]['peak_curvature'] if main_bend_regions else 0),
               "s_left": float(main_bend_regions[0]['s_left']) if main_bend_regions else "",
               "s_right": float(main_bend_regions[0]['s_right']) if main_bend_regions else ""
           }, None


# ---------------- Batch Processing ----------------
def improved_batch_process(img_dir, mask_dir, out_dir):
    """
    Batch process all image-mask pairs in the given directories.
    Save results to CSV and generate a statistical report.
    """
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir)
    os.makedirs(vis_dir, exist_ok=True)

    all_results = []
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))

    print(f"Processing {len(img_paths)} images...")

    for i, img_path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{name}.png")

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask not found for {name}, skipping")
            all_results.append({"filename": name, "note": "mask missing"})
            continue

        print(f"Processing {i + 1}/{len(img_paths)}: {name}")

        res, err = improved_process_pair(img_path, mask_path)

        if err is not None:
            print(f"[WARN] {name} processing failed: {err}")
            all_results.append({"filename": name, "note": str(err)})
            continue

        print(f"[INFO] {name}: endplate angle={res['endplate_angle_deg']}, "
              f"arc length={res['L_arc_px']:.1f}, max curvature={res['k_max_px_inv']:.5f}, "
              f"bend regions={res['num_bend_regions']}")

        all_results.append(res)

    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, 'improved_results_pca_curv.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[Done] Results saved to {csv_path}")

    generate_statistical_report(df, out_dir)

    return df


def generate_statistical_report(df, out_dir):
    """Generate a simple statistical report from the results DataFrame."""
    report_path = os.path.join(out_dir, 'statistical_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Spinal Curvature Analysis Statistical Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total samples: {len(df)}\n")

        valid_angles = df['endplate_angle_deg'].dropna()
        if len(valid_angles) > 0:
            f.write(f"\nEndplate angle statistics (n={len(valid_angles)}):\n")
            f.write(f"  Mean: {valid_angles.mean():.2f}°\n")
            f.write(f"  Std: {valid_angles.std():.2f}°\n")
            f.write(f"  Range: {valid_angles.min():.2f}° - {valid_angles.max():.2f}°\n")

        valid_k_max = df['k_max_px_inv'].dropna()
        if len(valid_k_max) > 0:
            f.write(f"\nMax curvature statistics (n={len(valid_k_max)}):\n")
            f.write(f"  Mean: {valid_k_max.mean():.6f} px^-1\n")
            f.write(f"  Std: {valid_k_max.std():.6f} px^-1\n")
            f.write(f"  Range: {valid_k_max.min():.6f} - {valid_k_max.max():.6f} px^-1\n")

        valid_regions = df['num_bend_regions'].dropna()
        if len(valid_regions) > 0:
            f.write(f"\nBend region distribution:\n")
            for i in range(1, int(valid_regions.max()) + 1):
                count = (valid_regions == i).sum()
                percentage = (count / len(valid_regions)) * 100
                f.write(f"  {i} region(s): {count} samples ({percentage:.1f}%)\n")

        successful = len(df) - df['note'].notna().sum()
        success_rate = (successful / len(df)) * 100
        f.write(f"\nProcessing success rate: {successful}/{len(df)} ({success_rate:.1f}%)\n")

    print(f"Statistical report saved to: {report_path}")


# ---------------- Paper-specific: Extract cubic spline curve alone ----------------
def extract_cubic_spline_curve(
    img_path, mask_path,
    save_dir=None,
    curve_color=(255, 0, 0),
    curve_width=2.0,
    mark_vertebra=True,
    vertebra_mark_color=(255, 255, 255),
    vertebra_text_color=(0, 0, 0),
    background_type="transparent",
    crop_roi=True,
    roi_pad_y=40, roi_pad_x=60,
    save_dpi=600,
    save_format="png"
):
    """
    Extract and visualize only the cubic spline curve from a single image-mask pair,
    producing a clean figure suitable for academic papers.
    """
    if save_dir is None:
        save_dir = os.path.join(OUT_DIR, "spline_curve_paper")
    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f"{basename}_spine_spline.{save_format}")

    img = read_image(img_path)
    mask_idx = read_mask_index(mask_path)
    name_to_idx = {name: i for i, name in enumerate(NAME_CLASSES)}
    results = {}

    for vname in ["L1", "L2", "L3", "L4", "L5", "S1"]:
        if vname not in name_to_idx:
            continue
        idx = name_to_idx[vname]
        m = (mask_idx == idx)
        if m.sum() < MIN_VERTEBRA_PIXELS:
            for nm in NAME_CLASSES:
                if '/' in nm and vname in nm:
                    m = m | (mask_idx == name_to_idx[nm])
        if m.sum() < MIN_VERTEBRA_PIXELS:
            continue
        mc = largest_component(m)
        coords = np.column_stack(np.nonzero(mc))
        coords_xy = np.column_stack((coords[:, 1].astype(float), coords[:, 0].astype(float)))
        centroid, axis = improved_pca_axis_from_coords(coords_xy)
        if centroid is None:
            continue
        results[vname] = {'centroid': centroid, 'axis': axis}

    present = [v for v in ["L1", "L2", "L3", "L4", "L5"] if v in results]
    if len(present) < 3:
        print(f"[WARN] {basename} insufficient vertebrae, cannot fit spline")
        return None

    ordered = sorted([(v, results[v]['centroid']) for v in present], key=lambda kv: kv[1][1])
    names = [k for k, _ in ordered]
    centers = np.array([c for _, c in ordered])

    cs_x, cs_y, t = fit_spine_spline(centers)
    if cs_x is None:
        print(f"[WARN] {basename} spline fitting failed")
        return None

    s_plot = np.linspace(0.0, 1.0, 3000)
    xs = cs_x(s_plot)
    ys = cs_y(s_plot)

    H, W = mask_idx.shape
    if crop_roi:
        rows, cols = np.where((mask_idx > 0) & (mask_idx <= 11))
        if len(rows) > 0:
            core_y0, core_y1 = int(rows.min()), int(rows.max())
            core_x0, core_x1 = int(cols.min()), int(cols.max())
            y0 = max(core_y0 - roi_pad_y, 0)
            y1 = min(core_y1 + roi_pad_y, H)
            x0 = max(core_x0 - roi_pad_x, 0)
            x1 = min(core_x1 + roi_pad_x, W)
        else:
            y0, y1, x0, x1 = 0, H, 0, W
        xs_roi = xs - x0
        ys_roi = ys - y0
        centers_roi = np.array([(c[0]-x0, c[1]-y0) for c in centers])
        roi_h, roi_w = y1-y0, x1-x0
    else:
        xs_roi, ys_roi = xs, ys
        centers_roi = centers
        roi_h, roi_w = H, W

    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(roi_w/100, roi_h/100), dpi=100)
    ax.set_xlim(0, roi_w)
    ax.set_ylim(roi_h, 0)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if background_type == "original":
        img_roi = img[y0:y1, x0:x1] if crop_roi else img
        ax.imshow(img_roi, interpolation='nearest')
    elif background_type == "white":
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    elif background_type == "transparent":
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')

    # Compute curvature for colormap (same as in original code)
    dxp = cs_x.derivative(1)(s_plot)
    dyp = cs_y.derivative(1)(s_plot)
    ddxp = cs_x.derivative(2)(s_plot)
    ddyp = cs_y.derivative(2)(s_plot)
    nump = np.abs(dxp * ddyp - dyp * ddxp)
    denp = (dxp * dxp + dyp * dyp) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        k_plot = np.nan_to_num(nump / denp)
    k_norm = (k_plot - k_plot.min()) / (k_plot.max() - k_plot.min() + 1e-12) if k_plot.max() > 0 else k_plot

    ax.scatter(
        xs_roi, ys_roi,
        c=k_norm,
        s=1.5,
        cmap='inferno',
        alpha=0.95,
        linewidths=0,
        marker='.'
    )

    if mark_vertebra:
        for name, (cx, cy) in zip(names, centers_roi):
            ax.scatter(cx, cy, s=30, c=np.array(vertebra_mark_color)/255,
                       edgecolors='black', linewidth=0.8, zorder=5)
            ax.text(cx + 5, cy, name, color=np.array(vertebra_text_color)/255,
                    fontsize=8, weight='bold', va='center', ha='left')

    plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.0, format=save_format)
    plt.close(fig)
    print(f"[SUCCESS] Cubic spline curve saved: {save_path}")
    return save_path


def batch_extract_spline_curve(img_dir, mask_dir):
    """Batch extract cubic spline curves for all images."""
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
    print(f"Batch extracting spline curves for {len(img_paths)} images...")
    for i, img_path in enumerate(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{basename}.png")
        if not os.path.exists(mask_path):
            print(f"[WARN] {basename} mask missing, skipping")
            continue
        print(f"Processing {i+1}/{len(img_paths)}: {basename}")
        extract_cubic_spline_curve(img_path, mask_path)
    print("Batch extraction completed! Curves saved in:", os.path.join(OUT_DIR, "spline_curve_paper"))


# ---------------- Main Entry ----------------
if __name__ == "__main__":
    # Single image extraction example (uncomment and adjust paths):
    IMG_PATH = "img/xx.jpg"
    MASK_PATH = "mask/xx.png"
    extract_cubic_spline_curve(IMG_PATH, MASK_PATH)

    # Batch processing (uncomment to run full analysis):
    # improved_batch_process(IMG_DIR, MASK_DIR, OUT_DIR)

    pass
