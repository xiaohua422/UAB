# 导入必要的库（原有代码不变，此处省略，直接接修改后的可视化函数）
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

# ------------ 用户配置 ------------（原有配置不变，直接保留）
IMG_DIR = "img"
MASK_DIR = "mask"
OUT_DIR = os.path.join('.', 'every_amace_cobb_results')
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
ANGLE_TEXT_COLOR = 'yellow'
# -------------------------------------

plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 10

# ------------------------------------------------------
# 工具函数（原有工具函数完全不变，直接保留）
# ------------------------------------------------------
def read_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot find or read image file: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_mask_index(path):
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
    labels = measure.label(mask_bool, connectivity=1)
    if labels.max() == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    props = measure.regionprops(labels)
    lab = props[np.argmax([p.area for p in props])].label
    return labels == lab

def find_edge_points(coords_xy):
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
    if coords_xy.shape[0] < min_points:
        return None, None
    centroid = coords_xy.mean(axis=0)
    distances = cdist([centroid], coords_xy)[0]
    valid_mask = distances < np.percentile(distances, 90)
    filtered_coords = coords_xy[valid_mask]
    if filtered_coords.shape[0] < min_points:
        filtered_coords = coords_xy
    edge_points = find_edge_points(filtered_coords)
    weights = np.ones(len(filtered_coords))
    if len(edge_points) > 0:
        for i, point in enumerate(filtered_coords):
            if any(np.all(point == edge_point) for edge_point in edge_points):
                weights[i] = PCA_WEIGHT_EDGE_FACTOR
    centroid = np.average(filtered_coords, axis=0, weights=weights)
    centered = filtered_coords - centroid
    weighted_cov = np.cov(centered.T, aweights=weights)
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    main_axis = eigvecs[:, np.argmax(eigvals)]
    return centroid, main_axis

def angle_between_vectors_deg(v1, v2):
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
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def calculate_endplate_angle_improved(results):
    if 'L1' not in results or 'L5' not in results:
        return None
    top_axis = results['L1']['axis']
    bot_axis = results['L5']['axis']
    if top_axis is None or bot_axis is None:
        return None
    end_top = normalize_vector(np.array([-top_axis[1], top_axis[0]]))
    end_bot = normalize_vector(np.array([-bot_axis[1], bot_axis[0]]))
    if end_top[1] > 0:
        end_top = -end_top
    if end_bot[1] > 0:
        end_bot = -end_bot
    angle = angle_between_vectors_deg(end_top, end_bot)
    if angle > 90.0:
        angle = 180.0 - angle
    return angle

def calculate_intervertebral_angles(results):
    intervertebral_angles = {}
    max_angle = 0
    max_segment = None
    vertebra_pairs = [
        ("L1", "L2"), ("L2", "L3"), ("L3", "L4"),
        ("L4", "L5")
    ]
    for upper, lower in vertebra_pairs:
        if upper in results and lower in results:
            upper_axis = results[upper]['axis']
            lower_axis = results[lower]['axis']
            if upper_axis is not None and lower_axis is not None:
                end_upper = normalize_vector(np.array([-upper_axis[1], upper_axis[0]]))
                end_lower = normalize_vector(np.array([-lower_axis[1], lower_axis[0]]))
                if end_upper[1] > 0:
                    end_upper = -end_upper
                if end_lower[1] > 0:
                    end_lower = -end_lower
                angle = angle_between_vectors_deg(end_upper, end_lower)
                if angle > 90.0:
                    angle = 180.0 - angle
                segment_name = f"{upper}-{lower}"
                intervertebral_angles[segment_name] = angle
                if angle > max_angle:
                    max_angle = angle
                    max_segment = segment_name
    return {
        'angles': intervertebral_angles,
        'max_angle': max_angle,
        'max_segment': max_segment,
        'num_segments': len(intervertebral_angles)
    }

def fit_spine_spline(centers):
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
    try:
        return savgol_filter(k, window_size, 2)
    except:
        return np.convolve(k, np.ones(window_size) / window_size, mode='same')

def detect_multiple_peaks(k, s_vals, min_prominence=None):
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
    s_vals, dx, dy, ddx, ddy, k, speed = compute_curvature(cs_x, cs_y, num_samples)
    k_smooth = smooth_curvature(k)
    peaks_info = detect_multiple_peaks(k_smooth, s_vals)
    main_bend_regions = identify_main_bend_regions(peaks_info, k_smooth, s_vals)
    return s_vals, k_smooth, speed, main_bend_regions, peaks_info

def find_nearest_s_for_point(cs_x, cs_y, point, s_vals=np.linspace(0, 1, 1000)):
    ptsx = cs_x(s_vals)
    ptsy = cs_y(s_vals)
    d2 = (ptsx - point[0]) ** 2 + (ptsy - point[1]) ** 2
    return s_vals[np.argmin(d2)]

def line_segment_within_bbox(pt_rel, v, w, h, eps=1e-6):
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

# ------------------------------------------------------
# 【修改1/2】可视化函数：vertebrae_angles - 左侧单独显示椎骨间角度
# ------------------------------------------------------
# ------------------------------------------------------
# 【修改后】可视化函数：vertebrae_angles - 椎骨间角度移到Angle与Max Bend之间
# ------------------------------------------------------
def visualize_vertebrae_angles(img_rgb, mask_idx, names_all, centers_all, axes_all, cs_x, cs_y,
                               s_vals, k, speed, main_bend_regions, peaks_info,
                               metrics, intervertebral_angles, out_path):
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

    fig = plt.figure(figsize=(10, 12))
    ax_img = fig.add_subplot(1, 1, 1)

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
    sc_img = ax_img.scatter(xs_rel, ys_rel, c=k_norm, s=6, cmap='inferno', alpha=0.95, linewidths=0, zorder=2)

    centers_rel = [(c[0] - x0, c[1] - y0) for c in centers_all]
    for i, (name, (cx, cy)) in enumerate(zip(names_all, centers_rel)):
        ax_img.scatter(cx, cy, s=28, c='white', edgecolors='black', linewidth=0.7, zorder=5)
        ax_img.text(cx + 6, cy, name, color='cyan', fontsize=9, weight='bold', va='center')

    for i, (name, center, axis) in enumerate(zip(names_all, centers_all, axes_all)):
        if axis is None:
            continue
        center_rel = (center[0] - x0, center[1] - y0)
        scale_len_main = max(w_crop, h_crop) * 0.15
        main_axis_start = (center_rel[0] - axis[0] * scale_len_main, center_rel[1] - axis[1] * scale_len_main)
        main_axis_end = (center_rel[0] + axis[0] * scale_len_main, center_rel[1] + axis[1] * scale_len_main)
        endplate_axis = np.array([-axis[1], axis[0]])
        scale_len_endplate = max(w_crop, h_crop) * 0.12
        endplate_start = (
            center_rel[0] - endplate_axis[0] * scale_len_endplate,
            center_rel[1] - endplate_axis[1] * scale_len_endplate)
        endplate_end = (
            center_rel[0] + endplate_axis[0] * scale_len_endplate,
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

    # 绘制弯曲区域和峰值点（原有逻辑不变）
    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break
        s_region = np.linspace(region['s_left'], region['s_right'], 100)
        x_region = cs_x(s_region) - x0
        y_region = cs_y(s_region) - y0
        ax_img.plot(x_region, y_region, '-',
                    color=BEND_REGION_COLORS[i], linewidth=2.5, alpha=0.7,
                    label=f'Bend Region {i + 1}')

    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break
        peak_x = cs_x(region['peak_s']) - x0
        peak_y = cs_y(region['peak_s']) - y0
        ax_img.scatter([peak_x], [peak_y], s=80,
                       color=BEND_REGION_COLORS[i], edgecolors='white', linewidth=1.5,
                       zorder=10, marker='*')

    # 绘制L1/L5端板线（原有逻辑不变）
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

    # ------------------- 核心重构：左上角标注（Angle → 椎骨间角度 → Max Bend → 弯曲区域） -------------------
    info_text_parts = []
    # 1. 第一行：Cobb Angle（原Angle）
    if 'endplate_angle_deg' in metrics:
        info_text_parts.append(f"Cobb Angle: {metrics['endplate_angle_deg']:.1f}°")
    # 2. 中间行：椎骨间角度（插入Angle和Max Bend之间，按L1-L2→L4-L5顺序）
    if intervertebral_angles and 'angles' in intervertebral_angles:
        angles_dict = intervertebral_angles['angles']
        segments_order = ["L1-L2", "L2-L3", "L3-L4", "L4-L5"]  # 固定顺序
        for seg in segments_order:
            if seg in angles_dict:
                info_text_parts.append(f"{seg}: {angles_dict[seg]:.1f}°")
    # 3. 后续行：Max Bend + 弯曲区域（原有逻辑）
    if intervertebral_angles and 'max_segment' in intervertebral_angles and intervertebral_angles['max_segment']:
        max_segment = intervertebral_angles['max_segment']
        max_angle = intervertebral_angles['max_angle']
        info_text_parts.append(f"Max Bend: {max_segment} = {max_angle:.1f}°")
    if main_bend_regions:
        info_text_parts.append(f"Bend Regions: {len(main_bend_regions)}")
    # 拼接所有标注，换行显示
    info_text = "\n".join(info_text_parts) if info_text_parts else ""
    # 绘制标注（样式与原有完全一致，黑底半透明+白字）
    if info_text:
        ax_img.text(0.02, 0.99, info_text,
                    transform=ax_img.transAxes, fontsize=10, weight='bold',
                    color='white', va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, pad=0.5))
    # ------------------------------------------------------------------------------------------------------

    ax_img.axis('off')
    # 【修复图例警告】仅当有标签时才显示图例
    handles, labels = ax_img.get_legend_handles_labels()
    if handles:
        ax_img.legend(handles=handles, labels=labels, loc='upper right', fontsize=9)

    plt.tight_layout()
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    # 【加异常处理】避免单文件保存失败中断批量处理
    try:
        plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01, format='pdf')
    except Exception as e:
        print(f"[WARN] 保存vertebrae PDF失败: {pdf_path}, 错误: {str(e)[:50]}")
    try:
        plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01)
    except Exception as e:
        print(f"[WARN] 保存vertebrae PNG失败: {out_path}, 错误: {str(e)[:50]}")
    plt.close(fig)
    return pdf_path

# ------------------------------------------------------
# 【修改2/2】可视化函数：spine_curve - 移除Adjacent Angles标题
# ------------------------------------------------------
# ------------------------------------------------------
# 【修改后】可视化函数：spine_curve - 彻底移除Adjacent Angles所有文字标注
# ------------------------------------------------------
def visualize_spine_curve(img_rgb, mask_idx, names_all, centers_all, axes_all, cs_x, cs_y,
                          s_vals, k, speed, main_bend_regions, peaks_info,
                          metrics, intervertebral_angles, out_path):
    fig = plt.figure(figsize=(10, 12))
    ax_curve = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0.3, right=0.95, bottom=0.1, top=0.9)

    s_plot = np.linspace(0.0, 1.0, 2000)
    xs = cs_x(s_plot)
    ys = cs_y(s_plot)

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

    curve_min_x, curve_max_x = xs.min(), xs.max()
    curve_min_y, curve_max_y = ys.min(), ys.max()
    curve_width = curve_max_x - curve_min_x
    curve_height = curve_max_y - curve_min_y

    sc_curve = ax_curve.scatter(xs, -ys, c=k_norm, s=10, cmap='viridis', alpha=1.0, linewidths=0)

    for i, (name, center) in enumerate(zip(names_all, centers_all)):
        center_x, center_y = center
        ax_curve.scatter(center_x, -center_y, s=40, c='white', edgecolors='black', linewidth=1.0, zorder=5)
        ax_curve.text(center_x + 0.02 * curve_width, -center_y, name,
                      color='cyan', fontsize=8, weight='bold', va='center')

    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break
        s_region = np.linspace(region['s_left'], region['s_right'], 100)
        x_region = cs_x(s_region)
        y_region = cs_y(s_region)
        ax_curve.plot(x_region, -y_region, '-',
                      color=BEND_REGION_COLORS[i], linewidth=2.0, alpha=0.8,
                      label=f'Region {i + 1}')
        peak_x = cs_x(region['peak_s'])
        peak_y = cs_y(region['peak_s'])
        ax_curve.scatter([peak_x], [-peak_y], s=100,
                         color=BEND_REGION_COLORS[i], edgecolors='white', linewidth=2.0,
                         zorder=10, marker='*')

    ax_curve.set_title('Spine Fitting Curve', fontsize=12, weight='bold')
    ax_curve.set_xlabel('X (pixels)', fontsize=10)
    ax_curve.set_ylabel('Y (pixels, inverted)', fontsize=10)
    ax_curve.grid(True, linestyle='--', alpha=0.5)
    # 【修复图例警告】仅当有标签时才显示图例
    handles, labels = ax_curve.get_legend_handles_labels()
    if handles:
        ax_curve.legend(handles=handles, labels=labels, fontsize=9, loc='upper right')

    margin_x = curve_width * 0.1
    margin_y = curve_height * 0.1
    ax_curve.set_xlim(curve_min_x - margin_x, curve_max_x + margin_x)
    ax_curve.set_ylim(-curve_max_y - margin_y, -curve_min_y + margin_y)

    cbar = fig.colorbar(sc_curve, ax=ax_curve, fraction=0.046, pad=0.04)
    cbar.set_label('Curvature (normalized)', fontsize=10)

    # ------------------- 核心修改：彻底删除所有Adjacent Angles相关文字标注 -------------------
    # 【已完全删除】原所有椎骨间角度的文字拼接、绘制代码，无任何残留
    # --------------------------------------------------------------------------------------

    plt.tight_layout()
    pdf_path = os.path.splitext(out_path)[0] + '.pdf'
    # 【加异常处理】避免单文件保存失败中断批量处理
    try:
        plt.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01, format='pdf')
    except Exception as e:
        print(f"[WARN] 保存spine PDF失败: {pdf_path}, 错误: {str(e)[:50]}")
    try:
        plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01)
    except Exception as e:
        print(f"[WARN] 保存spine PNG失败: {out_path}, 错误: {str(e)[:50]}")
    plt.close(fig)
    return pdf_path

# ---------------- 改进的处理流程（原有逻辑完全不变，直接保留） ----------------
def improved_process_pair(img_path, mask_path):
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
    intervertebral_angles = calculate_intervertebral_angles(results)
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
    vertebrae_dir = os.path.join(VIS_DIR, 'vertebrae_angles')
    spine_curves_dir = os.path.join(VIS_DIR, 'spine_curves')
    os.makedirs(vertebrae_dir, exist_ok=True)
    os.makedirs(spine_curves_dir, exist_ok=True)
    vertebrae_out_path = os.path.join(vertebrae_dir, f"{basename}_vertebrae_angles.png")
    vertebrae_pdf_path = visualize_vertebrae_angles(img, mask_idx, names, centers, axes, cs_x, cs_y,
                                                    s_vals, k_smooth, speed, main_bend_regions, peaks_info,
                                                    metrics, intervertebral_angles, vertebrae_out_path)
    spine_out_path = os.path.join(spine_curves_dir, f"{basename}_spine_curve.png")
    spine_pdf_path = visualize_spine_curve(img, mask_idx, names, centers, axes, cs_x, cs_y,
                                           s_vals, k_smooth, speed, main_bend_regions, peaks_info,
                                           metrics, intervertebral_angles, spine_out_path)
    intervertebral_angles_str = ""
    if intervertebral_angles and 'angles' in intervertebral_angles:
        angles_dict = intervertebral_angles['angles']
        angle_pairs = []
        for segment, angle in angles_dict.items():
            angle_pairs.append(f"{segment}:{angle:.1f}")
        intervertebral_angles_str = ";".join(angle_pairs)
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
               "s_right": float(main_bend_regions[0]['s_right']) if main_bend_regions else "",
               "intervertebral_angles": intervertebral_angles_str,
               "max_segment": intervertebral_angles.get('max_segment', ''),
               "max_segment_angle": intervertebral_angles.get('max_angle', 0),
               "num_segments": intervertebral_angles.get('num_segments', 0),
               "angle_L1_L2": intervertebral_angles.get('angles', {}).get('L1-L2', ''),
               "angle_L2_L3": intervertebral_angles.get('angles', {}).get('L2-L3', ''),
               "angle_L3_L4": intervertebral_angles.get('angles', {}).get('L3-L4', ''),
               "angle_L4_L5": intervertebral_angles.get('angles', {}).get('L4-L5', ''),
               "vertebrae_image_png": vertebrae_out_path,
               "vertebrae_image_pdf": vertebrae_pdf_path,
               "spine_curve_image_png": spine_out_path,
               "spine_curve_image_pdf": spine_pdf_path,
           }, None

# ---------------- 批量处理（原有逻辑完全不变，直接保留） ----------------
def improved_batch_process(img_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir)
    os.makedirs(vis_dir, exist_ok=True)
    all_results = []
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
    print(f"Start processing {len(img_paths)} images...")
    for i, img_path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{name}.png")
        if not os.path.exists(mask_path):
            print(f"[WARN] Mask file for {name} does not exist, skipping")
            all_results.append({"filename": name, "note": "mask missing"})
            continue
        print(f"Processing {i + 1}/{len(img_paths)}: {name}")
        res, err = improved_process_pair(img_path, mask_path)
        if err is not None:
            print(f"[WARN] {name} processing failed: {err}")
            all_results.append({"filename": name, "note": str(err)})
            continue
        print(f"[INFO] {name}: Endplate angle={res['endplate_angle_deg']}, "
              f"Max bend segment={res['max_segment']}({res['max_segment_angle']:.1f}°), "
              f"Bend regions={res['num_bend_regions']}")
        all_results.append(res)
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, 'improved_results_pca_curv.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[DONE] Results saved to {csv_path}")
    generate_statistical_report(df, out_dir)
    return df

def generate_statistical_report(df, out_dir):
    report_path = os.path.join(out_dir, 'statistical_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Spine Curvature Analysis Statistical Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(df)}\n")
        valid_angles = df['endplate_angle_deg'].dropna()
        if len(valid_angles) > 0:
            f.write(f"\nEndplate Angle Statistics (n={len(valid_angles)}):\n")
            f.write(f"  Mean: {valid_angles.mean():.2f}°\n")
            f.write(f"  Std: {valid_angles.std():.2f}°\n")
            f.write(f"  Range: {valid_angles.min():.2f}° - {valid_angles.max():.2f}°\n")
        valid_max_segment_angles = df['max_segment_angle'].dropna()
        if len(valid_max_segment_angles) > 0:
            f.write(f"\nMax Segment Angle Statistics (n={len(valid_max_segment_angles)}):\n")
            f.write(f"  Mean: {valid_max_segment_angles.mean():.2f}°\n")
            f.write(f"  Std: {valid_max_segment_angles.std():.2f}°\n")
            f.write(f"  Range: {valid_max_segment_angles.min():.2f}° - {valid_max_segment_angles.max():.2f}°\n")
        valid_max_segments = df['max_segment'].dropna()
        if len(valid_max_segments) > 0:
            f.write(f"\nMax Segment Distribution:\n")
            segment_counts = valid_max_segments.value_counts()
            for segment, count in segment_counts.items():
                percentage = (count / len(valid_max_segments)) * 100
                f.write(f"  {segment}: {count} samples ({percentage:.1f}%)\n")
        segment_pairs = ['L1-L2', 'L2-L3', 'L3-L4', 'L4-L5']
        for segment in segment_pairs:
            col_name = f"angle_{segment.replace('-', '_')}"
            if col_name in df.columns:
                valid_angles = df[col_name].dropna()
                if len(valid_angles) > 0:
                    f.write(f"\n{segment} Angle Statistics (n={len(valid_angles)}):\n")
                    f.write(f"  Mean: {valid_angles.mean():.2f}°\n")
                    f.write(f"  Std: {valid_angles.std():.2f}°\n")
                    f.write(f"  Range: {valid_angles.min():.2f}° - {valid_angles.max():.2f}°\n")
        valid_k_max = df['k_max_px_inv'].dropna()
        if len(valid_k_max) > 0:
            f.write(f"\nMax Curvature Statistics (n={len(valid_k_max)}):\n")
            f.write(f"  Mean: {valid_k_max.mean():.6f} px^-1\n")
            f.write(f"  Std: {valid_k_max.std():.6f} px^-1\n")
            f.write(f"  Range: {valid_k_max.min():.6f} - {valid_k_max.max():.6f} px^-1\n")
        valid_regions = df['num_bend_regions'].dropna()
        if len(valid_regions) > 0:
            f.write(f"\nBend Region Statistics:\n")
            for i in range(1, int(valid_regions.max()) + 1):
                count = (valid_regions == i).sum()
                percentage = (count / len(valid_regions)) * 100
                f.write(f"  {i} regions: {count} samples ({percentage:.1f}%)\n")
        successful = len(df) - df['note'].notna().sum()
        success_rate = (successful / len(df)) * 100
        f.write(f"\nProcessing Success Rate: {successful}/{len(df)} ({success_rate:.1f}%)\n")
    print(f"Statistical report saved to: {report_path}")

# ---------------- 主程序入口（原有逻辑完全不变，直接保留） ----------------
if __name__ == "__main__":
    improved_batch_process(IMG_DIR, MASK_DIR, OUT_DIR)
