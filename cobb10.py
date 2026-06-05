#cobb7改进的amace
# 1）基于椎体 mask 做 PCA 并以其垂线近似端板方向 和 （2）在主弯曲区域计算局部曲率指标（最大曲率、曲率积分、弯曲半径、弧长）并可视化
# 核心功能：
# 1) 基于椎体 mask，使用改进的PCA方法（加权PCA和边缘点检测）来更准确地
#    确定椎体的主轴，并以此近似端板方向，进而计算Cobb角。
# 2) 在拟合的脊柱曲线上，计算并可视化局部曲率指标，包括：
#    - 最大曲率 (k_max)
#    - 曲率积分 (I)
#    - 弯曲半径 (R)
#    - 弧长 (L_arc)
# 3) 通过多峰值检测和区域识别，自动识别并可视化多个脊柱弯曲区域。
#
# 主要改进点：
# 1. 改进PCA方法，使用加权PCA和边缘点检测，提高端板方向估计的准确性。
# 2. 改进端板角度计算，确保方向一致性，使角度计算更稳定。
# 3. 曲率分析改进：增加曲率平滑步骤、多峰值检测、多弯曲区域识别。
# 4. 可视化增强：在图像上直接显示多个弯曲区域、峰值点和关键指标。
# 5. 配置参数优化，将关键参数集中管理，便于调整。
# ------------------------------------------------------

# 导入必要的库
import os
import math
from glob import glob  # 用于查找匹配的文件路径
import numpy as np  # 用于数值计算
import cv2  # 用于图像处理
from scipy.interpolate import CubicSpline  # 用于三次样条插值
from scipy.signal import find_peaks, savgol_filter  # 用于峰值检测和平滑
from scipy.spatial import ConvexHull  # 用于计算凸包，找到边缘点
from scipy.spatial.distance import cdist  # 用于计算点之间的距离
from skimage import measure  # 用于图像测量，如连通区域分析
import matplotlib.pyplot as plt  # 用于绘图和可视化
import pandas as pd  # 用于数据处理和保存结果到CSV

# ------------ 用户配置 ------------
# 定义输入图像和mask的目录
IMG_DIR = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\img"
MASK_DIR = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\mask"

# 定义输出目录
OUT_DIR = os.path.join('.', 'cobb10_amace_cobb_results')
VIS_DIR = os.path.join(OUT_DIR)  # 可视化结果保存目录
os.makedirs(VIS_DIR, exist_ok=True)  # 创建目录，如果已存在则忽略

# 定义mask中各类别的名称
NAME_CLASSES = ["_background_", "L1", "L2", "L3", "L4", "L5",
                "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]
# 定义我们主要关注的目标椎体
TARGET_VERTEBRAE = ["L1", "L2", "L3", "L4", "L5", "S1"]

# 裁剪ROI时的padding大小，用于在椎体周围留出一定空间
ROI_PAD_Y = 60
ROI_PAD_X = 80
# 一个椎体mask至少需要的像素数，用于过滤小的、无效的分割结果
MIN_VERTEBRA_PIXELS = 50  # 提高最小像素要求，以排除噪声
# 计算曲率时，在样条曲线上采样的点数
CURV_SAMPLES = 1200
# 旧版峰值检测的相对高度阈值（在改进的检测中不再直接使用）
CURV_PEAK_REL_H = 0.12
# 旧版确定弯曲区域边界的阈值比例（在改进的检测中不再直接使用）
CURV_BOUND_FRAC = 0.2
# 旧版参数（可能用于兼容或未使用）
CURV_OFFSET = 40
CURV_SCALE = 200.0
# 保存图像的DPI（分辨率）
SAVE_DPI = 300

# 改进的配置参数
CURV_SMOOTH_WINDOW = 7  # 用于Savitzky-Golay滤波器的窗口大小，用于平滑曲率曲线
PEAK_PROMINENCE_RATIO = 0.15  # 峰值显著性( prominence)的比例，用于多峰值检测
MAX_BEND_REGIONS = 3  # 最多识别的弯曲区域数量
PCA_WEIGHT_EDGE_FACTOR = 2.5  # 在加权PCA中，边缘点的权重因子

# 定义mask可视化的颜色映射表
COLOR_MAP = [
    (0, 0, 0), (200, 80, 20), (30, 160, 40), (200, 200, 40), (30, 30, 160), (200, 30, 160),
    (30, 160, 160), (160, 160, 160), (120, 40, 40), (240, 40, 40), (240, 180, 40), (120, 180, 40)
]

# 定义不同弯曲区域的显示颜色
BEND_REGION_COLORS = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
# -------------------------------------

# 设置matplotlib的绘图默认样式，使其更简洁、线条更细
plt.rcParams['lines.linewidth'] = 1.0
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['font.size'] = 10


# ------------------------------------------------------
# 工具函数
# ------------------------------------------------------

def read_image(path):
    """
    读取图像文件，并从BGR格式转换为RGB格式。
    OpenCV默认读取为BGR，而matplotlib显示需要RGB。
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"无法找到或读取图像文件: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask_index(path):
    """
    读取mask图像，并将其转换为整数索引格式。
    如果mask是3通道且所有通道相同，则取单通道；否则转换为灰度图。
    """
    m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"无法找到或读取mask文件: {path}")
    # 处理多通道mask的情况
    if m.ndim == 3:
        if np.all(m[..., 0] == m[..., 1]) and np.all(m[..., 1] == m[..., 2]):
            m = m[..., 0]  # 如果三通道相同，取第一通道
        else:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)  # 否则转为灰度
    return m.astype(np.int32)


def largest_component(mask_bool):
    """
    从一个布尔型mask中提取出面积最大的连通组件。
    这有助于去除分割结果中的小噪声区域。
    """
    # 对mask进行连通区域标记
    labels = measure.label(mask_bool, connectivity=1)
    if labels.max() == 0:  # 如果没有找到任何连通区域
        return np.zeros_like(mask_bool, dtype=bool)

    # 获取每个连通区域的属性
    props = measure.regionprops(labels)
    # 找到面积最大的连通区域的标签
    lab = props[np.argmax([p.area for p in props])].label
    return labels == lab


def find_edge_points(coords_xy):
    """
    从一组点中找到边缘点。
    优先使用凸包(ConvexHull)算法，如果失败（点数太少或共线），
    则退而求其次，返回边界框上的点。
    """
    if len(coords_xy) < 10:  # 如果点数太少，无法构成有意义的边缘
        return []

    try:
        # 使用凸包算法找到边缘点
        hull = ConvexHull(coords_xy)
        return coords_xy[hull.vertices]
    except:  # 捕获所有可能的异常，如QhullError（点数不足或共线）
        # 如果凸包失败，使用简单的边界框方法
        x_min, x_max = coords_xy[:, 0].min(), coords_xy[:, 0].max()
        y_min, y_max = coords_xy[:, 1].min(), coords_xy[:, 1].max()

        # 筛选出位于边界框边缘上的点
        edge_mask = (
                (coords_xy[:, 0] == x_min) | (coords_xy[:, 0] == x_max) |
                (coords_xy[:, 1] == y_min) | (coords_xy[:, 1] == y_max)
        )
        return coords_xy[edge_mask]


def improved_pca_axis_from_coords(coords_xy, min_points=10):
    """
    改进的PCA方法，用于从椎体的坐标点中估计其主轴。
    改进之处：
    1. 去除异常点，使结果更稳健。
    2. 对边缘点施加更高的权重，因为它们更能代表椎体的形状。
    """
    if coords_xy.shape[0] < min_points:  # 如果点数不足，返回None
        return None, None

    # 1. 去除异常点 (改进点)
    centroid = coords_xy.mean(axis=0)
    distances = cdist([centroid], coords_xy)[0]
    # 保留距离质心较近的90%的点，去除最远的10%
    valid_mask = distances < np.percentile(distances, 90)
    filtered_coords = coords_xy[valid_mask]

    if filtered_coords.shape[0] < min_points:  # 如果过滤后点数太少，则使用原始点
        filtered_coords = coords_xy

    # 2. 加权PCA - 给边缘点更高权重 (改进点)
    edge_points = find_edge_points(filtered_coords)
    weights = np.ones(len(filtered_coords))  # 初始化权重为1
    if len(edge_points) > 0:
        # 找到边缘点在filtered_coords中的索引，并增加其权重
        for i, point in enumerate(filtered_coords):
            if any(np.all(point == edge_point) for edge_point in edge_points):
                weights[i] = PCA_WEIGHT_EDGE_FACTOR

    # 3. 执行加权PCA
    # 计算加权质心
    centroid = np.average(filtered_coords, axis=0, weights=weights)
    # 中心化坐标
    centered = filtered_coords - centroid
    # 计算加权协方差矩阵
    weighted_cov = np.cov(centered.T, aweights=weights)

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(weighted_cov)
    # 对应最大特征值的特征向量就是主轴方向
    main_axis = eigvecs[:, np.argmax(eigvals)]

    return centroid, main_axis


def angle_between_vectors_deg(v1, v2):
    """计算两个向量之间的夹角（以度为单位）。"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:  # 避免除以零
        return 0.0
    # 使用点积公式计算夹角余弦值，并限制在[-1, 1]范围内以防数值溢出
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    # 转换为角度
    ang = math.degrees(math.acos(cosang))
    return ang


def normalize_vector(v):
    """归一化向量，使其成为单位向量。"""
    norm = np.linalg.norm(v)
    if norm == 0:  # 避免除以零
        return v
    return v / norm


def calculate_endplate_angle_improved(results):
    """
    改进的端板角度计算方法。
    确保端板方向（垂直于椎体主轴）的一致性，并计算锐角。
    """
    if 'L1' not in results or 'L5' not in results:  # 必须同时有L1和L5的结果
        return None

    top_axis = results['L1']['axis']
    bot_axis = results['L5']['axis']

    if top_axis is None or bot_axis is None:  # 如果主轴计算失败
        return None

    # 确保端板方向的一致性 (改进点)
    # 端板方向近似为垂直于主轴的方向（旋转90度）
    end_top = normalize_vector(np.array([-top_axis[1], top_axis[0]]))
    end_bot = normalize_vector(np.array([-bot_axis[1], bot_axis[0]]))

    # 强制端板方向朝上（即y轴负方向），使角度计算更稳定
    if end_top[1] > 0:
        end_top = -end_top
    if end_bot[1] > 0:
        end_bot = -end_bot

    # 计算两向量的夹角
    angle = angle_between_vectors_deg(end_top, end_bot)

    # 确保返回的是锐角
    if angle > 90.0:
        angle = 180.0 - angle

    return angle


def fit_spine_spline(centers):
    """
    根据一系列椎体中心点拟合一条平滑的三次样条曲线。
    返回的是x和y方向的样条函数，以及用于参数化的归一化弧长。
    """
    pts = np.array(centers)
    if pts.shape[0] < 2:  # 至少需要2个点来拟合曲线
        return None, None, None

    # 计算点与点之间的欧几里得距离
    diffs = np.diff(pts, axis=0)
    seglen = np.sqrt((diffs ** 2).sum(axis=1))
    # 计算累计弧长，作为样条的参数t
    t = np.concatenate(([0.0], np.cumsum(seglen)))

    # 归一化t到[0, 1]范围
    if t[-1] == 0:  # 所有点重合的特殊情况
        t = np.linspace(0, 1, len(pts))
    else:
        t = t / t[-1]

    # 创建x和y关于t的三次样条
    cs_x = CubicSpline(t, pts[:, 0])
    cs_y = CubicSpline(t, pts[:, 1])
    return cs_x, cs_y, t


def compute_curvature(cs_x, cs_y, num_samples=CURV_SAMPLES):
    """
    在拟合的样条曲线上计算曲率。
    曲率公式: k = |x'*y'' - y'*x''| / (x'^2 + y'^2)^(3/2)
    """
    # 在[0, 1]范围内均匀采样
    s_vals = np.linspace(0.0, 1.0, num_samples)

    # 计算一阶和二阶导数
    dx = cs_x.derivative(1)(s_vals)
    dy = cs_y.derivative(1)(s_vals)
    ddx = cs_x.derivative(2)(s_vals)
    ddy = cs_y.derivative(2)(s_vals)

    # 计算曲率的分子和分母
    num = np.abs(dx * ddy - dy * ddx)
    den = (dx * dx + dy * dy) ** 1.5

    # 计算曲率，并用np.errstate忽略除零警告，用np.nan_to_num处理NaN和Inf
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.nan_to_num(num / den)

    # 计算曲线的"速度"，即ds/dt (s是弧长参数)
    speed = np.sqrt(dx * dx + dy * dy)

    return s_vals, dx, dy, ddx, ddy, k, speed


def smooth_curvature(k, window_size=CURV_SMOOTH_WINDOW):
    """
    使用Savitzky-Golay滤波器平滑曲率曲线，以减少噪声影响。
    如果Savitzky-Golay失败（例如窗口大小不合适），则使用简单的移动平均作为备选。
    """
    try:
        # Savitzky-Golay滤波器能在平滑的同时保留信号的趋势
        return savgol_filter(k, window_size, 2)
    except:
        # 移动平均作为备选方案
        return np.convolve(k, np.ones(window_size) / window_size, mode='same')


def detect_multiple_peaks(k, s_vals, min_prominence=None):
    """
    在平滑后的曲率曲线上检测多个峰值。
    使用`prominence`（峰值显著性）作为检测标准，这比简单的高度阈值更稳健。
    """
    if min_prominence is None:
        # 如果未指定，则根据曲率最大值的一定比例来设置
        min_prominence = PEAK_PROMINENCE_RATIO * np.max(k) if np.max(k) > 0 else 0.1

    # 使用scipy的find_peaks函数检测峰值
    peaks, properties = find_peaks(k, prominence=min_prominence)

    # 整理峰值信息，包括其位置、对应的曲率值和显著性
    peaks_info = []
    for i, peak_idx in enumerate(peaks):
        peaks_info.append({
            'index': peak_idx,
            's_value': s_vals[peak_idx],
            'curvature': k[peak_idx],
            'prominence': properties['prominences'][i] if 'prominences' in properties else 0
        })

    # 按曲率值从大到小排序
    peaks_info.sort(key=lambda x: x['curvature'], reverse=True)
    return peaks_info


def detect_main_bend(k, s_vals, rel_peak_h=CURV_PEAK_REL_H, bound_frac=CURV_BOUND_FRAC, peak_idx=None):
    """
    给定一个峰值点，确定其对应的弯曲区域的左右边界。
    边界由曲率下降到峰值曲率的一定比例（bound_frac）来确定。
    """
    if peak_idx is None:  # 如果未指定峰值索引，则默认为全局最大值点
        peak_idx = int(np.argmax(k))

    maxk = np.max(k)
    if maxk <= 0:  # 如果曲率都为零或负值
        return s_vals[max(peak_idx - 1, 0)], s_vals[min(peak_idx + 1, len(s_vals) - 1)], peak_idx

    # 计算边界阈值
    peak_k = k[peak_idx]
    thr = max(peak_k * bound_frac, 1e-9)  # 确保阈值为正

    # 从峰值点向左搜索边界
    left = peak_idx
    while left > 0 and k[left] > thr:
        left -= 1

    # 从峰值点向右搜索边界
    right = peak_idx
    while right < len(k) - 1 and k[right] > thr:
        right += 1

    # 稍微扩展边界以确保包含完整的弯曲
    left = max(left - 1, 0)
    right = min(right + 1, len(k) - 1)

    return s_vals[left], s_vals[right], peak_idx


def identify_main_bend_regions(peaks_info, k, s_vals, num_regions=MAX_BEND_REGIONS):
    """
    根据检测到的峰值，识别并返回主要的弯曲区域。
    每个区域由其左右边界、峰值位置和峰值曲率定义。
    """
    regions = []

    # 遍历排序后的峰值，最多识别num_regions个区域
    for i, peak in enumerate(peaks_info[:num_regions]):
        peak_idx = peak['index']
        # 为每个峰值确定其弯曲区域
        s_left, s_right, _ = detect_main_bend(k, s_vals, peak_idx=peak_idx)

        regions.append({
            'peak_s': s_vals[peak_idx],  # 峰值在样条参数s上的位置
            's_left': s_left,  # 区域左边界
            's_right': s_right,  # 区域右边界
            'peak_curvature': k[peak_idx],  # 峰值处的曲率
            'region_index': i  # 区域索引
        })

    return regions


def improved_curvature_analysis(cs_x, cs_y, num_samples=CURV_SAMPLES):
    """
    整合所有改进的曲率分析步骤：
    1. 计算原始曲率。
    2. 平滑曲率曲线。
    3. 检测多个曲率峰值。
    4. 识别每个峰值对应的弯曲区域。
    """
    # 1. 计算曲率
    s_vals, dx, dy, ddx, ddy, k, speed = compute_curvature(cs_x, cs_y, num_samples)

    # 2. 平滑曲率信号 (改进点)
    k_smooth = smooth_curvature(k)

    # 3. 多峰值检测 (改进点)
    peaks_info = detect_multiple_peaks(k_smooth, s_vals)

    # 4. 主弯曲区域识别 (改进点)
    main_bend_regions = identify_main_bend_regions(peaks_info, k_smooth, s_vals)

    return s_vals, k_smooth, speed, main_bend_regions, peaks_info


def find_nearest_s_for_point(cs_x, cs_y, point, s_vals=np.linspace(0, 1, 1000)):
    """
    找到样条曲线上离给定点`point`最近的点所对应的参数`s`值。
    """
    ptsx = cs_x(s_vals)
    ptsy = cs_y(s_vals)
    # 计算给定点到曲线上所有采样点的距离平方
    d2 = (ptsx - point[0]) ** 2 + (ptsy - point[1]) ** 2
    # 返回距离最近的点的s值
    return s_vals[np.argmin(d2)]


def line_segment_within_bbox(pt_rel, v, w, h, eps=1e-6):
    """
    计算一条从点`pt_rel`出发、沿向量`v`方向的直线，与给定边界框（宽w，高h）的交点。
    返回边界框内的线段两个端点。这用于将端板线延伸至图像边缘。
    """
    x0, y0 = float(pt_rel[0]), float(pt_rel[1])
    vx, vy = float(v[0]), float(v[1])
    candidates = []

    # 计算与左右边界 (x=0, x=w) 的交点
    if abs(vx) > eps:
        t1 = (0 - x0) / vx
        y1 = y0 + t1 * vy
        if -eps <= y1 <= h + eps:
            candidates.append((x0 + t1 * vx, y1, t1))
        t2 = (w - x0) / vx
        y2 = y0 + t2 * vy
        if -eps <= y2 <= h + eps:
            candidates.append((x0 + t2 * vx, y2, t2))

    # 计算与上下边界 (y=0, y=h) 的交点
    if abs(vy) > eps:
        t3 = (0 - y0) / vy
        x3 = x0 + t3 * vx
        if -eps <= x3 <= w + eps:
            candidates.append((x3, y0 + t3 * vy, t3))
        t4 = (h - y0) / vy
        x4 = x0 + t4 * vx
        if -eps <= x4 <= w + eps:
            candidates.append((x4, y0 + t4 * vy, t4))

    # 如果找到了两个或更多交点，取参数t最小和最大的两个点作为线段端点
    if len(candidates) >= 2:
        candidates_sorted = sorted(candidates, key=lambda it: it[2])
        p1 = (candidates_sorted[0][0], candidates_sorted[0][1])
        p2 = (candidates_sorted[-1][0], candidates_sorted[-1][1])
        return p1, p2
    else:
        # 如果没有找到交点（线段完全在边界框外或方向向量为零）
        # 则创建一个以pt_rel为中心的短小线段
        seg_half = min(w, h) * 0.25
        vnorm = np.array([vx, vy], dtype=float)
        nrm = np.linalg.norm(vnorm)
        if nrm < eps:  # 方向向量为零
            p1 = (max(0, min(w, x0 - 2)), max(0, min(h, y0 - 2)))
            p2 = (max(0, min(w, x0 + 2)), max(0, min(h, y0 + 2)))
            return p1, p2
        vunit = vnorm / nrm
        p1 = (x0 - vunit[0] * seg_half, y0 - vunit[1] * seg_half)
        p2 = (x0 + vunit[0] * seg_half, y0 + vunit[1] * seg_half)
        # 将线段端点限制在边界框内
        p1 = (max(0, min(w, p1[0])), max(0, min(h, p1[1])))
        p2 = (max(0, min(w, p2[0])), max(0, min(h, p2[1])))
        return p1, p2


# -------- 改进的可视化函数 --------
def enhanced_visualization(img_rgb, mask_idx, names_all, centers_all, axes_all, cs_x, cs_y,
                           s_vals, k, speed, main_bend_regions, peaks_info,
                           metrics, out_path):
    """
    生成增强的可视化结果。
    左侧显示原始图像叠加mask、椎体中心、PCA轴、弯曲区域和关键指标。
    右侧显示曲率曲线图，突出显示主要弯曲区域。
    """

    H, W = mask_idx.shape
    # 找到mask中非零区域的边界，用于裁剪ROI
    rows, cols = np.where((mask_idx > 0) & (mask_idx <= 11))
    if len(rows) == 0:  # 如果mask为空
        y0, y1, x0, x1 = 0, H, 0, W
    else:
        core_y0 = int(rows.min())
        core_y1 = int(rows.max())
        core_x0 = int(cols.min())
        core_x1 = int(cols.max())
        # 加上padding
        y0 = max(core_y0 - ROI_PAD_Y, 0)
        y1 = min(core_y1 + ROI_PAD_Y, H)
        x0 = max(core_x0 - ROI_PAD_X, 0)
        x1 = min(core_x1 + ROI_PAD_X, W)

    # 裁剪图像和mask
    img_crop = img_rgb[y0:y1, x0:x1].copy()
    mask_crop = mask_idx[y0:y1, x0:x1]
    h_crop, w_crop = img_crop.shape[:2]

    # 创建彩色mask用于叠加显示
    color_mask = np.zeros_like(img_crop, dtype=np.uint8)
    for i in range(1, len(COLOR_MAP)):
        color_mask[mask_crop == i] = COLOR_MAP[i]
    # 将彩色mask与原图叠加
    overlay = cv2.addWeighted(img_crop.astype(np.uint8), 0.85, color_mask, 0.35, 0)

    # 设置matplotlib图形
    fig = plt.figure(figsize=(14, 10))
    ax_img = fig.add_subplot(1, 2, 1)  # 左图用于显示图像
    ax_plot = fig.add_subplot(1, 2, 2)  # 右图用于显示曲率曲线

    # 为了平滑显示样条曲线和曲率颜色映射，使用更高分辨率的采样
    s_plot = np.linspace(0.0, 1.0, 2000)
    xs = cs_x(s_plot)
    ys = cs_y(s_plot)
    # 将坐标转换为裁剪后的图像坐标系
    xs_rel = xs - x0
    ys_rel = ys - y0

    # 在高分辨率采样点上计算曲率，用于颜色映射
    dxp = cs_x.derivative(1)(s_plot)
    dyp = cs_y.derivative(1)(s_plot)
    ddxp = cs_x.derivative(2)(s_plot)
    ddyp = cs_y.derivative(2)(s_plot)
    nump = np.abs(dxp * ddyp - dyp * ddxp)
    denp = (dxp * dxp + dyp * dyp) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        k_plot = np.nan_to_num(nump / denp)

    # 归一化曲率用于颜色映射
    k_vis = k_plot
    if k_vis.max() > 0:
        k_norm = (k_vis - k_vis.min()) / (k_vis.max() - k_vis.min() + 1e-12)
    else:
        k_norm = k_vis

    # 显示叠加后的图像
    ax_img.imshow(overlay, interpolation='nearest')

    # 绘制样条曲线，并用颜色映射表示曲率大小
    sc = ax_img.scatter(xs_rel, ys_rel, c=k_norm, s=6, cmap='inferno', alpha=0.95, linewidths=0)

    # 标记每个椎体的中心点和名称
    centers_rel = [(c[0] - x0, c[1] - y0) for c in centers_all]
    for i, (name, (cx, cy)) in enumerate(zip(names_all, centers_rel)):
        ax_img.scatter(cx, cy, s=28, c='white', edgecolors='black', linewidth=0.7, zorder=5)
        ax_img.text(cx + 6, cy, name, color='cyan', fontsize=9, weight='bold', va='center')

    # 绘制所有椎体的PCA主轴和端板方向
    for i, (name, center, axis) in enumerate(zip(names_all, centers_all, axes_all)):
        if axis is None:
            continue

        center_rel = (center[0] - x0, center[1] - y0)

        # 绘制PCA主轴（长轴）- 红色
        scale_len_main = max(w_crop, h_crop) * 0.15  # 根据图像大小自适应缩放
        main_axis_start = (center_rel[0] - axis[0] * scale_len_main, center_rel[1] - axis[1] * scale_len_main)
        main_axis_end = (center_rel[0] + axis[0] * scale_len_main, center_rel[1] + axis[1] * scale_len_main)

        # 绘制端板方向（短轴，垂直于主轴）- 绿色
        endplate_axis = np.array([-axis[1], axis[0]])  # 旋转90度
        scale_len_endplate = max(w_crop, h_crop) * 0.12
        endplate_start = (
            center_rel[0] - endplate_axis[0] * scale_len_endplate,
            center_rel[1] - endplate_axis[1] * scale_len_endplate)
        endplate_end = (
            center_rel[0] + endplate_axis[0] * scale_len_endplate,
            center_rel[1] + endplate_axis[1] * scale_len_endplate)

        # 只为第一个椎体添加图例，避免重复
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

    # 新增：显示多个弯曲区域
    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break

        # 在弯曲区域的边界之间采样点
        s_region = np.linspace(region['s_left'], region['s_right'], 100)
        x_region = cs_x(s_region) - x0
        y_region = cs_y(s_region) - y0

        # 绘制弯曲区域
        ax_img.plot(x_region, y_region, '-',
                    color=BEND_REGION_COLORS[i], linewidth=2.5, alpha=0.7,
                    label=f'Bend Region {i + 1}')

    # 新增：显示曲率峰值点
    for i, region in enumerate(main_bend_regions):
        if i >= len(BEND_REGION_COLORS):
            break

        # 计算峰值点在裁剪图像中的坐标
        peak_x = cs_x(region['peak_s']) - x0
        peak_y = cs_y(region['peak_s']) - y0

        # 用星形标记峰值点
        ax_img.scatter([peak_x], [peak_y], s=80,
                       color=BEND_REGION_COLORS[i], edgecolors='white', linewidth=1.5,
                       zorder=10, marker='*')

    # 绘制L1和L5的端板线（延伸至图像边界）
    if 'L1' in names_all and 'L5' in names_all:
        idx_top = names_all.index('L1')
        idx_bot = names_all.index('L5')
        top_center = centers_all[idx_top]
        bot_center = centers_all[idx_bot]
        top_axis = axes_all[idx_top]
        bot_axis = axes_all[idx_bot]
        if top_axis is not None and bot_axis is not None:
            # 计算端板方向
            end_top = np.array([-top_axis[1], top_axis[0]], dtype=float)
            end_bot = np.array([-bot_axis[1], bot_axis[0]], dtype=float)

            # 将端板线延伸至裁剪后图像边界
            top_center_rel = (top_center[0] - x0, top_center[1] - y0)
            bot_center_rel = (bot_center[0] - x0, bot_center[1] - y0)
            top_seg_p1, top_seg_p2 = line_segment_within_bbox(top_center_rel, end_top, w_crop, h_crop)
            bot_seg_p1, bot_seg_p2 = line_segment_within_bbox(bot_center_rel, end_bot, w_crop, h_crop)

            # 绘制延伸后的端板线
            ax_img.plot([top_seg_p1[0], top_seg_p2[0]], [top_seg_p1[1], top_seg_p2[1]],
                        color='yellow', linewidth=1.8, label='L1 Endplate (extended)')
            ax_img.plot([bot_seg_p1[0], bot_seg_p2[0]], [bot_seg_p1[1], bot_seg_p2[1]],
                        color='orange', linewidth=1.8, label='L5 Endplate (extended)')

            # 标注端板角度
            ax_img.text(6, 18, f"Endplate angle (PCA) = {metrics['endplate_angle_deg']:.2f}°",
                        color='yellow', fontsize=9, bbox=dict(facecolor='black', alpha=0.6, pad=2))

    # 可视化主要弯曲区域的左右切线
    if main_bend_regions:
        primary_region = main_bend_regions[0]
        s_left = primary_region['s_left']
        s_right = primary_region['s_right']

        # 计算切线方向（样条曲线的一阶导数）
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
            v_l_u = v_l_u / np.linalg.norm(v_l_u)  # 归一化
        if np.linalg.norm(v_r_u) > 0:
            v_r_u = v_r_u / np.linalg.norm(v_r_u)

        # 绘制短小的切线段
        short_half = min(w_crop, h_crop) * 0.12
        pt_l = np.array([xl - x0, yl - y0], dtype=float)
        pt_r = np.array([xr - x0, yr - y0], dtype=float)
        p_l1 = (pt_l - v_l_u * short_half)
        p_l2 = (pt_l + v_l_u * short_half)
        p_r1 = (pt_r - v_r_u * short_half)
        p_r2 = (pt_r + v_r_u * short_half)

        # 确保线段在图像内
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

        # 在切点处绘制标记
        ax_img.scatter([cs_x(s_left) - x0], [cs_y(s_left) - y0], s=32, c='magenta', edgecolors='black', zorder=6)
        ax_img.scatter([cs_x(s_right) - x0], [cs_y(s_right) - y0], s=32, c='lime', edgecolors='black', zorder=6)

    # 在图像上标注主要曲率指标
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

    # 新增：在图上显示多个区域的曲率指标
    txt_y_offset = 64
    for i, region in enumerate(main_bend_regions[:2]):  # 只显示前两个区域
        if i >= 2:
            break

        region_text = f"Region {i + 1}: k_max={region['peak_curvature']:.5f}"
        ax_img.text(6, txt_y + txt_y_offset, region_text,
                    color=BEND_REGION_COLORS[i], fontsize=8,
                    bbox=dict(facecolor='black', alpha=0.6))
        txt_y_offset += 14

    # 隐藏坐标轴
    ax_img.axis('off')
    # 将图例放在图像外部，避免遮挡
    ax_img.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9)

    # ---- 右图：曲率 vs 弧长 ----
    if main_bend_regions:
        # 为每个弯曲区域绘制曲率曲线
        for i, region in enumerate(main_bend_regions):
            if i >= len(BEND_REGION_COLORS):
                break

            # 找到该区域在s_vals数组中的索引范围
            li = np.argmin(np.abs(s_vals - region['s_left']))
            ri = np.argmin(np.abs(s_vals - region['s_right']))
            if ri <= li:
                ri = min(li + 1, len(s_vals) - 1)

            # 计算该区域内的弧长
            l_region = l_vec = np.zeros_like(s_vals[li:ri + 1])
            if len(l_region) > 1:
                # 使用梯形积分计算累积弧长
                l_region[1:] = np.cumsum(
                    0.5 * (speed[li:ri] + speed[li + 1:ri + 1]) * (s_vals[li + 1:ri + 1] - s_vals[li:ri]))

            k_region = k[li:ri + 1]

            # 绘制曲率曲线并填充下方区域
            ax_plot.plot(l_region, k_region, '-', color=BEND_REGION_COLORS[i], linewidth=1.5,
                         label=f'Region {i + 1}')
            ax_plot.fill_between(l_region, 0, k_region, color=BEND_REGION_COLORS[i], alpha=0.18)

            # 标记峰值点
            peak_idx_in_region = np.argmin(np.abs(s_vals[li:ri + 1] - region['peak_s']))
            if peak_idx_in_region < len(l_region):
                ax_plot.scatter([l_region[peak_idx_in_region]], [k_region[peak_idx_in_region]],
                                color=BEND_REGION_COLORS[i], s=40, zorder=5)

    # 设置图表标题和坐标轴标签
    ax_plot.set_xlabel('arc length (px)')
    ax_plot.set_ylabel('curvature k (px^-1)')
    ax_plot.set_title('Curvature vs arc length (all bend regions)')
    ax_plot.grid(True, linewidth=0.5, alpha=0.6)  # 显示网格
    ax_plot.legend()  # 显示图例

    # 在图表上标注主要区域的指标
    R_value = f"{metrics['R']:.1f}" if (metrics['R'] is not None and not np.isnan(metrics['R'])) else "inf"
    ax_plot.text(0.02, 0.95,
                 f"Primary Region:\nL={metrics['L_arc']:.1f}px\nk_max={metrics['k_max']:.5f}\nI={metrics['k_integral']:.3f}\nR={R_value}",
                 transform=ax_plot.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

    # 添加颜色条，解释曲率颜色映射
    cbar = fig.colorbar(sc, ax=[ax_img, ax_plot], fraction=0.045, pad=0.02)
    cbar.set_label('relative curvature')

    # 调整布局并保存图像
    plt.tight_layout()
    plt.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)  # 关闭图形，释放内存


# ---------------- 改进的处理流程 ----------------
def improved_process_pair(img_path, mask_path):
    """
    对单对图像和mask执行完整的处理流程：
    1. 读取和预处理数据。
    2. 对每个椎体进行分割和分析，使用改进的PCA计算主轴。
    3. 拟合脊柱样条曲线。
    4. 进行改进的曲率分析，识别多个弯曲区域。
    5. 计算关键指标。
    6. 生成增强的可视化结果。
    7. 返回计算结果。
    """
    # 1. 读取图像和mask
    img = read_image(img_path)
    mask_idx = read_mask_index(mask_path)

    # 2. 提取每个椎体的中心点和主轴 (使用改进的PCA)
    name_to_idx = {name: i for i, name in enumerate(NAME_CLASSES)}
    results = {}  # 存储每个椎体的分析结果
    for vname in ["L1", "L2", "L3", "L4", "L5", "S1"]:
        if vname not in name_to_idx:
            continue
        idx = name_to_idx[vname]
        # 创建该椎体的mask
        m = (mask_idx == idx)

        # 如果该椎体mask像素太少，尝试合并相邻椎间盘的mask（例如L1和L1/L2）
        if m.sum() < MIN_VERTEBRA_PIXELS:
            for nm in NAME_CLASSES:
                if '/' in nm and vname in nm:
                    m = m | (mask_idx == name_to_idx[nm])

        # 如果合并后像素仍然太少，则跳过该椎体
        if m.sum() < MIN_VERTEBRA_PIXELS:
            continue

        # 提取mask中最大的连通组件
        mc = largest_component(m)
        # 获取该组件的所有坐标点 (row, col)
        coords = np.column_stack(np.nonzero(mc))
        # 转换坐标为 (x, y) 格式 (x=col, y=row)
        coords_xy = np.column_stack((coords[:, 1].astype(float), coords[:, 0].astype(float)))

        # 使用改进的PCA计算质心和主轴
        centroid, axis = improved_pca_axis_from_coords(coords_xy)
        if centroid is None:  # 如果PCA失败，跳过
            continue

        results[vname] = {'centroid': centroid, 'axis': axis, 'mask': mc}

    # 检查是否至少检测到3个关键椎体 (L1-L5)，否则无法进行有效的曲线拟合
    present = [v for v in ["L1", "L2", "L3", "L4", "L5"] if v in results]
    if len(present) < 3:
        return None, {"reason": "not_enough_L1-L5", "detected": list(results.keys())}

    # 按颅尾方向（y坐标递增）排序椎体
    ordered = sorted([(v, results[v]['centroid']) for v in present], key=lambda kv: kv[1][1])
    names = [k for k, _ in ordered]
    centers = np.array([c for _, c in ordered])
    axes = np.array([results[n]['axis'] for n in names])

    # 3. 拟合脊柱样条曲线
    cs_x, cs_y, t = fit_spine_spline(centers)
    if cs_x is None:  # 如果样条拟合失败
        return None, {"reason": "spline_failed"}

    # 4. 使用改进的曲率分析方法
    s_vals, k_smooth, speed, main_bend_regions, peaks_info = improved_curvature_analysis(cs_x, cs_y)

    # 5. 计算每个弯曲区域的曲率指标
    region_metrics = []
    for region in main_bend_regions:
        # 找到该区域在s_vals数组中的索引范围
        li = np.argmin(np.abs(s_vals - region['s_left']))
        ri = np.argmin(np.abs(s_vals - region['s_right']))

        if ri <= li:  # 防止索引错误
            ri = min(li + 1, len(s_vals) - 1)

        # 提取该区域内的参数、速度和曲率
        s_region = s_vals[li:ri + 1]
        speed_region = speed[li:ri + 1]
        k_region = k_smooth[li:ri + 1]

        # 计算指标
        L_arc = float(np.trapz(speed_region, s_region))  # 弧长
        k_max = float(np.max(k_region))  # 最大曲率
        # 曲率积分（加权积分，权重为速度）
        k_integral = float(np.trapz(np.abs(k_region) * speed_region, s_region))
        # 弯曲半径 (R = 1/k_max)
        R = 1.0 / k_max if k_max > 1e-12 else None

        region_metrics.append({
            'L_arc': L_arc,
            'k_max': k_max,
            'k_integral': k_integral,
            'R': R
        })

    # 确定主要区域的指标（通常是曲率最大的区域）
    main_metrics = region_metrics[0] if region_metrics else {
        'L_arc': 0,
        'k_max': 0,
        'k_integral': 0,
        'R': None
    }

    # 使用改进的方法计算端板角度
    endplate_angle = calculate_endplate_angle_improved(results)

    # 整理所有计算出的指标
    metrics = {
        'endplate_angle_deg': float(endplate_angle) if endplate_angle is not None else np.nan,
        'L_arc': main_metrics['L_arc'],
        'k_max': main_metrics['k_max'],
        'k_integral': main_metrics['k_integral'],
        'R': main_metrics['R'] if main_metrics['R'] is not None else np.nan,
        'num_bend_regions': len(main_bend_regions),
        'primary_peak_curvature': main_bend_regions[0]['peak_curvature'] if main_bend_regions else 0
    }

    # 6. 生成增强的可视化结果
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_vis = os.path.join(VIS_DIR, f"{basename}_improved_pca_curv.png")

    enhanced_visualization(img, mask_idx, names, centers, axes, cs_x, cs_y,
                           s_vals, k_smooth, speed, main_bend_regions, peaks_info,
                           metrics, out_vis)

    # 7. 返回结果，用于写入CSV文件
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


# ---------------- 批量处理 ----------------
def improved_batch_process(img_dir, mask_dir, out_dir):
    """
    批量处理指定目录下的所有图像和mask。
    对每一对数据调用improved_process_pair，并将所有结果汇总保存到CSV文件。
    同时生成一个统计报告。
    """
    # 确保输出目录存在
    os.makedirs(out_dir, exist_ok=True)
    vis_dir = os.path.join(out_dir)
    os.makedirs(vis_dir, exist_ok=True)

    all_results = []  # 存储所有图像的处理结果
    # 获取所有.jpg图像文件的路径
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))

    print(f"开始处理 {len(img_paths)} 个图像...")

    # 遍历所有图像
    for i, img_path in enumerate(img_paths):
        name = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{name}.png")  # 假设mask文件为.png格式且同名

        # 检查mask文件是否存在
        if not os.path.exists(mask_path):
            print(f"[WARN] {name} 的mask文件不存在，跳过")
            all_results.append({"filename": name, "note": "mask missing"})
            continue

        print(f"处理 {i + 1}/{len(img_paths)}: {name}")

        # 处理单对图像和mask
        res, err = improved_process_pair(img_path, mask_path)

        if err is not None:  # 如果处理过程中出错
            print(f"[WARN] {name} 处理失败: {err}")
            all_results.append({"filename": name, "note": str(err)})
            continue

        # 打印成功处理的图像的关键结果
        print(f"[INFO] {name}: 端板角度={res['endplate_angle_deg']}, "
              f"弧长={res['L_arc_px']:.1f}, 最大曲率={res['k_max_px_inv']:.5f}, "
              f"弯曲区域数={res['num_bend_regions']}")

        all_results.append(res)

    # 将所有结果转换为DataFrame并保存到CSV文件
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, 'improved_results_pca_curv.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')  # 使用utf-8-sig编码以支持中文
    print(f"[完成] 结果已保存到 {csv_path}")

    # 生成并保存统计报告
    generate_statistical_report(df, out_dir)

    return df


def generate_statistical_report(df, out_dir):
    """
    根据处理结果的DataFrame生成一个简单的统计报告，并保存为文本文件。
    """
    report_path = os.path.join(out_dir, 'statistical_report.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("脊柱曲率分析统计报告\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"总样本数: {len(df)}\n")

        # 统计端板角度
        valid_angles = df['endplate_angle_deg'].dropna()
        if len(valid_angles) > 0:
            f.write(f"\n端板角度统计 (n={len(valid_angles)}):\n")
            f.write(f"  平均值: {valid_angles.mean():.2f}°\n")
            f.write(f"  标准差: {valid_angles.std():.2f}°\n")
            f.write(f"  范围: {valid_angles.min():.2f}° - {valid_angles.max():.2f}°\n")

        # 统计最大曲率
        valid_k_max = df['k_max_px_inv'].dropna()
        if len(valid_k_max) > 0:
            f.write(f"\n最大曲率统计 (n={len(valid_k_max)}):\n")
            f.write(f"  平均值: {valid_k_max.mean():.6f} px^-1\n")
            f.write(f"  标准差: {valid_k_max.std():.6f} px^-1\n")
            f.write(f"  范围: {valid_k_max.min():.6f} - {valid_k_max.max():.6f} px^-1\n")

        # 统计弯曲区域数量分布
        valid_regions = df['num_bend_regions'].dropna()
        if len(valid_regions) > 0:
            f.write(f"\n弯曲区域统计:\n")
            for i in range(1, int(valid_regions.max()) + 1):
                count = (valid_regions == i).sum()
                percentage = (count / len(valid_regions)) * 100
                f.write(f"  {i}个区域: {count}个样本 ({percentage:.1f}%)\n")

        # 统计处理成功率
        successful = len(df) - df['note'].notna().sum()
        success_rate = (successful / len(df)) * 100
        f.write(f"\n处理成功率: {successful}/{len(df)} ({success_rate:.1f}%)\n")

    print(f"统计报告已保存到: {report_path}")


# ---------------- 论文专用：三次样条曲线单独提取与可视化 ----------------
def extract_cubic_spline_curve(
    img_path, mask_path,
    save_dir=None,                # 曲线保存目录，默认保存到原OUT_DIR下的spline_curve子目录
    curve_color=(255, 0, 0),      # 曲线颜色，RGB格式，默认红色(255,0,0)，黑色为(0,0,0)
    curve_width=2.0,              # 曲线宽度，论文推荐1.5-2.0pt
    mark_vertebra=True,           # 是否标记椎体中心点和名称，True/False
    vertebra_mark_color=(255,255,255), # 椎体中心标记颜色，默认白色
    vertebra_text_color=(0,0,0),  # 椎体名称文字颜色，默认黑色
    background_type="transparent",# 背景类型：transparent(透明)、original(原图叠加)、white(白色背景)
    crop_roi=True,                # 是否裁剪ROI（仅保留脊柱区域），建议True
    roi_pad_y=40, roi_pad_x=60,   # ROI裁剪的padding，比原代码稍小，更紧凑
    save_dpi=600,                 # 保存分辨率，论文推荐600DPI
    save_format="png"             # 保存格式：png(透明/高清)、svg(矢量图，适合论文缩放)、jpg(原图叠加)
):
    """
    从单对图像和mask中单独提取三次样条拟合的脊柱曲线，生成论文专用的简洁图
    :param img_path: 原始图像路径
    :param mask_path: 对应的mask路径
    :return: 曲线保存路径（方便批量调用）
    """
    # 初始化保存目录
    if save_dir is None:
        save_dir = os.path.join(OUT_DIR, "spline_curve_paper")
    os.makedirs(save_dir, exist_ok=True)
    # 获取文件名（用于保存）
    basename = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(save_dir, f"{basename}_spine_spline.{save_format}")

    # 1. 复用原代码逻辑：读取图像、mask，提取椎体中心（核心，保证曲线一致）
    img = read_image(img_path)
    mask_idx = read_mask_index(mask_path)
    name_to_idx = {name: i for i, name in enumerate(NAME_CLASSES)}
    results = {}

    # 提取L1-L5/S1的椎体中心（与原代码完全一致）
    for vname in ["L1", "L2", "L3", "L4", "L5", "S1"]:
        if vname not in name_to_idx:
            continue
        idx = name_to_idx[vname]
        m = (mask_idx == idx)
        # 合并相邻椎间盘mask（原代码的容错逻辑）
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

    # 检查有效椎体（至少3个，与原代码一致）
    present = [v for v in ["L1", "L2", "L3", "L4", "L5"] if v in results]
    if len(present) < 3:
        print(f"[WARN] {basename} 有效椎体不足，无法拟合曲线")
        return None
    # 按y坐标排序（颅尾方向，与原代码一致）
    ordered = sorted([(v, results[v]['centroid']) for v in present], key=lambda kv: kv[1][1])
    names = [k for k, _ in ordered]
    centers = np.array([c for _, c in ordered])

    # 2. 复用原代码的三次样条拟合核心函数（保证曲线完全一致）
    cs_x, cs_y, t = fit_spine_spline(centers)
    if cs_x is None:
        print(f"[WARN] {basename} 样条拟合失败")
        return None

    # 3. 高分辨率采样拟合曲线（比原代码更密，曲线更光滑，论文更美观）
    s_plot = np.linspace(0.0, 1.0, 3000)  # 3000个采样点，远高于原代码，曲线无锯齿
    xs = cs_x(s_plot)  # 拟合曲线的x坐标
    ys = cs_y(s_plot)  # 拟合曲线的y坐标

    # 4. ROI裁剪（仅保留脊柱区域，剔除多余空白，论文排版更紧凑）
    H, W = mask_idx.shape
    if crop_roi:
        # 取mask非零区域+padding作为ROI（与原代码可视化的ROI逻辑一致，稍紧凑）
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
        # 转换曲线和椎体中心到ROI坐标系（裁剪后）
        xs_roi = xs - x0
        ys_roi = ys - y0
        centers_roi = np.array([(c[0]-x0, c[1]-y0) for c in centers])
        roi_h, roi_w = y1-y0, x1-x0
    else:
        xs_roi, ys_roi = xs, ys
        centers_roi = centers
        roi_h, roi_w = H, W

    # 5. 初始化画布（适配论文需求：透明/白色/原图背景）
    plt.rcParams['font.sans-serif'] = ['Arial']  # 论文常用无衬线字体，避免中文乱码
    plt.rcParams['axes.unicode_minus'] = False
    fig, ax = plt.subplots(figsize=(roi_w/100, roi_h/100), dpi=100)  # 按像素比例定尺寸，无拉伸
    ax.set_xlim(0, roi_w)
    ax.set_ylim(roi_h, 0)  # 反转y轴，匹配图像坐标系（原点在左上角）
    ax.axis('off')  # 关闭所有坐标轴，论文图无轴
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 无白边，满画布

    # 设置背景
    if background_type == "original":
        # 原图叠加曲线：裁剪原图并显示
        img_roi = img[y0:y1, x0:x1] if crop_roi else img
        ax.imshow(img_roi, interpolation='nearest')
    elif background_type == "white":
        # 白色背景
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
    elif background_type == "transparent":
        # 透明背景（论文最常用，方便叠加到论文模板）
        ax.set_facecolor('none')
        fig.patch.set_facecolor('none')

    # 6. 绘制核心：三次样条拟合曲线（论文级光滑）
    # ax.plot(xs_roi, ys_roi, color=np.array(curve_color)/255, linewidth=curve_width, solid_capstyle='round')
    # 新增：计算每个采样点的曲率（用于渐变配色，和原代码逻辑完全一致）
    dxp = cs_x.derivative(1)(s_plot)
    dyp = cs_y.derivative(1)(s_plot)
    ddxp = cs_x.derivative(2)(s_plot)
    ddyp = cs_y.derivative(2)(s_plot)
    nump = np.abs(dxp * ddyp - dyp * ddxp)
    denp = (dxp * dxp + dyp * dyp) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore'):
        k_plot = np.nan_to_num(nump / denp)
    # 曲率归一化（映射到[0,1]，用于颜色分配）
    k_norm = (k_plot - k_plot.min()) / (k_plot.max() - k_plot.min() + 1e-12) if k_plot.max() > 0 else k_plot

    # 核心：高密度彩色散点绘制渐变曲线（论文适配版）
    ax.scatter(
        xs_roi, ys_roi,  # 曲线采样点坐标
        c=k_norm,  # 按归一化曲率配色
        s=1.5,  # 点大小，1.0-2.0最佳（论文打印无锯齿）
        cmap='inferno',  # 渐变配色（和原代码一致：黑→红→黄，曲率越大越黄）
        alpha=0.95,  # 透明度，避免点重叠过亮
        linewidths=0,  # 无边框，更顺滑
        marker='.'  # 圆点，视觉最连续
    )

    # 7. 可选：标记椎体中心点和名称（轻量，无冗余，论文适配）
    if mark_vertebra:
        for name, (cx, cy) in zip(names, centers_roi):
            # 椎体中心标记：白色实心圆+黑边（清晰，不突兀）
            ax.scatter(cx, cy, s=30, c=np.array(vertebra_mark_color)/255,
                       edgecolors='black', linewidth=0.8, zorder=5)
            # 椎体名称：小字体，紧贴中心，无遮挡
            ax.text(cx + 5, cy, name, color=np.array(vertebra_text_color)/255,
                    fontsize=8, weight='bold', va='center', ha='left')

    # 8. 保存：高分辨率，无白边，适配论文格式
    plt.savefig(save_path, dpi=save_dpi, bbox_inches='tight', pad_inches=0.0, format=save_format)
    plt.close(fig)  # 释放内存，批量处理不卡顿
    print(f"[SUCCESS] 三次样条曲线已保存：{save_path}")
    return save_path

# ---------------- 批量提取曲线（可选，论文批量处理用） ----------------
def batch_extract_spline_curve(img_dir, mask_dir):
    """批量提取目录下所有图像的三次样条曲线，与原代码批量逻辑一致"""
    img_paths = sorted(glob(os.path.join(img_dir, '*.jpg')))
    print(f"开始批量提取 {len(img_paths)} 个脊柱的三次样条曲线...")
    for i, img_path in enumerate(img_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, f"{basename}.png")
        if not os.path.exists(mask_path):
            print(f"[WARN] {basename} mask缺失，跳过")
            continue
        print(f"处理 {i+1}/{len(img_paths)}: {basename}")
        extract_cubic_spline_curve(img_path, mask_path)
    print("批量提取完成！所有曲线保存在：", os.path.join(OUT_DIR, "spline_curve_paper"))

# ---------------- 主程序入口 ----------------
# if __name__ == "__main__":
#     # 当脚本直接运行时，执行批量处理
#     improved_batch_process(IMG_DIR, MASK_DIR, OUT_DIR)

if __name__ == "__main__":
    # 单张提取：替换为你的图像和mask路径
    IMG_PATH = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\img\22.jpg"
    MASK_PATH = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\mask\22.png"
    extract_cubic_spline_curve(IMG_PATH, MASK_PATH)
