import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math

# 设置中文字体支持
plt.rcParams["font.sans-serif"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 配置参数 --------------------------
# 输入路径（改为目录路径）
ORIGINAL_IMG_DIR = "img/"
MASK_DIR = "mask/"

# 输出父目录
OUTPUT_PARENT_DIR = "cobb8_amace_cobb_results"
os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

# 椎体标签与名称映射
VERTEBRA_LABELS = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    11: "S1"
}

# 角点可视化配置（不同椎体不同颜色）
VERTEBRA_COLORS = {
    "L1": "red",
    "L2": "green",
    "L3": "blue",
    "L4": "yellow",
    "L5": "magenta",
    "S1": "cyan"
}


# -------------------------- 角点检测核心函数 --------------------------
def fallback_HV_calculation(src):
    """边界框法获取角点（备用方案）"""
    points = np.argwhere(src > 0)
    if len(points) == 0:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]])  # 空角点

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)

    # 返回 TL, TR, BL, BR 顺序的角点
    return np.array([
        [x_min, y_min],  # TL
        [x_max, y_min],  # TR
        [x_min, y_max],  # BL
        [x_max, y_max]  # BR
    ], dtype=np.int32)


def detect_vertebra_corners(vertebra_mask, is_S1=False):
    """检测单个椎体的4个角点（TL, TR, BL, BR）"""
    src = vertebra_mask.copy().astype(np.uint8)

    # 检查掩码有效性
    if np.sum(src) == 0:
        print("警告：椎体掩码为空，使用边界框角点")
        return fallback_HV_calculation(src)

    # 转换为适合角点检测的格式
    gray = np.float32(src)

    # Shi-Tomasi角点检测参数（S1单独调整）
    max_corners = 4
    qualityLevel = 0.005 if is_S1 else 0.01
    minDistance = 10 if is_S1 else 21
    blockSize = 9

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        blockSize=blockSize
    )

    # 若检测失败，使用边界框角点
    if corners is None or len(corners) < 4:
        print(f"警告：角点检测不足4个，使用边界框角点")
        return fallback_HV_calculation(src)

    # 处理角点格式
    corners = np.int32(corners.squeeze())

    # 确保角点数量为4
    if len(corners) > 4:
        corners = corners[:4]
    elif len(corners) < 4:
        return fallback_HV_calculation(src)

    # 角点排序：TL(左上)、TR(右上)、BL(左下)、BR(右下)
    # 1. 按y坐标区分上下（y值小的为上，y值大的为下）
    corners_sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_corners = corners_sorted_by_y[:2]  # 上半部分2个点
    bottom_corners = corners_sorted_by_y[2:]  # 下半部分2个点

    # 2. 按x坐标区分左右（x值小的为左，x值大的为右）
    top_corners_sorted = top_corners[np.argsort(top_corners[:, 0])]
    bottom_corners_sorted = bottom_corners[np.argsort(bottom_corners[:, 0])]

    # 组合为最终顺序
    return np.array([
        top_corners_sorted[0],  # TL
        top_corners_sorted[1],  # TR
        bottom_corners_sorted[0],  # BL
        bottom_corners_sorted[1]  # BR
    ], dtype=np.int32)


def extend_line_to_border(p1, p2, img_shape):
    """
    将线段延伸到图像边界但不超出
    返回延伸后的两个端点
    """
    h, w = img_shape[:2]
    x1, y1 = p1
    x2, y2 = p2

    # 计算直线参数
    if x2 == x1:  # 垂直线
        # 延伸到图像上下边界
        return (x1, 0), (x1, h - 1)

    # 计算斜率
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # 计算与图像边界的交点
    intersections = []

    # 与左边界 (x=0) 的交点
    y_left = m * 0 + b
    if 0 <= y_left <= h - 1:
        intersections.append((0, y_left))

    # 与右边界 (x=w-1) 的交点
    y_right = m * (w - 1) + b
    if 0 <= y_right <= h - 1:
        intersections.append((w - 1, y_right))

    # 与上边界 (y=0) 的交点
    if m != 0:
        x_top = (0 - b) / m
        if 0 <= x_top <= w - 1:
            intersections.append((x_top, 0))

    # 与下边界 (y=h-1) 的交点
    if m != 0:
        x_bottom = (h - 1 - b) / m
        if 0 <= x_bottom <= w - 1:
            intersections.append((x_bottom, h - 1))

    # 如果找到至少两个交点，取最远的两个
    if len(intersections) >= 2:
        # 计算交点与原线段的距离
        distances = [np.linalg.norm(np.array(p) - np.array(p1)) + np.linalg.norm(np.array(p) - np.array(p2))
                     for p in intersections]

        # 取距离最远的两个点
        sorted_indices = np.argsort(distances)[-2:]
        return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
    else:
        # 如果没有足够交点，返回原始点
        return p1, p2


def calculate_perpendicular_line(p1, p2, img_shape):
    """
    计算通过线段中点的垂直线，并延伸到图像边界
    返回垂直线段的两个端点
    """
    # 计算线段中点
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    midpoint = (mid_x, mid_y)

    # 计算线段向量
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # 计算垂直向量（旋转90度）
    if dx == 0:  # 垂直线段
        # 原线段垂直，垂直线水平
        perp_p1 = (0, mid_y)
        perp_p2 = (img_shape[1] - 1, mid_y)
    elif dy == 0:  # 水平线段
        # 原线段水平，垂直线垂直
        perp_p1 = (mid_x, 0)
        perp_p2 = (mid_x, img_shape[0] - 1)
    else:
        # 计算斜率
        m = dy / dx
        # 垂直线斜率是原斜率的负倒数
        m_perp = -1 / m

        # 计算垂直线方程 y - mid_y = m_perp * (x - mid_x)
        # 找到垂直线与图像边界的交点
        perp_p1, perp_p2 = extend_line_to_border(
            (mid_x, mid_y),
            (mid_x + 1, mid_y + m_perp),  # 使用斜率定义第二个点
            img_shape
        )

    return perp_p1, perp_p2, midpoint


# -------------------------- 可视化函数 --------------------------
def visualize_corners(original_img, mask, all_corners, output_path):
    """在原图上绘制所有椎体的角点并保存"""
    # 转换原图为RGB格式
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # 创建画布
    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # 绘制每个椎体的角点
    for vertebra_name, corners in all_corners.items():
        color = VERTEBRA_COLORS[vertebra_name]

        # 绘制角点
        for (x, y) in corners:
            plt.scatter(x, y, color=color, s=80, marker='o',
                        edgecolors='white', linewidth=2, zorder=5)

        # 标注角点名称（TL, TR, BL, BR）
        corner_labels = [f"{vertebra_name}_TL", f"{vertebra_name}_TR",
                         f"{vertebra_name}_BL", f"{vertebra_name}_BR"]
        for i, (x, y) in enumerate(corners):
            plt.text(x + 5, y - 5, corner_labels[i],
                     color=color, fontsize=10, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.7, pad=1), zorder=6)

        # 绘制椎体轮廓（增强可视化）
        vertebra_mask = (mask == [k for k, v in VERTEBRA_LABELS.items() if v == vertebra_name][0]).astype(np.uint8)
        contours, _ = cv2.findContours(vertebra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 3:
                plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=color, linewidth=2, linestyle='--', zorder=4)

    # 绘制L1上终板线和L5下终板线（改为 L5）
    if 'L1' in all_corners:
        l1_corners = all_corners['L1']
        # L1上终板线：连接TL和TR，并延伸到图像边界
        l1_tl = l1_corners[0]  # TL
        l1_tr = l1_corners[1]  # TR

        # 延伸线段到图像边界
        l1_extended_p1, l1_extended_p2 = extend_line_to_border(l1_tl, l1_tr, img_rgb.shape[:2])

        # 绘制L1上终板线
        plt.plot([l1_extended_p1[0], l1_extended_p2[0]],
                 [l1_extended_p1[1], l1_extended_p2[1]],
                 color='red', linewidth=3, linestyle='-', label='L1上终板线', zorder=3)

        # 标记L1终板线
        mid_x = (l1_extended_p1[0] + l1_extended_p2[0]) / 2
        mid_y = (l1_extended_p1[1] + l1_extended_p2[1]) / 2
        plt.text(mid_x, mid_y - 10, 'L1上终板线', color='red', fontsize=12,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=2))

    if 'L5' in all_corners:
        l5_corners = all_corners['L5']
        # L5下终板线：连接BR和BL，并延伸到图像边界
        l5_br = l5_corners[3]  # BR
        l5_bl = l5_corners[2]  # BL

        # 延伸线段到图像边界
        l5_extended_p1, l5_extended_p2 = extend_line_to_border(l5_br, l5_bl, img_rgb.shape[:2])

        # 绘制L5下终板线
        plt.plot([l5_extended_p1[0], l5_extended_p2[0]],
                 [l5_extended_p1[1], l5_extended_p2[1]],
                 color='cyan', linewidth=3, linestyle='-', label='L5下终板线', zorder=3)

        # 标记L5终板线
        mid_x = (l5_extended_p1[0] + l5_extended_p2[0]) / 2
        mid_y = (l5_extended_p1[1] + l5_extended_p2[1]) / 2
        plt.text(mid_x, mid_y + 10, 'L5下终板线', color='cyan', fontsize=12,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=2))

    # 绘制L1上终板线的垂线
    if 'L1' in all_corners:
        l1_corners = all_corners['L1']
        l1_tl = l1_corners[0]  # TL
        l1_tr = l1_corners[1]  # TR

        # 计算L1上终板线的垂线
        l1_perp_p1, l1_perp_p2, l1_midpoint = calculate_perpendicular_line(l1_tl, l1_tr, img_rgb.shape[:2])

        # 绘制L1上终板线的垂线
        plt.plot([l1_perp_p1[0], l1_perp_p2[0]],
                 [l1_perp_p1[1], l1_perp_p2[1]],
                 color='orange', linewidth=2.5, linestyle='-', label='L1终板垂线', zorder=3)

        # 标记L1垂线中点
        plt.scatter([l1_midpoint[0]], [l1_midpoint[1]],
                    color='orange', s=60, marker='s', edgecolors='white', linewidth=1.5, zorder=5)

        # 标记L1垂线
        mid_x = (l1_perp_p1[0] + l1_perp_p2[0]) / 2
        mid_y = (l1_perp_p1[1] + l1_perp_p2[1]) / 2
        plt.text(mid_x + 5, mid_y, 'L1终板垂线', color='orange', fontsize=10,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=1))

    # 绘制L5下终板线的垂线（替换原来的S1）
    if 'L5' in all_corners:
        l5_corners = all_corners['L5']
        l5_br = l5_corners[3]  # BR
        l5_bl = l5_corners[2]  # BL

        # 计算L5下终板线的垂线
        l5_perp_p1, l5_perp_p2, l5_midpoint = calculate_perpendicular_line(l5_br, l5_bl, img_rgb.shape[:2])

        # 绘制L5下终板线的垂线
        plt.plot([l5_perp_p1[0], l5_perp_p2[0]],
                 [l5_perp_p1[1], l5_perp_p2[1]],
                 color='purple', linewidth=2.5, linestyle='-', label='L5终板垂线', zorder=3)

        # 标记L5垂线中点
        plt.scatter([l5_midpoint[0]], [l5_midpoint[1]],
                    color='purple', s=60, marker='s', edgecolors='white', linewidth=1.5, zorder=5)

        # 标记L5垂线
        mid_x = (l5_perp_p1[0] + l5_perp_p2[0]) / 2
        mid_y = (l5_perp_p1[1] + l5_perp_p2[1]) / 2
        plt.text(mid_x + 5, mid_y, 'L5终板垂线', color='purple', fontsize=10,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=1))

    # 计算Cobb角（L1和L5垂线之间的夹角）
    if 'L1' in all_corners and 'L5' in all_corners:
        # 计算L1和L5垂线的方向向量
        l1_perp_p1, l1_perp_p2, _ = calculate_perpendicular_line(
            all_corners['L1'][0], all_corners['L1'][1], img_rgb.shape[:2]
        )
        l5_perp_p1, l5_perp_p2, _ = calculate_perpendicular_line(
            all_corners['L5'][3], all_corners['L5'][2], img_rgb.shape[:2]  # 使用 L5 的 BR 和 BL
        )

        # 计算方向向量
        l1_vec = np.array([l1_perp_p2[0] - l1_perp_p1[0], l1_perp_p2[1] - l1_perp_p1[1]])
        l5_vec = np.array([l5_perp_p2[0] - l5_perp_p1[0], l5_perp_p2[1] - l5_perp_p1[1]])

        # 计算向量夹角（Cobb角）
        dot_product = np.dot(l1_vec, l5_vec)
        l1_norm = np.linalg.norm(l1_vec)
        l5_norm = np.linalg.norm(l5_vec)

        if l1_norm > 0 and l5_norm > 0:
            cos_angle = dot_product / (l1_norm * l5_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免浮点误差
            cobb_angle = np.degrees(np.arccos(cos_angle))
            # 取锐角（小于90度）
            if cobb_angle > 90:
                cobb_angle = 180 - cobb_angle

            # 显示Cobb角
            plt.text(20, 30, f'Cobb角 (L1-L5): {cobb_angle:.2f}°',
                     color='white', fontsize=14, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.8, pad=3))

    # 配置图像
    plt.title("椎体角点标注结果（含终板线及其垂线）", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # 保存结果
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"角点标注结果已保存至：{output_path}")


# -------------------------- 批量处理函数 --------------------------
def process_single_image(original_img_path, mask_path, output_dir):
    """处理单张图像"""
    # 读取输入数据
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"错误：无法读取原图：{original_img_path}")
        return False

    mask = Image.open(mask_path)
    mask_array = np.array(mask, dtype=np.int32)

    # 检测所有椎体的角点
    all_corners = {}
    for label, name in VERTEBRA_LABELS.items():
        print(f"检测 {name} 角点...")
        # 提取单个椎体的掩码
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        # 检测角点（S1单独处理）
        is_S1 = (name == "S1")
        corners = detect_vertebra_corners(vertebra_mask, is_S1)
        all_corners[name] = corners

    # 可视化并保存结果
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_corners.png"
    output_path = os.path.join(output_dir, output_filename)
    visualize_corners(original_img, mask_array, all_corners, output_path)

    # 保存角点坐标到文本文件
    coords_path = os.path.join(output_dir, output_filename.replace(".png", "_coords.txt"))
    with open(coords_path, "w", encoding="utf-8") as f:
        for name, corners in all_corners.items():
            f.write(f"====={name}角点坐标=====\n")
            f.write("TL(左上) | TR(右上) | BL(左下) | BR(右下)\n")
            f.write(f"{corners[0]} | {corners[1]} | {corners[2]} | {corners[3]}\n\n")

    # 保存终板线和垂线信息
    lines_path = os.path.join(output_dir, output_filename.replace(".png", "_lines.txt"))
    with open(lines_path, "w", encoding="utf-8") as f:
        f.write("=====终板线信息=====\n")
        if 'L1' in all_corners:
            l1_corners = all_corners['L1']
            l1_tl = l1_corners[0]
            l1_tr = l1_corners[1]
            l1_extended_p1, l1_extended_p2 = extend_line_to_border(l1_tl, l1_tr, original_img.shape[:2])
            l1_perp_p1, l1_perp_p2, l1_midpoint = calculate_perpendicular_line(l1_tl, l1_tr, original_img.shape[:2])

            f.write("L1上终板线:\n")
            f.write(f"原始点: TL{l1_tl} -> TR{l1_tr}\n")
            f.write(f"延伸后: {l1_extended_p1} -> {l1_extended_p2}\n")
            f.write(f"垂线: {l1_perp_p1} -> {l1_perp_p2}\n")
            f.write(f"中点: {l1_midpoint}\n\n")

        # 将 S1 部分替换为 L5（保存 L5 的终板信息）
        if 'L5' in all_corners:
            l5_corners = all_corners['L5']
            l5_br = l5_corners[3]  # BR
            l5_bl = l5_corners[2]  # BL

            l5_extended_p1, l5_extended_p2 = extend_line_to_border(l5_br, l5_bl, original_img.shape[:2])
            l5_perp_p1, l5_perp_p2, l5_midpoint = calculate_perpendicular_line(l5_br, l5_bl, original_img.shape[:2])

            f.write("L5下终板线:\n")
            f.write(f"原始点: BR{l5_br} -> BL{l5_bl}\n")
            f.write(f"延伸后: {l5_extended_p1} -> {l5_extended_p2}\n")
            f.write(f"垂线: {l5_perp_p1} -> {l5_perp_p2}\n")
            f.write(f"中点: {l5_midpoint}\n\n")

        # 计算并保存Cobb角（改为 L1 与 L5 的垂线之间角度）
        if 'L1' in all_corners and 'L5' in all_corners:
            l1_perp_p1, l1_perp_p2, _ = calculate_perpendicular_line(
                all_corners['L1'][0], all_corners['L1'][1], original_img.shape[:2]
            )
            l5_perp_p1, l5_perp_p2, _ = calculate_perpendicular_line(
                all_corners['L5'][3], all_corners['L5'][2], original_img.shape[:2]  # 使用 L5 的 BR 和 BL
            )

            l1_vec = np.array([l1_perp_p2[0] - l1_perp_p1[0], l1_perp_p2[1] - l1_perp_p1[1]])
            l5_vec = np.array([l5_perp_p2[0] - l5_perp_p1[0], l5_perp_p2[1] - l5_perp_p1[1]])

            dot_product = np.dot(l1_vec, l5_vec)
            l1_norm = np.linalg.norm(l1_vec)
            l5_norm = np.linalg.norm(l5_vec)

            if l1_norm > 0 and l5_norm > 0:
                cos_angle = dot_product / (l1_norm * l5_norm)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                cobb_angle = np.degrees(np.arccos(cos_angle))
                if cobb_angle > 90:
                    cobb_angle = 180 - cobb_angle

                f.write(f"Cobb角 (L1-L5): {cobb_angle:.2f}°\n")

    print(f"角点坐标已保存至：{coords_path}")
    print(f"终板线和垂线信息已保存至：{lines_path}")
    return True


def main():
    # 获取所有图像文件
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    original_images = []

    for filename in os.listdir(ORIGINAL_IMG_DIR):
        if any(filename.lower().endswith(ext) for ext in img_extensions):
            original_images.append(filename)

    if not original_images:
        print(f"在目录 {ORIGINAL_IMG_DIR} 中未找到图像文件")
        return

    print(f"找到 {len(original_images)} 张图像需要处理")

    # 处理每张图像
    for img_filename in original_images:
        print(f"\n正在处理图像：{img_filename}")

        # 构建完整路径
        original_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)

        # 查找对应的掩码文件（假设文件名相同，扩展名不同）
        img_name_without_ext = os.path.splitext(img_filename)[0]
        mask_filename = None

        # 尝试不同的掩码文件扩展名
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_mask_path = os.path.join(MASK_DIR, img_name_without_ext + ext)
            if os.path.exists(potential_mask_path):
                mask_filename = img_name_without_ext + ext
                break

        if mask_filename is None:
            print(f"警告：未找到 {img_filename} 对应的掩码文件，跳过处理")
            continue

        mask_path = os.path.join(MASK_DIR, mask_filename)

        # 为当前图像创建单独的输出目录
        image_output_dir = os.path.join(OUTPUT_PARENT_DIR, img_name_without_ext)
        os.makedirs(image_output_dir, exist_ok=True)

        # 处理单张图像
        success = process_single_image(original_img_path, mask_path, image_output_dir)

        if success:
            print(f"成功处理图像：{img_filename}")
        else:
            print(f"处理图像失败：{img_filename}")

    print(f"\n批量处理完成！所有结果保存在：{OUTPUT_PARENT_DIR}")


if __name__ == "__main__":
    main()
