import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import csv

# 设置字体：英文优先Times New Roman，中文后备SimHei
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- 配置参数 --------------------------
ORIGINAL_IMG_DIR = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-05_compute\img"
MASK_DIR = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-05_compute\mask"
OUTPUT_PARENT_DIR = r"D:\unet_test\Deeplabv3+\Deeplabv3_plus_ours\deeplabv3-plus-pytorch-main-ours\cobb12_amace_cobb_results"
os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

VERTEBRA_LABELS = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5"
}

VERTEBRA_COLORS = {
    "L1": "red",
    "L2": "green",
    "L3": "blue",
    "L4": "yellow",
    "L5": "magenta"
}

# -------------------------- 直线检测与处理函数 --------------------------
def extend_line_to_border(p1, p2, img_shape):
    """将线段延长至图像边界，返回两个边界交点"""
    h, w = img_shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:  # 垂直线
        return (x1, 0), (x1, h-1)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    intersections = []
    # 左边界 x=0
    y_left = b
    if 0 <= y_left <= h-1:
        intersections.append((0, y_left))
    # 右边界 x=w-1
    y_right = m * (w-1) + b
    if 0 <= y_right <= h-1:
        intersections.append((w-1, y_right))
    # 上边界 y=0
    if m != 0:
        x_top = -b / m
        if 0 <= x_top <= w-1:
            intersections.append((x_top, 0))
    # 下边界 y=h-1
    if m != 0:
        x_bottom = (h-1 - b) / m
        if 0 <= x_bottom <= w-1:
            intersections.append((x_bottom, h-1))
    if len(intersections) >= 2:
        # 选择距离两个端点之和最远的两个点（即直线贯穿图像的两个交点）
        distances = [np.linalg.norm(np.array(p) - np.array(p1)) + np.linalg.norm(np.array(p) - np.array(p2)) for p in intersections]
        sorted_indices = np.argsort(distances)[-2:]
        return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
    return p1, p2  # 保底返回原线段端点

def calculate_cobb_angle(line1, line2):
    """计算两条直线之间的锐角（Cobb角）"""
    v1 = np.array([line1[1][0] - line1[0][0], line1[1][1] - line1[0][1]])
    v2 = np.array([line2[1][0] - line2[0][0], line2[1][1] - line2[0][1]])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0
    cos_angle = np.clip(dot / norm, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    if angle > 90:
        angle = 180 - angle
    return angle

def detect_endplate_lines(vertebra_mask, label_name):
    """
    对单个椎体掩码，使用Canny边缘检测和霍夫变换检测上下终板直线。
    返回两条直线，每条直线用两个点表示（线段的两个端点）。
    """
    # 确保掩码为二值图
    mask = vertebra_mask.copy().astype(np.uint8)
    if np.sum(mask) == 0:
        print(f"警告：{label_name}掩码为空，跳过")
        return None, None

    # 获取椎体边界框，用于后续筛选直线
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    center_y = (y_min + y_max) // 2

    # 边缘检测（Canny）
    edges = cv2.Canny(mask, 50, 150)  # 阈值可根据实际情况调整

    # 霍夫变换检测线段（使用概率霍夫变换）
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=max(20, (x_max-x_min)//3),
                            maxLineGap=10)

    if lines is None or len(lines) == 0:
        print(f"警告：{label_name}未检测到直线，使用边界框上下边代替")
        # 回退：用边界框的上边和下边作为终板线
        top_line = [(x_min, y_min), (x_max, y_min)]
        bottom_line = [(x_min, y_max), (x_max, y_max)]
        return top_line, bottom_line

    # 收集所有线段
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((x1, y1), (x2, y2)))

    # 根据线段中点的y坐标分配到上/下候选集
    top_candidates = []
    bottom_candidates = []
    for seg in segments:
        mid_y = (seg[0][1] + seg[1][1]) / 2
        if mid_y < center_y:
            top_candidates.append(seg)
        else:
            bottom_candidates.append(seg)

    # 如果没有候选，则全部放入对应集
    if len(top_candidates) == 0:
        top_candidates = segments
    if len(bottom_candidates) == 0:
        bottom_candidates = segments

    # 从候选中选择最长的线段作为代表（也可考虑合并多条线段）
    def select_longest(candidates):
        if not candidates:
            return None
        longest = max(candidates, key=lambda s: np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]))
        return longest

    top_line = select_longest(top_candidates)
    bottom_line = select_longest(bottom_candidates)

    # 如果某一侧没有直线，则用边界框边代替
    if top_line is None:
        top_line = [(x_min, y_min), (x_max, y_min)]
        print(f"警告：{label_name}上终板未检测到直线，使用边界框上边")
    if bottom_line is None:
        bottom_line = [(x_min, y_max), (x_max, y_max)]
        print(f"警告：{label_name}下终板未检测到直线，使用边界框下边")

    return top_line, bottom_line

# -------------------------- 可视化函数 --------------------------
def visualize_endplates(original_img, mask, vertebra_lines, output_path):
    """
    vertebra_lines: dict, key为椎体名，value为 (top_line, bottom_line)
    每条line为两个点组成的元组 ((x1,y1),(x2,y2))
    """
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # 绘制每个椎体的终板线（延伸至边界）
    extended_lines = {}  # 保存延伸后的直线，用于计算Cobb角
    for name, (top_line, bottom_line) in vertebra_lines.items():
        color = VERTEBRA_COLORS[name]
        # 延伸至图像边界
        top_ext = extend_line_to_border(top_line[0], top_line[1], img_rgb.shape)
        bottom_ext = extend_line_to_border(bottom_line[0], bottom_line[1], img_rgb.shape)
        extended_lines[name] = (top_ext, bottom_ext)

        # 绘制延伸后的直线（红色）
        plt.plot([top_ext[0][0], top_ext[1][0]], [top_ext[0][1], top_ext[1][1]],
                 color='red', linewidth=2, zorder=3)
        plt.plot([bottom_ext[0][0], bottom_ext[1][0]], [bottom_ext[0][1], bottom_ext[1][1]],
                 color='red', linewidth=2, zorder=3)

        # 可选：绘制原始线段（更细的虚线），用于展示检测到的线段
        plt.plot([top_line[0][0], top_line[1][0]], [top_line[0][1], top_line[1][1]],
                 color=color, linewidth=1, linestyle='--', zorder=2)
        plt.plot([bottom_line[0][0], bottom_line[1][0]], [bottom_line[0][1], bottom_line[1][1]],
                 color=color, linewidth=1, linestyle='--', zorder=2)

        # 标注椎体名称
        mid_x = (top_line[0][0] + top_line[1][0] + bottom_line[0][0] + bottom_line[1][0]) / 4
        mid_y = (top_line[0][1] + top_line[1][1] + bottom_line[0][1] + bottom_line[1][1]) / 4
        plt.text(mid_x, mid_y, name, color=color, fontsize=12, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # 绘制每对Cobb角弧线
    for i in range(1, 5):
        upper = extended_lines[f"L{i}"][1]   # 上椎体下终板
        lower = extended_lines[f"L{i+1}"][0] # 下椎体上终板
        angle = calculate_cobb_angle(upper, lower)
        # 在两线中间位置标注角度
        mid1 = ((upper[0][0] + upper[1][0]) / 2, (upper[0][1] + upper[1][1]) / 2)
        mid2 = ((lower[0][0] + lower[1][0]) / 2, (lower[0][1] + lower[1][1]) / 2)
        center = ((mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2)
        plt.text(center[0], center[1], f"L{i}-L{i+1} Cobb:{angle:.1f}°",
                 color='yellow', fontsize=10, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))

    # 计算L1-L5总Cobb角
    L1_line = extended_lines["L1"][1]
    L5_line = extended_lines["L5"][0]
    total_angle = calculate_cobb_angle(L1_line, L5_line)
    plt.text(20, 30, f"L1-L5 Cobb Angle: {total_angle:.2f}°",
             color='white', fontsize=14, weight='bold',
             bbox=dict(facecolor='black', alpha=0.8, pad=3))

    plt.title("Cobb Angle Measurement based on Hough Transform", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Cobb angle result saved to：{output_path}")

# -------------------------- 批量处理函数（返回角度数据） --------------------------
def process_single_image(original_img_path, mask_path, output_dir):
    """
    处理单张图像，返回包含角度信息的字典；若失败返回None。
    """
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Error: cannot read original image：{original_img_path}")
        return None

    mask = Image.open(mask_path)
    mask_array = np.array(mask, dtype=np.int32)

    vertebra_lines = {}  # 存储每个椎体的 (top_line, bottom_line)
    for label, name in VERTEBRA_LABELS.items():
        print(f"Processing {name}...")
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        top_line, bottom_line = detect_endplate_lines(vertebra_mask, name)
        if top_line is None or bottom_line is None:
            print(f"Error: {name} endplate detection failed, skip this image.")
            return None
        vertebra_lines[name] = (top_line, bottom_line)

    # 计算各节段Cobb角（需先延伸至边界）
    img_shape = original_img.shape[:2]
    extended_lines = {}
    for name, (top_line, bottom_line) in vertebra_lines.items():
        top_ext = extend_line_to_border(top_line[0], top_line[1], img_shape)
        bottom_ext = extend_line_to_border(bottom_line[0], bottom_line[1], img_shape)
        extended_lines[name] = (top_ext, bottom_ext)

    segment_angles = {}
    for i in range(1, 5):
        upper = extended_lines[f"L{i}"][1]      # L_i下终板
        lower = extended_lines[f"L{i+1}"][0]    # L_{i+1}上终板
        angle = calculate_cobb_angle(upper, lower)
        segment_angles[f"L{i}_L{i+1}"] = angle

    # 计算总体L1-L5 Cobb角（L1下终板与L5上终板）注意与原可视化保持一致
    L1_line = extended_lines["L1"][1]
    L5_line = extended_lines["L5"][0]
    total_angle = calculate_cobb_angle(L1_line, L5_line)

    # 保存可视化结果
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_cobb_hough.png"
    output_path = os.path.join(output_dir, output_filename)
    visualize_endplates(original_img, mask_array, vertebra_lines, output_path)

    # 保存直线坐标信息
    coords_path = os.path.join(output_dir, output_filename.replace(".png", "_lines.txt"))
    with open(coords_path, "w", encoding="utf-8") as f:
        for name, (top, bottom) in vertebra_lines.items():
            f.write(f"====={name} Endplate Lines=====\n")
            f.write(f"Upper endplate: {top[0]} -> {top[1]}\n")
            f.write(f"Lower endplate: {bottom[0]} -> {bottom[1]}\n\n")
    print(f"Line coordinates saved to：{coords_path}")

    # 返回角度数据
    result = {
        'image': os.path.splitext(os.path.basename(original_img_path))[0],
        'L1_L5_total': total_angle,
        **segment_angles
    }
    return result

def main():
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    original_images = [f for f in os.listdir(ORIGINAL_IMG_DIR) if any(f.lower().endswith(ext) for ext in img_extensions)]
    if not original_images:
        print(f"No images found in {ORIGINAL_IMG_DIR}")
        return

    all_results = []  # 收集所有成功图像的角度数据

    for img_filename in original_images:
        print(f"\nProcessing image：{img_filename}")
        original_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)
        img_name_without_ext = os.path.splitext(img_filename)[0]
        mask_filename = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_mask_path = os.path.join(MASK_DIR, img_name_without_ext + ext)
            if os.path.exists(potential_mask_path):
                mask_filename = img_name_without_ext + ext
                break
        if mask_filename is None:
            print(f"Warning: mask file for {img_filename} not found, skip.")
            continue
        mask_path = os.path.join(MASK_DIR, mask_filename)
        image_output_dir = os.path.join(OUTPUT_PARENT_DIR, img_name_without_ext)
        os.makedirs(image_output_dir, exist_ok=True)

        result = process_single_image(original_img_path, mask_path, image_output_dir)
        if result is not None:
            all_results.append(result)

    # 生成汇总表格
    if all_results:
        summary_path = os.path.join(OUTPUT_PARENT_DIR, "cobb_summary.csv")
        fieldnames = ['image', 'L1_L5_total', 'L1_L2', 'L2_L3', 'L3_L4', 'L4_L5']
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in all_results:
                row = {
                    'image': res['image'],
                    'L1_L5_total': f"{res['L1_L5_total']:.2f}",
                    'L1_L2': f"{res.get('L1_L2', 0):.2f}",
                    'L2_L3': f"{res.get('L2_L3', 0):.2f}",
                    'L3_L4': f"{res.get('L3_L4', 0):.2f}",
                    'L4_L5': f"{res.get('L4_L5', 0):.2f}",
                }
                writer.writerow(row)
        print(f"\nSummary table saved to: {summary_path}")

    print(f"\nBatch processing completed! Results saved in：{OUTPUT_PARENT_DIR}")

if __name__ == "__main__":
    main()
