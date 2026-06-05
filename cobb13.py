# 终板线求cobb角（L1-L5版本）——基于轮廓拟合直线
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
OUTPUT_PARENT_DIR = r"D:\unet_test\Deeplabv3+\Deeplabv3_plus_ours\deeplabv3-plus-pytorch-main-ours\cobb13_amace_cobb_results"
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

# -------------------------- 直线检测函数（基于轮廓拟合） --------------------------
def fit_line_from_points(points):
    """对点集进行最小二乘直线拟合，返回直线斜率和截距 (k, b) 或垂直线标志"""
    if len(points) < 2:
        return None, None
    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    # 如果x坐标变化很小，视为垂直线
    if np.max(x) - np.min(x) < 1e-6:
        return None, x[0]  # 返回垂直线x坐标
    # 最小二乘拟合 y = kx + b
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

def line_to_two_points(k, b, is_vertical, x_fixed, img_shape):
    """将直线参数转换为图像边界上的两个点"""
    h, w = img_shape[:2]
    if is_vertical:
        x = int(x_fixed)
        return (x, 0), (x, h-1)
    else:
        # 计算直线与图像左右边界的交点
        y_left = k * 0 + b
        y_right = k * (w-1) + b
        # 确保交点在图像高度范围内
        p_left = (0, int(np.clip(y_left, 0, h-1)))
        p_right = (w-1, int(np.clip(y_right, 0, h-1)))
        return p_left, p_right

def detect_endplate_lines_from_mask(vertebra_mask, label_name, img_shape):
    """
    从椎体掩码中提取上下终板直线。
    返回 (top_line, bottom_line)，每条线用两个端点表示（图像边界交点）。
    """
    mask = vertebra_mask.copy().astype(np.uint8)
    if np.sum(mask) == 0:
        print(f"警告：{label_name}掩码为空")
        return None, None

    # 获取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"警告：{label_name}未找到轮廓")
        return None, None

    # 取最大轮廓（应为椎体外轮廓）
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.squeeze().reshape(-1, 2)  # (N, 2)

    # 根据y坐标分离上下边缘点
    ys = contour_points[:, 1]
    y_min, y_max = np.min(ys), np.max(ys)
    center_y = (y_min + y_max) / 2

    # 上半部分点（y < center_y）
    top_mask = contour_points[:, 1] < center_y
    top_points = contour_points[top_mask]
    # 下半部分点（y > center_y）
    bottom_mask = contour_points[:, 1] > center_y
    bottom_points = contour_points[bottom_mask]

    # 如果某部分点太少，使用边界框替代
    if len(top_points) < 5:
        print(f"警告：{label_name}上边缘点不足，使用边界框上边")
        xs = contour_points[:, 0]
        top_line = [(np.min(xs), y_min), (np.max(xs), y_min)]
    else:
        # 拟合上边缘直线
        k, b = fit_line_from_points(top_points)
        if k is None:
            # 垂直线情况
            top_line = line_to_two_points(None, None, True, b, img_shape)
        else:
            top_line = line_to_two_points(k, b, False, None, img_shape)

    if len(bottom_points) < 5:
        print(f"警告：{label_name}下边缘点不足，使用边界框下边")
        xs = contour_points[:, 0]
        bottom_line = [(np.min(xs), y_max), (np.max(xs), y_max)]
    else:
        k, b = fit_line_from_points(bottom_points)
        if k is None:
            bottom_line = line_to_two_points(None, None, True, b, img_shape)
        else:
            bottom_line = line_to_two_points(k, b, False, None, img_shape)

    return top_line, bottom_line

def extend_line_to_border(p1, p2, img_shape):
    """将线段延长至图像边界（兼容原接口）"""
    # 由于line_to_two_points已经直接返回边界点，此函数可简化直接返回输入
    # 保留以兼容原可视化代码
    return p1, p2

# -------------------------- 角度计算函数 --------------------------
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

# -------------------------- 可视化函数 --------------------------
def visualize_endplates(original_img, mask, vertebra_lines, output_path):
    """
    vertebra_lines: dict, key为椎体名，value为 (top_line, bottom_line)
    每条line为两个点组成的元组 ((x1,y1),(x2,y2))，已延伸至图像边界。
    """
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # 绘制每个椎体的终板线（已延伸至边界）
    for name, (top_line, bottom_line) in vertebra_lines.items():
        color = VERTEBRA_COLORS[name]

        # 绘制终板线（红色实线）
        plt.plot([top_line[0][0], top_line[1][0]], [top_line[0][1], top_line[1][1]],
                 color='red', linewidth=2, zorder=3)
        plt.plot([bottom_line[0][0], bottom_line[1][0]], [bottom_line[0][1], bottom_line[1][1]],
                 color='red', linewidth=2, zorder=3)

        # 标注椎体名称
        mid_x = (top_line[0][0] + top_line[1][0] + bottom_line[0][0] + bottom_line[1][0]) / 4
        mid_y = (top_line[0][1] + top_line[1][1] + bottom_line[0][1] + bottom_line[1][1]) / 4
        plt.text(mid_x, mid_y, name, color=color, fontsize=12, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # 绘制每对Cobb角弧线（L1-L2, L2-L3, ...）
    for i in range(1, 5):
        upper = vertebra_lines[f"L{i}"][1]   # 上椎体下终板
        lower = vertebra_lines[f"L{i+1}"][0] # 下椎体上终板
        angle = calculate_cobb_angle(upper, lower)
        mid1 = ((upper[0][0] + upper[1][0]) / 2, (upper[0][1] + upper[1][1]) / 2)
        mid2 = ((lower[0][0] + lower[1][0]) / 2, (lower[0][1] + lower[1][1]) / 2)
        center = ((mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2)
        plt.text(center[0], center[1], f"L{i}-L{i+1} Cobb:{angle:.1f}°",
                 color='yellow', fontsize=10, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))

    # 计算L1-L5总体Cobb角（L1上终板与L5下终板）
    L1_top = vertebra_lines["L1"][0]   # L1上终板
    L5_bottom = vertebra_lines["L5"][1] # L5下终板
    total_angle = calculate_cobb_angle(L1_top, L5_bottom)
    plt.text(20, 30, f"L1-L5 Cobb Angle: {total_angle:.2f}°",
             color='white', fontsize=14, weight='bold',
             bbox=dict(facecolor='black', alpha=0.8, pad=3))

    plt.title("Cobb Angle Measurement based on Contour Fitting", fontsize=16)
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
    img_shape = original_img.shape[:2]  # (h, w)

    vertebra_lines = {}  # 存储每个椎体的 (top_line, bottom_line) 已延伸至边界
    for label, name in VERTEBRA_LABELS.items():
        print(f"Processing {name}...")
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        top_line, bottom_line = detect_endplate_lines_from_mask(vertebra_mask, name, img_shape)
        if top_line is None or bottom_line is None:
            print(f"Error: {name} endplate detection failed, skip this image.")
            return None
        vertebra_lines[name] = (top_line, bottom_line)

    # 计算各节段Cobb角
    segment_angles = {}
    for i in range(1, 5):
        upper = vertebra_lines[f"L{i}"][1]      # L_i下终板
        lower = vertebra_lines[f"L{i+1}"][0]    # L_{i+1}上终板
        angle = calculate_cobb_angle(upper, lower)
        segment_angles[f"L{i}_L{i+1}"] = angle

    # 计算总体L1-L5 Cobb角（L1上终板与L5下终板）
    L1_top = vertebra_lines["L1"][0]
    L5_bottom = vertebra_lines["L5"][1]
    total_angle = calculate_cobb_angle(L1_top, L5_bottom)

    # 保存可视化结果
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_cobb_contour.png"
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
        **segment_angles   # 解包节段角度
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
                # 确保字段顺序与fieldnames一致
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
