import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# 路径设置
img_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\img"
mask_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\mask"
save_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\output4_CSF"

os.makedirs(save_dir, exist_ok=True)

# CSF标签值 - 根据您的数据集调整
CSF_LABEL = 12

# 支持的原图格式
SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


def find_image_file(base_name, img_dir):
    """查找对应base_name的各种格式的原图文件"""
    for ext in SUPPORTED_IMAGE_EXTS:
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


# 创建数据列表来存储结果
results = []

# 处理所有文件
for file in tqdm(os.listdir(mask_dir)):
    if not file.endswith('.png'):
        continue

    mask_path = os.path.join(mask_dir, file)
    base = os.path.splitext(file)[0]

    # 使用新的查找函数
    img_path = find_image_file(base, img_dir)

    if not img_path:
        print(f"[警告] 原图不存在，跳过：{file} (在 {img_dir} 中找不到 {base}.*)")
        continue

    print(f"[信息] 处理文件：{file} -> 原图：{os.path.basename(img_path)}")

    # 读取图像
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[错误] 无法读取原图：{img_path}")
        continue

    if mask is None:
        print(f"[错误] 无法读取mask：{mask_path}")
        continue

    # 获取图像尺寸
    height, width = mask.shape[:2]
    total_pixels = height * width

    # 生成 CSF 区域
    csf_region = np.where(mask == CSF_LABEL, 255, 0).astype(np.uint8)
    csf_area = np.sum(csf_region == 255)

    # 收集结果
    if csf_area == 0:
        print(f"[信息] {file} 中没有检测到CSF区域")
        results.append({
            'Filename': file,
            'CSF_Pixel_Count': 0,
            'CSF_Percentage': 0.0,
            'CSF_Min_Intensity': 0,
            'CSF_Max_Intensity': 0,
            'CSF_Mean_Intensity': 0.0,
            'CSF_Std_Intensity': 0.0,
            'Contour_Count': 0,
            'Largest_Contour_Area': 0,
            'Bounding_Box': 'None',
            'Bounding_Box_Width': 0,
            'Bounding_Box_Height': 0,
            'Aspect_Ratio': 0.0,
            'Centroid_X': 0.0,
            'Centroid_Y': 0.0
        })
    else:
        # 计算CSF区域在灰度图像上的强度统计
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        csf_intensities = gray_img[csf_region == 255]

        # 强度统计
        csf_min_intensity = np.min(csf_intensities) if len(csf_intensities) > 0 else 0
        csf_max_intensity = np.max(csf_intensities) if len(csf_intensities) > 0 else 0
        csf_mean_intensity = np.mean(csf_intensities) if len(csf_intensities) > 0 else 0.0
        csf_std_intensity = np.std(csf_intensities) if len(csf_intensities) > 0 else 0.0

        # CSF区域占比
        csf_percentage = (csf_area / total_pixels) * 100

        # 查找轮廓
        contours, _ = cv2.findContours(csf_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        # 计算最大轮廓面积
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_area = cv2.contourArea(largest_contour)

            # 获取边界框
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_box_str = f'{x},{y},{w},{h}'

            # 计算宽高比（防止除零）
            aspect_ratio = w / h if h > 0 else 0.0

            # 计算CSF区域的质心（整个CSF区域，不只是最大轮廓）
            # 使用图像矩计算质心
            M = cv2.moments(csf_region)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"]
                centroid_y = M["m01"] / M["m00"]
            else:
                centroid_x, centroid_y = 0.0, 0.0

            # 在原图上绘制矩形框
            result_img = img.copy()
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # ============ 计算 CSF 区域信号值分布 ============
            csf_crop_mask = csf_region[y:y + h, x:x + w]
            gray_crop = gray_img[y:y + h, x:x + w]

            signal_distribution = []
            for row in range(h):
                row_pixels = gray_crop[row, :][csf_crop_mask[row, :] == 255]
                if len(row_pixels) > 0:
                    signal_distribution.append(np.mean(row_pixels))
                else:
                    signal_distribution.append(0)

            signal_distribution = np.array(signal_distribution)

            # ========== 平滑方式一：高斯平滑 ==========
            if len(signal_distribution) > 0:
                # 确保信号分布长度足够进行平滑
                if len(signal_distribution) > 9:
                    gaussian_smoothed = cv2.GaussianBlur(signal_distribution.reshape(-1, 1), (9, 9), sigmaX=3).flatten()
                else:
                    # 如果信号太短，使用较小的核
                    kernel_size = min(len(signal_distribution), 5)
                    if kernel_size % 2 == 0:  # 确保核大小为奇数
                        kernel_size -= 1
                    if kernel_size >= 3:
                        gaussian_smoothed = cv2.GaussianBlur(signal_distribution.reshape(-1, 1),
                                                             (kernel_size, kernel_size), sigmaX=1).flatten()
                    else:
                        gaussian_smoothed = signal_distribution

                # ========== 平滑方式二：Savitzky-Golay 滤波 ==========
                from scipy.signal import savgol_filter

                # window_length 必须为奇数且小于等于信号长度
                if len(signal_distribution) >= 5:
                    window_length = min(len(signal_distribution) // 5 * 2 + 1, len(signal_distribution))
                    if window_length < 5:
                        window_length = 5
                    if window_length > len(signal_distribution):
                        window_length = len(signal_distribution) - 1 if len(signal_distribution) % 2 == 0 else len(
                            signal_distribution)

                    if window_length >= 5 and window_length <= len(signal_distribution):
                        try:
                            sg_smoothed = savgol_filter(signal_distribution, window_length=window_length,
                                                        polyorder=min(3, window_length - 1))
                        except:
                            sg_smoothed = signal_distribution
                    else:
                        sg_smoothed = signal_distribution
                else:
                    sg_smoothed = signal_distribution

                # ========== 绘制三条曲线对比 ==========
                plt.figure(figsize=(7, 5))
                plt.plot(signal_distribution, label="Raw Signal", linewidth=1)
                plt.plot(gaussian_smoothed, label="Gaussian Smoothed", linewidth=2)
                plt.plot(sg_smoothed, label="Savitzky-Golay Smoothed", linewidth=2)
                plt.xlabel("Position along CSF length (pixels)")
                plt.ylabel("Average Signal Intensity")
                plt.title(f"CSF Signal Profile - {base}")
                plt.legend()

                curve_path = os.path.join(save_dir, base + "_CSF_signal_curve_compare.png")
                plt.savefig(curve_path, dpi=300)
                plt.close()

            # 保存结果图像
            save_path = os.path.join(save_dir, file)
            cv2.imwrite(save_path, result_img)

            results.append({
                'Filename': file,
                'CSF_Pixel_Count': csf_area,
                'CSF_Percentage': round(csf_percentage, 4),
                'CSF_Min_Intensity': int(csf_min_intensity),
                'CSF_Max_Intensity': int(csf_max_intensity),
                'CSF_Mean_Intensity': round(csf_mean_intensity, 2),
                'CSF_Std_Intensity': round(csf_std_intensity, 2),
                'Contour_Count': contour_count,
                'Largest_Contour_Area': round(largest_contour_area, 2),
                'Bounding_Box': bounding_box_str,
                'Bounding_Box_Width': w,
                'Bounding_Box_Height': h,
                'Aspect_Ratio': round(aspect_ratio, 3),
                'Centroid_X': round(centroid_x, 2),
                'Centroid_Y': round(centroid_y, 2)
            })
            print(f"[信息] {file} 处理完成，CSF像素数：{csf_area}")
        else:
            # 有CSF像素但没有检测到轮廓的情况
            results.append({
                'Filename': file,
                'CSF_Pixel_Count': csf_area,
                'CSF_Percentage': round(csf_percentage, 4),
                'CSF_Min_Intensity': int(csf_min_intensity),
                'CSF_Max_Intensity': int(csf_max_intensity),
                'CSF_Mean_Intensity': round(csf_mean_intensity, 2),
                'CSF_Std_Intensity': round(csf_std_intensity, 2),
                'Contour_Count': 0,
                'Largest_Contour_Area': 0,
                'Bounding_Box': 'None',
                'Bounding_Box_Width': 0,
                'Bounding_Box_Height': 0,
                'Aspect_Ratio': 0.0,
                'Centroid_X': 0.0,
                'Centroid_Y': 0.0
            })

# 将结果保存为CSV文件
csv_file = os.path.join(save_dir, "CSF_highlight_statistics.csv")
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

# 同时保存为TXT文件（可选）
txt_file = os.path.join(save_dir, "CSF_Area_Statistics.txt")
with open(txt_file, "w", encoding="utf-8") as f:
    # 写入标题行
    headers = ['Filename', 'CSF_Pixel_Count', 'CSF_Percentage', 'CSF_Min_Intensity',
               'CSF_Max_Intensity', 'CSF_Mean_Intensity', 'CSF_Std_Intensity',
               'Contour_Count', 'Largest_Contour_Area', 'Bounding_Box',
               'Bounding_Box_Width', 'Bounding_Box_Height', 'Aspect_Ratio',
               'Centroid_X', 'Centroid_Y']
    f.write("\t".join(headers) + "\n")

    # 写入数据行
    for result in results:
        row = [
            result['Filename'],
            str(result['CSF_Pixel_Count']),
            f"{result['CSF_Percentage']:.4f}",
            str(result['CSF_Min_Intensity']),
            str(result['CSF_Max_Intensity']),
            f"{result['CSF_Mean_Intensity']:.2f}",
            f"{result['CSF_Std_Intensity']:.2f}",
            str(result['Contour_Count']),
            f"{result['Largest_Contour_Area']:.2f}",
            result['Bounding_Box'],
            str(result['Bounding_Box_Width']),
            str(result['Bounding_Box_Height']),
            f"{result['Aspect_Ratio']:.3f}",
            f"{result['Centroid_X']:.2f}",
            f"{result['Centroid_Y']:.2f}"
        ]
        f.write("\t".join(row) + "\n")

print("✅ CSF 区域高亮与面积统计完成！")
print("✅ CSF 信号分布曲线绘制完成！")
print(f"📄 CSV统计文件：{csv_file}")
print(f"📄 TXT统计文件：{txt_file}")
print(f"🖼️ 输出图保存目录：{save_dir}")