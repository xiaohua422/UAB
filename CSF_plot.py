import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 路径设置
img_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\img"
mask_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\mask"
save_dir = r"D:\unet_test\Deeplabv3+\deeplabv3-plus-pytorch-main-08\output3_CSF"
csv_file = os.path.join(save_dir, "CSF_highlight_statistics.csv")

# 读取CSV数据
df = pd.read_csv(csv_file)

# 只选择有CSF区域的图像
csf_images = df[df['CSF_Pixel_Count'] > 0]

if len(csf_images) == 0:
    print("没有找到包含CSF区域的图像")
    exit()

# 选择第一个有CSF的图像进行分析
sample_file = csf_images.iloc[0]['Filename']
print(f"分析图像: {sample_file}")

# 读取原图和掩码
base_name = os.path.splitext(sample_file)[0]
img_path = os.path.join(img_dir, base_name + ".jpg")
mask_path = os.path.join(mask_dir, sample_file)

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# CSF标签值
CSF_LABEL = 12

# 生成CSF区域
csf_region = np.where(mask == CSF_LABEL, 255, 0).astype(np.uint8)

# 获取CSF区域的轮廓
contours, _ = cv2.findContours(csf_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("未找到CSF轮廓")
    exit()

# 选择最大的轮廓
main_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(main_contour)

# 提取CSF区域沿着脊椎走向的强度分布
# 假设脊椎走向是垂直方向，我们沿着y轴分析每一行的CSF像素数量
csf_intensity = []
for row in range(y, y + h):
    row_pixels = csf_region[row, x:x+w]
    csf_count = np.sum(row_pixels == 255)
    csf_intensity.append(csf_count)

# 创建可视化
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 原图与CSF边界框
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
ax1.set_title(f'原图与CSF区域\n{base_name}')
ax1.axis('off')

# 子图2: CSF区域掩码
ax2.imshow(csf_region, cmap='gray')
ax2.set_title('CSF区域掩码')
ax2.axis('off')

# 子图3: CSF强度分布曲线
y_positions = np.arange(len(csf_intensity))
ax3.plot(csf_intensity, y_positions, 'b-', linewidth=2)
ax3.set_xlabel('CSF像素数量')
ax3.set_ylabel('脊椎位置 (像素行)')
ax3.set_title('CSF区域沿着脊椎走向的强度分布')
ax3.grid(True, alpha=0.3)

# 反转y轴，使顶部对应图像顶部
ax3.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{base_name}_CSF_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"分析完成! 结果已保存为: {base_name}_CSF_analysis.png")