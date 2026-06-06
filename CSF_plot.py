import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Set font for English (use default sans-serif)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Path settings
img_dir = "img"
mask_dir = "mask"
save_dir = "output3_CSF"
csv_file = os.path.join(save_dir, "CSF_highlight_statistics.csv")

# Read CSV data
df = pd.read_csv(csv_file)

# Select only images with CSF regions
csf_images = df[df['CSF_Pixel_Count'] > 0]

if len(csf_images) == 0:
    print("No images with CSF regions found")
    exit()

# Select the first image with CSF for analysis
sample_file = csf_images.iloc[0]['Filename']
print(f"Analyzing image: {sample_file}")

# Read original image and mask
base_name = os.path.splitext(sample_file)[0]
img_path = os.path.join(img_dir, base_name + ".jpg")
mask_path = os.path.join(mask_dir, sample_file)

img = cv2.imread(img_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# CSF label value
CSF_LABEL = 12

# Generate CSF region
csf_region = np.where(mask == CSF_LABEL, 255, 0).astype(np.uint8)

# Find contours of the CSF region
contours, _ = cv2.findContours(csf_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not contours:
    print("No CSF contour found")
    exit()

# Select the largest contour
main_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(main_contour)

# Extract CSF intensity distribution along the spine direction
# Assuming the spine runs vertically, analyze the number of CSF pixels per row
csf_intensity = []
for row in range(y, y + h):
    row_pixels = csf_region[row, x:x+w]
    csf_count = np.sum(row_pixels == 255)
    csf_intensity.append(csf_count)

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Subplot 1: Original image with CSF bounding box
ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax1.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
ax1.set_title(f'Original Image with CSF Region\n{base_name}')
ax1.axis('off')

# Subplot 2: CSF region mask
ax2.imshow(csf_region, cmap='gray')
ax2.set_title('CSF Region Mask')
ax2.axis('off')

# Subplot 3: CSF intensity distribution curve
y_positions = np.arange(len(csf_intensity))
ax3.plot(csf_intensity, y_positions, 'b-', linewidth=2)
ax3.set_xlabel('Number of CSF Pixels')
ax3.set_ylabel('Spine Position (pixel row)')
ax3.set_title('CSF Intensity Distribution Along Spine')
ax3.grid(True, alpha=0.3)

# Invert y-axis so top corresponds to top of image
ax3.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'{base_name}_CSF_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print(f"Analysis completed! Result saved as: {base_name}_CSF_analysis.png")
