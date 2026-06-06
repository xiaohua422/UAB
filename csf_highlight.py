import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Path Configuration
img_dir = "img"
mask_dir = "mask"
save_dir = "output4_CSF"

os.makedirs(save_dir, exist_ok=True)

# CSF Label Value - Adjust according to your dataset
CSF_LABEL = 12

# Supported original image formats
SUPPORTED_IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']


def find_image_file(base_name, img_dir):
    """Find image file with various formats for the given base name"""
    for ext in SUPPORTED_IMAGE_EXTS:
        img_path = os.path.join(img_dir, base_name + ext)
        if os.path.exists(img_path):
            return img_path
    return None


# Create list to store results
results = []

# Process all files
for file in tqdm(os.listdir(mask_dir)):
    if not file.endswith('.png'):
        continue

    mask_path = os.path.join(mask_dir, file)
    base = os.path.splitext(file)[0]

    # Use the new find function
    img_path = find_image_file(base, img_dir)

    if not img_path:
        print(f"[Warning] Original image not found, skipping: {file} (could not find {base}.* in {img_dir})")
        continue

    print(f"[Info] Processing: {file} -> Original image: {os.path.basename(img_path)}")

    # Read images
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"[Error] Failed to read original image: {img_path}")
        continue

    if mask is None:
        print(f"[Error] Failed to read mask: {mask_path}")
        continue

    # Get image dimensions
    height, width = mask.shape[:2]
    total_pixels = height * width

    # Generate CSF region
    csf_region = np.where(mask == CSF_LABEL, 255, 0).astype(np.uint8)
    csf_area = np.sum(csf_region == 255)

    # Collect results
    if csf_area == 0:
        print(f"[Info] No CSF region detected in {file}")
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
        # Calculate intensity statistics on grayscale image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        csf_intensities = gray_img[csf_region == 255]

        # Intensity statistics
        csf_min_intensity = np.min(csf_intensities) if len(csf_intensities) > 0 else 0
        csf_max_intensity = np.max(csf_intensities) if len(csf_intensities) > 0 else 0
        csf_mean_intensity = np.mean(csf_intensities) if len(csf_intensities) > 0 else 0.0
        csf_std_intensity = np.std(csf_intensities) if len(csf_intensities) > 0 else 0.0

        # CSF area percentage
        csf_percentage = (csf_area / total_pixels) * 100

        # Find contours
        contours, _ = cv2.findContours(csf_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)

        # Calculate largest contour area
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_area = cv2.contourArea(largest_contour)

            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            bounding_box_str = f'{x},{y},{w},{h}'

            # Calculate aspect ratio (prevent division by zero)
            aspect_ratio = w / h if h > 0 else 0.0

            # Calculate centroid of CSF region using image moments
            M = cv2.moments(csf_region)
            if M["m00"] != 0:
                centroid_x = M["m10"] / M["m00"]
                centroid_y = M["m01"] / M["m00"]
            else:
                centroid_x, centroid_y = 0.0, 0.0

            # Draw rectangle on original image
            result_img = img.copy()
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # ============ Calculate CSF signal distribution ============
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

            # ========== Smoothing Method 1: Gaussian Smoothing ==========
            if len(signal_distribution) > 0:
                # Ensure signal length is sufficient for smoothing
                if len(signal_distribution) > 9:
                    gaussian_smoothed = cv2.GaussianBlur(signal_distribution.reshape(-1, 1), (9, 9), sigmaX=3).flatten()
                else:
                    # Use smaller kernel if signal is too short
                    kernel_size = min(len(signal_distribution), 5)
                    if kernel_size % 2 == 0:
                        kernel_size -= 1
                    if kernel_size >= 3:
                        gaussian_smoothed = cv2.GaussianBlur(signal_distribution.reshape(-1, 1),
                                                             (kernel_size, kernel_size), sigmaX=1).flatten()
                    else:
                        gaussian_smoothed = signal_distribution

                # ========== Smoothing Method 2: Savitzky-Golay Filter ==========
                from scipy.signal import savgol_filter

                # window_length must be odd and <= signal length
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

                # ========== Plot three curves for comparison ==========
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

            # Save result image
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
            print(f"[Info] Completed processing {file}, CSF pixels: {csf_area}")
        else:
            # Case with CSF pixels but no contours detected
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

# Save results to CSV file
csv_file = os.path.join(save_dir, "CSF_highlight_statistics.csv")
df = pd.DataFrame(results)
df.to_csv(csv_file, index=False)

# Save as TXT file (optional)
txt_file = os.path.join(save_dir, "CSF_Area_Statistics.txt")
with open(txt_file, "w", encoding="utf-8") as f:
    # Write header line
    headers = ['Filename', 'CSF_Pixel_Count', 'CSF_Percentage', 'CSF_Min_Intensity',
               'CSF_Max_Intensity', 'CSF_Mean_Intensity', 'CSF_Std_Intensity',
               'Contour_Count', 'Largest_Contour_Area', 'Bounding_Box',
               'Bounding_Box_Width', 'Bounding_Box_Height', 'Aspect_Ratio',
               'Centroid_X', 'Centroid_Y']
    f.write("\t".join(headers) + "\n")

    # Write data rows
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

print("✅ CSF region highlighting and area statistics completed!")
print("✅ CSF signal distribution curves generated successfully!")
print(f"📄 CSV statistics file: {csv_file}")
print(f"📄 TXT statistics file: {txt_file}")
print(f"🖼️  Output directory: {save_dir}")
