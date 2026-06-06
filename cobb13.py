import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import csv

# Set font: Times New Roman for English, fallback to Arial
plt.rcParams['font.family'] = ['Times New Roman', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- Configuration Parameters --------------------------
ORIGINAL_IMG_DIR = "img/"
MASK_DIR = "mask/"
OUTPUT_PARENT_DIR = "cobb13_amace_cobb_results"
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

# -------------------------- Line Detection Functions (based on contour fitting) --------------------------
def fit_line_from_points(points):
    """Least squares line fitting from points; returns (slope, intercept) or vertical line indicator"""
    if len(points) < 2:
        return None, None
    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    # If x coordinates vary very little, treat as vertical line
    if np.max(x) - np.min(x) < 1e-6:
        return None, x[0]  # return vertical line x-coordinate
    # Least squares fit y = kx + b
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return k, b

def line_to_two_points(k, b, is_vertical, x_fixed, img_shape):
    """Convert line parameters to two points at image boundaries"""
    h, w = img_shape[:2]
    if is_vertical:
        x = int(x_fixed)
        return (x, 0), (x, h-1)
    else:
        # Compute intersections with left and right borders
        y_left = k * 0 + b
        y_right = k * (w-1) + b
        # Clip to image height range
        p_left = (0, int(np.clip(y_left, 0, h-1)))
        p_right = (w-1, int(np.clip(y_right, 0, h-1)))
        return p_left, p_right

def detect_endplate_lines_from_mask(vertebra_mask, label_name, img_shape):
    """
    Extract top and bottom endplate lines from a vertebra mask.
    Returns (top_line, bottom_line), each line given as two endpoints (intersections with image border).
    """
    mask = vertebra_mask.copy().astype(np.uint8)
    if np.sum(mask) == 0:
        print(f"Warning: {label_name} mask is empty")
        return None, None

    # Get contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print(f"Warning: No contour found for {label_name}")
        return None, None

    # Take the largest contour (should be the outer boundary of the vertebra)
    contour = max(contours, key=cv2.contourArea)
    contour_points = contour.squeeze().reshape(-1, 2)  # (N, 2)

    # Separate top and bottom edge points by y-coordinate
    ys = contour_points[:, 1]
    y_min, y_max = np.min(ys), np.max(ys)
    center_y = (y_min + y_max) / 2

    # Upper half points (y < center_y)
    top_mask = contour_points[:, 1] < center_y
    top_points = contour_points[top_mask]
    # Lower half points (y > center_y)
    bottom_mask = contour_points[:, 1] > center_y
    bottom_points = contour_points[bottom_mask]

    # If too few points in a part, fallback to bounding box edge
    if len(top_points) < 5:
        print(f"Warning: {label_name} top edge points insufficient, using bounding box top edge")
        xs = contour_points[:, 0]
        top_line = [(np.min(xs), y_min), (np.max(xs), y_min)]
    else:
        # Fit top edge line
        k, b = fit_line_from_points(top_points)
        if k is None:
            # Vertical line case
            top_line = line_to_two_points(None, None, True, b, img_shape)
        else:
            top_line = line_to_two_points(k, b, False, None, img_shape)

    if len(bottom_points) < 5:
        print(f"Warning: {label_name} bottom edge points insufficient, using bounding box bottom edge")
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
    """Extend line segment to image borders (kept for compatibility with original visualization code)"""
    # Since line_to_two_points already returns border points, this function can just return input
    return p1, p2

# -------------------------- Cobb Angle Calculation --------------------------
def calculate_cobb_angle(line1, line2):
    """Calculate the acute angle between two lines (Cobb angle)"""
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

# -------------------------- Visualization --------------------------
def visualize_endplates(original_img, mask, vertebra_lines, output_path):
    """
    vertebra_lines: dict, key = vertebra name, value = (top_line, bottom_line)
    Each line is a tuple of two points ((x1,y1),(x2,y2)), already extended to image borders.
    """
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # Draw endplate lines for each vertebra (already extended to borders)
    for name, (top_line, bottom_line) in vertebra_lines.items():
        color = VERTEBRA_COLORS[name]

        # Draw endplate lines (solid red)
        plt.plot([top_line[0][0], top_line[1][0]], [top_line[0][1], top_line[1][1]],
                 color='red', linewidth=2, zorder=3)
        plt.plot([bottom_line[0][0], bottom_line[1][0]], [bottom_line[0][1], bottom_line[1][1]],
                 color='red', linewidth=2, zorder=3)

        # Label vertebra name
        mid_x = (top_line[0][0] + top_line[1][0] + bottom_line[0][0] + bottom_line[1][0]) / 4
        mid_y = (top_line[0][1] + top_line[1][1] + bottom_line[0][1] + bottom_line[1][1]) / 4
        plt.text(mid_x, mid_y, name, color=color, fontsize=12, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Draw Cobb angle arcs for each adjacent pair (L1-L2, L2-L3, ...)
    for i in range(1, 5):
        upper = vertebra_lines[f"L{i}"][1]   # lower endplate of upper vertebra
        lower = vertebra_lines[f"L{i+1}"][0] # upper endplate of lower vertebra
        angle = calculate_cobb_angle(upper, lower)
        mid1 = ((upper[0][0] + upper[1][0]) / 2, (upper[0][1] + upper[1][1]) / 2)
        mid2 = ((lower[0][0] + lower[1][0]) / 2, (lower[0][1] + lower[1][1]) / 2)
        center = ((mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2)
        plt.text(center[0], center[1], f"L{i}-L{i+1} Cobb:{angle:.1f}°",
                 color='yellow', fontsize=10, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))

    # Calculate overall L1-L5 Cobb angle (L1 upper endplate vs L5 lower endplate)
    L1_top = vertebra_lines["L1"][0]   # L1 upper endplate
    L5_bottom = vertebra_lines["L5"][1] # L5 lower endplate
    total_angle = calculate_cobb_angle(L1_top, L5_bottom)
    plt.text(20, 30, f"L1-L5 Cobb Angle: {total_angle:.2f}°",
             color='white', fontsize=14, weight='bold',
             bbox=dict(facecolor='black', alpha=0.8, pad=3))

    plt.title("Cobb Angle Measurement based on Contour Fitting", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Cobb angle result saved to: {output_path}")

# -------------------------- Batch Processing (returns angle data) --------------------------
def process_single_image(original_img_path, mask_path, output_dir):
    """
    Process a single image; returns a dictionary with angle information, or None on failure.
    """
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Error: cannot read original image: {original_img_path}")
        return None

    mask = Image.open(mask_path)
    mask_array = np.array(mask, dtype=np.int32)
    img_shape = original_img.shape[:2]  # (h, w)

    vertebra_lines = {}  # store (top_line, bottom_line) for each vertebra, extended to borders
    for label, name in VERTEBRA_LABELS.items():
        print(f"Processing {name}...")
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        top_line, bottom_line = detect_endplate_lines_from_mask(vertebra_mask, name, img_shape)
        if top_line is None or bottom_line is None:
            print(f"Error: {name} endplate detection failed, skipping this image.")
            return None
        vertebra_lines[name] = (top_line, bottom_line)

    # Calculate segment Cobb angles
    segment_angles = {}
    for i in range(1, 5):
        upper = vertebra_lines[f"L{i}"][1]      # lower endplate of L_i
        lower = vertebra_lines[f"L{i+1}"][0]    # upper endplate of L_{i+1}
        angle = calculate_cobb_angle(upper, lower)
        segment_angles[f"L{i}_L{i+1}"] = angle

    # Calculate overall L1-L5 Cobb angle (L1 upper endplate vs L5 lower endplate)
    L1_top = vertebra_lines["L1"][0]
    L5_bottom = vertebra_lines["L5"][1]
    total_angle = calculate_cobb_angle(L1_top, L5_bottom)

    # Save visualization result
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_cobb_contour.png"
    output_path = os.path.join(output_dir, output_filename)
    visualize_endplates(original_img, mask_array, vertebra_lines, output_path)

    # Save line coordinate information
    coords_path = os.path.join(output_dir, output_filename.replace(".png", "_lines.txt"))
    with open(coords_path, "w", encoding="utf-8") as f:
        for name, (top, bottom) in vertebra_lines.items():
            f.write(f"====={name} Endplate Lines=====\n")
            f.write(f"Upper endplate: {top[0]} -> {top[1]}\n")
            f.write(f"Lower endplate: {bottom[0]} -> {bottom[1]}\n\n")
    print(f"Line coordinates saved to: {coords_path}")

    # Return angle data
    result = {
        'image': os.path.splitext(os.path.basename(original_img_path))[0],
        'L1_L5_total': total_angle,
        **segment_angles   # unpack segment angles
    }
    return result

def main():
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    original_images = [f for f in os.listdir(ORIGINAL_IMG_DIR) if any(f.lower().endswith(ext) for ext in img_extensions)]
    if not original_images:
        print(f"No images found in {ORIGINAL_IMG_DIR}")
        return

    all_results = []  # collect angle data from all successfully processed images

    for img_filename in original_images:
        print(f"\nProcessing image: {img_filename}")
        original_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)
        img_name_without_ext = os.path.splitext(img_filename)[0]
        mask_filename = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_mask_path = os.path.join(MASK_DIR, img_name_without_ext + ext)
            if os.path.exists(potential_mask_path):
                mask_filename = img_name_without_ext + ext
                break
        if mask_filename is None:
            print(f"Warning: mask file for {img_filename} not found, skipping.")
            continue
        mask_path = os.path.join(MASK_DIR, mask_filename)
        image_output_dir = os.path.join(OUTPUT_PARENT_DIR, img_name_without_ext)
        os.makedirs(image_output_dir, exist_ok=True)

        result = process_single_image(original_img_path, mask_path, image_output_dir)
        if result is not None:
            all_results.append(result)

    # Generate summary table
    if all_results:
        summary_path = os.path.join(OUTPUT_PARENT_DIR, "cobb_summary.csv")
        fieldnames = ['image', 'L1_L5_total', 'L1_L2', 'L2_L3', 'L3_L4', 'L4_L5']
        with open(summary_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for res in all_results:
                # Ensure field order matches fieldnames
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

    print(f"\nBatch processing completed! Results saved in: {OUTPUT_PARENT_DIR}")

if __name__ == "__main__":
    main()
