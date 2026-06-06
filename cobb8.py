import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math

# Set font for English (use default sans-serif)
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- Configuration Parameters --------------------------
# Input paths (directory paths)
ORIGINAL_IMG_DIR = "img/"
MASK_DIR = "mask/"

# Output parent directory
OUTPUT_PARENT_DIR = "cobb8_amace_cobb_results"
os.makedirs(OUTPUT_PARENT_DIR, exist_ok=True)

# Vertebra label to name mapping
VERTEBRA_LABELS = {
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "L5",
    11: "S1"
}

# Visualization colors for different vertebrae
VERTEBRA_COLORS = {
    "L1": "red",
    "L2": "green",
    "L3": "blue",
    "L4": "yellow",
    "L5": "magenta",
    "S1": "cyan"
}


# -------------------------- Core Corner Detection Functions --------------------------
def fallback_HV_calculation(src):
    """Bounding box method to get corners (fallback)"""
    points = np.argwhere(src > 0)
    if len(points) == 0:
        return np.array([[0, 0], [0, 0], [0, 0], [0, 0]])  # empty corners

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)

    # Return corners in order: TL, TR, BL, BR
    return np.array([
        [x_min, y_min],  # TL
        [x_max, y_min],  # TR
        [x_min, y_max],  # BL
        [x_max, y_max]   # BR
    ], dtype=np.int32)


def detect_vertebra_corners(vertebra_mask, is_S1=False):
    """Detect 4 corners (TL, TR, BL, BR) of a single vertebra"""
    src = vertebra_mask.copy().astype(np.uint8)

    # Check mask validity
    if np.sum(src) == 0:
        print("Warning: Vertebra mask is empty, using bounding box corners")
        return fallback_HV_calculation(src)

    # Convert to format suitable for corner detection
    gray = np.float32(src)

    # Shi-Tomasi corner detection parameters (adjust for S1)
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

    # If detection fails, use bounding box corners
    if corners is None or len(corners) < 4:
        print(f"Warning: Detected fewer than 4 corners, using bounding box corners")
        return fallback_HV_calculation(src)

    # Process corner format
    corners = np.int32(corners.squeeze())

    # Ensure exactly 4 corners
    if len(corners) > 4:
        corners = corners[:4]
    elif len(corners) < 4:
        return fallback_HV_calculation(src)

    # Sort corners: TL (top-left), TR (top-right), BL (bottom-left), BR (bottom-right)
    # 1. Separate by y-coordinate (smaller y = top, larger y = bottom)
    corners_sorted_by_y = corners[np.argsort(corners[:, 1])]
    top_corners = corners_sorted_by_y[:2]    # top two points
    bottom_corners = corners_sorted_by_y[2:] # bottom two points

    # 2. Separate by x-coordinate (smaller x = left, larger x = right)
    top_corners_sorted = top_corners[np.argsort(top_corners[:, 0])]
    bottom_corners_sorted = bottom_corners[np.argsort(bottom_corners[:, 0])]

    # Combine into final order
    return np.array([
        top_corners_sorted[0],  # TL
        top_corners_sorted[1],  # TR
        bottom_corners_sorted[0],  # BL
        bottom_corners_sorted[1]   # BR
    ], dtype=np.int32)


def extend_line_to_border(p1, p2, img_shape):
    """
    Extend a line segment to the image borders without exceeding.
    Returns the two extended endpoints.
    """
    h, w = img_shape[:2]
    x1, y1 = p1
    x2, y2 = p2

    # Compute line parameters
    if x2 == x1:  # vertical line
        # Extend to top and bottom borders
        return (x1, 0), (x1, h - 1)

    # Compute slope
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    # Compute intersections with image borders
    intersections = []

    # Intersection with left border (x=0)
    y_left = m * 0 + b
    if 0 <= y_left <= h - 1:
        intersections.append((0, y_left))

    # Intersection with right border (x=w-1)
    y_right = m * (w - 1) + b
    if 0 <= y_right <= h - 1:
        intersections.append((w - 1, y_right))

    # Intersection with top border (y=0)
    if m != 0:
        x_top = (0 - b) / m
        if 0 <= x_top <= w - 1:
            intersections.append((x_top, 0))

    # Intersection with bottom border (y=h-1)
    if m != 0:
        x_bottom = (h - 1 - b) / m
        if 0 <= x_bottom <= w - 1:
            intersections.append((x_bottom, h - 1))

    # If at least two intersections found, take the farthest two
    if len(intersections) >= 2:
        # Compute distance from original line segment
        distances = [np.linalg.norm(np.array(p) - np.array(p1)) + np.linalg.norm(np.array(p) - np.array(p2))
                     for p in intersections]
        # Take the two points with largest distances
        sorted_indices = np.argsort(distances)[-2:]
        return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
    else:
        # If not enough intersections, return original points
        return p1, p2


def calculate_perpendicular_line(p1, p2, img_shape):
    """
    Compute the perpendicular line through the midpoint of a segment and extend it to image borders.
    Returns the two endpoints of the perpendicular line and the midpoint.
    """
    # Compute midpoint
    mid_x = (p1[0] + p2[0]) / 2
    mid_y = (p1[1] + p2[1]) / 2
    midpoint = (mid_x, mid_y)

    # Compute segment vector
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]

    # Compute perpendicular vector (rotate 90 degrees)
    if dx == 0:  # vertical segment
        # Original vertical => perpendicular is horizontal
        perp_p1 = (0, mid_y)
        perp_p2 = (img_shape[1] - 1, mid_y)
    elif dy == 0:  # horizontal segment
        # Original horizontal => perpendicular is vertical
        perp_p1 = (mid_x, 0)
        perp_p2 = (mid_x, img_shape[0] - 1)
    else:
        # Compute slope
        m = dy / dx
        # Perpendicular slope = negative reciprocal
        m_perp = -1 / m

        # Compute perpendicular line equation: y - mid_y = m_perp * (x - mid_x)
        # Find intersections with image borders
        perp_p1, perp_p2 = extend_line_to_border(
            (mid_x, mid_y),
            (mid_x + 1, mid_y + m_perp),  # second point defined by slope
            img_shape
        )

    return perp_p1, perp_p2, midpoint


# -------------------------- Visualization Functions --------------------------
def visualize_corners(original_img, mask, all_corners, output_path):
    """Draw all vertebra corners on the original image and save"""
    # Convert original image to RGB format
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    # Create figure
    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # Draw corners for each vertebra
    for vertebra_name, corners in all_corners.items():
        color = VERTEBRA_COLORS[vertebra_name]

        # Draw corner points
        for (x, y) in corners:
            plt.scatter(x, y, color=color, s=80, marker='o',
                        edgecolors='white', linewidth=2, zorder=5)

        # Label corners (TL, TR, BL, BR)
        corner_labels = [f"{vertebra_name}_TL", f"{vertebra_name}_TR",
                         f"{vertebra_name}_BL", f"{vertebra_name}_BR"]
        for i, (x, y) in enumerate(corners):
            plt.text(x + 5, y - 5, corner_labels[i],
                     color=color, fontsize=10, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.7, pad=1), zorder=6)

        # Draw vertebra contour (for better visualization)
        vertebra_mask = (mask == [k for k, v in VERTEBRA_LABELS.items() if v == vertebra_name][0]).astype(np.uint8)
        contours, _ = cv2.findContours(vertebra_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if len(cnt) > 3:
                plt.plot(cnt[:, 0, 0], cnt[:, 0, 1], color=color, linewidth=2, linestyle='--', zorder=4)

    # Draw L1 upper endplate line and L5 lower endplate line
    if 'L1' in all_corners:
        l1_corners = all_corners['L1']
        # L1 upper endplate line: connect TL and TR, extend to image borders
        l1_tl = l1_corners[0]  # TL
        l1_tr = l1_corners[1]  # TR

        # Extend segment to image borders
        l1_extended_p1, l1_extended_p2 = extend_line_to_border(l1_tl, l1_tr, img_rgb.shape[:2])

        # Draw L1 upper endplate line
        plt.plot([l1_extended_p1[0], l1_extended_p2[0]],
                 [l1_extended_p1[1], l1_extended_p2[1]],
                 color='red', linewidth=3, linestyle='-', label='L1 Upper Endplate', zorder=3)

        # Label L1 endplate line
        mid_x = (l1_extended_p1[0] + l1_extended_p2[0]) / 2
        mid_y = (l1_extended_p1[1] + l1_extended_p2[1]) / 2
        plt.text(mid_x, mid_y - 10, 'L1 Upper Endplate', color='red', fontsize=12,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=2))

    if 'L5' in all_corners:
        l5_corners = all_corners['L5']
        # L5 lower endplate line: connect BR and BL, extend to image borders
        l5_br = l5_corners[3]  # BR
        l5_bl = l5_corners[2]  # BL

        # Extend segment to image borders
        l5_extended_p1, l5_extended_p2 = extend_line_to_border(l5_br, l5_bl, img_rgb.shape[:2])

        # Draw L5 lower endplate line
        plt.plot([l5_extended_p1[0], l5_extended_p2[0]],
                 [l5_extended_p1[1], l5_extended_p2[1]],
                 color='cyan', linewidth=3, linestyle='-', label='L5 Lower Endplate', zorder=3)

        # Label L5 endplate line
        mid_x = (l5_extended_p1[0] + l5_extended_p2[0]) / 2
        mid_y = (l5_extended_p1[1] + l5_extended_p2[1]) / 2
        plt.text(mid_x, mid_y + 10, 'L5 Lower Endplate', color='cyan', fontsize=12,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=2))

    # Draw perpendicular line to L1 upper endplate
    if 'L1' in all_corners:
        l1_corners = all_corners['L1']
        l1_tl = l1_corners[0]  # TL
        l1_tr = l1_corners[1]  # TR

        # Compute perpendicular line to L1 upper endplate
        l1_perp_p1, l1_perp_p2, l1_midpoint = calculate_perpendicular_line(l1_tl, l1_tr, img_rgb.shape[:2])

        # Draw perpendicular line
        plt.plot([l1_perp_p1[0], l1_perp_p2[0]],
                 [l1_perp_p1[1], l1_perp_p2[1]],
                 color='orange', linewidth=2.5, linestyle='-', label='L1 Perpendicular', zorder=3)

        # Mark midpoint
        plt.scatter([l1_midpoint[0]], [l1_midpoint[1]],
                    color='orange', s=60, marker='s', edgecolors='white', linewidth=1.5, zorder=5)

        # Label perpendicular line
        mid_x = (l1_perp_p1[0] + l1_perp_p2[0]) / 2
        mid_y = (l1_perp_p1[1] + l1_perp_p2[1]) / 2
        plt.text(mid_x + 5, mid_y, 'L1 Perpendicular', color='orange', fontsize=10,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=1))

    # Draw perpendicular line to L5 lower endplate (replacing original S1)
    if 'L5' in all_corners:
        l5_corners = all_corners['L5']
        l5_br = l5_corners[3]  # BR
        l5_bl = l5_corners[2]  # BL

        # Compute perpendicular line to L5 lower endplate
        l5_perp_p1, l5_perp_p2, l5_midpoint = calculate_perpendicular_line(l5_br, l5_bl, img_rgb.shape[:2])

        # Draw perpendicular line
        plt.plot([l5_perp_p1[0], l5_perp_p2[0]],
                 [l5_perp_p1[1], l5_perp_p2[1]],
                 color='purple', linewidth=2.5, linestyle='-', label='L5 Perpendicular', zorder=3)

        # Mark midpoint
        plt.scatter([l5_midpoint[0]], [l5_midpoint[1]],
                    color='purple', s=60, marker='s', edgecolors='white', linewidth=1.5, zorder=5)

        # Label perpendicular line
        mid_x = (l5_perp_p1[0] + l5_perp_p2[0]) / 2
        mid_y = (l5_perp_p1[1] + l5_perp_p2[1]) / 2
        plt.text(mid_x + 5, mid_y, 'L5 Perpendicular', color='purple', fontsize=10,
                 weight='bold', bbox=dict(facecolor='black', alpha=0.7, pad=1))

    # Calculate Cobb angle (angle between L1 and L5 perpendicular lines)
    if 'L1' in all_corners and 'L5' in all_corners:
        # Compute direction vectors of perpendicular lines
        l1_perp_p1, l1_perp_p2, _ = calculate_perpendicular_line(
            all_corners['L1'][0], all_corners['L1'][1], img_rgb.shape[:2]
        )
        l5_perp_p1, l5_perp_p2, _ = calculate_perpendicular_line(
            all_corners['L5'][3], all_corners['L5'][2], img_rgb.shape[:2]
        )

        # Direction vectors
        l1_vec = np.array([l1_perp_p2[0] - l1_perp_p1[0], l1_perp_p2[1] - l1_perp_p1[1]])
        l5_vec = np.array([l5_perp_p2[0] - l5_perp_p1[0], l5_perp_p2[1] - l5_perp_p1[1]])

        # Compute angle between vectors (Cobb angle)
        dot_product = np.dot(l1_vec, l5_vec)
        l1_norm = np.linalg.norm(l1_vec)
        l5_norm = np.linalg.norm(l5_vec)

        if l1_norm > 0 and l5_norm > 0:
            cos_angle = dot_product / (l1_norm * l5_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)  # avoid floating point errors
            cobb_angle = np.degrees(np.arccos(cos_angle))
            # Take acute angle (< 90°)
            if cobb_angle > 90:
                cobb_angle = 180 - cobb_angle

            # Display Cobb angle
            plt.text(20, 30, f'Cobb Angle (L1-L5): {cobb_angle:.2f}°',
                     color='white', fontsize=14, weight='bold',
                     bbox=dict(facecolor='black', alpha=0.8, pad=3))

    # Configure image
    plt.title("Vertebra Corner Annotation (with Endplates and Perpendicular Lines)", fontsize=16)
    plt.axis('off')
    plt.tight_layout()

    # Save result
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Corner annotation result saved to: {output_path}")


# -------------------------- Batch Processing Function --------------------------
def process_single_image(original_img_path, mask_path, output_dir):
    """Process a single image"""
    # Read input data
    original_img = cv2.imread(original_img_path)
    if original_img is None:
        print(f"Error: Cannot read original image: {original_img_path}")
        return False

    mask = Image.open(mask_path)
    mask_array = np.array(mask, dtype=np.int32)

    # Detect corners for all vertebrae
    all_corners = {}
    for label, name in VERTEBRA_LABELS.items():
        print(f"Detecting corners for {name}...")
        # Extract mask for a single vertebra
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        # Detect corners (special handling for S1)
        is_S1 = (name == "S1")
        corners = detect_vertebra_corners(vertebra_mask, is_S1)
        all_corners[name] = corners

    # Visualize and save result
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_corners.png"
    output_path = os.path.join(output_dir, output_filename)
    visualize_corners(original_img, mask_array, all_corners, output_path)

    # Save corner coordinates to text file
    coords_path = os.path.join(output_dir, output_filename.replace(".png", "_coords.txt"))
    with open(coords_path, "w", encoding="utf-8") as f:
        for name, corners in all_corners.items():
            f.write(f"====={name} Corner Coordinates=====\n")
            f.write("TL (Top-Left) | TR (Top-Right) | BL (Bottom-Left) | BR (Bottom-Right)\n")
            f.write(f"{corners[0]} | {corners[1]} | {corners[2]} | {corners[3]}\n\n")

    # Save endplate and perpendicular line information
    lines_path = os.path.join(output_dir, output_filename.replace(".png", "_lines.txt"))
    with open(lines_path, "w", encoding="utf-8") as f:
        f.write("=====Endplate Information=====\n")
        if 'L1' in all_corners:
            l1_corners = all_corners['L1']
            l1_tl = l1_corners[0]
            l1_tr = l1_corners[1]
            l1_extended_p1, l1_extended_p2 = extend_line_to_border(l1_tl, l1_tr, original_img.shape[:2])
            l1_perp_p1, l1_perp_p2, l1_midpoint = calculate_perpendicular_line(l1_tl, l1_tr, original_img.shape[:2])

            f.write("L1 Upper Endplate Line:\n")
            f.write(f"Original points: TL{l1_tl} -> TR{l1_tr}\n")
            f.write(f"Extended: {l1_extended_p1} -> {l1_extended_p2}\n")
            f.write(f"Perpendicular line: {l1_perp_p1} -> {l1_perp_p2}\n")
            f.write(f"Midpoint: {l1_midpoint}\n\n")

        if 'L5' in all_corners:
            l5_corners = all_corners['L5']
            l5_br = l5_corners[3]  # BR
            l5_bl = l5_corners[2]  # BL

            l5_extended_p1, l5_extended_p2 = extend_line_to_border(l5_br, l5_bl, original_img.shape[:2])
            l5_perp_p1, l5_perp_p2, l5_midpoint = calculate_perpendicular_line(l5_br, l5_bl, original_img.shape[:2])

            f.write("L5 Lower Endplate Line:\n")
            f.write(f"Original points: BR{l5_br} -> BL{l5_bl}\n")
            f.write(f"Extended: {l5_extended_p1} -> {l5_extended_p2}\n")
            f.write(f"Perpendicular line: {l5_perp_p1} -> {l5_perp_p2}\n")
            f.write(f"Midpoint: {l5_midpoint}\n\n")

        # Calculate and save Cobb angle (angle between L1 and L5 perpendicular lines)
        if 'L1' in all_corners and 'L5' in all_corners:
            l1_perp_p1, l1_perp_p2, _ = calculate_perpendicular_line(
                all_corners['L1'][0], all_corners['L1'][1], original_img.shape[:2]
            )
            l5_perp_p1, l5_perp_p2, _ = calculate_perpendicular_line(
                all_corners['L5'][3], all_corners['L5'][2], original_img.shape[:2]
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

                f.write(f"Cobb Angle (L1-L5): {cobb_angle:.2f}°\n")

    print(f"Corner coordinates saved to: {coords_path}")
    print(f"Endplate and perpendicular line information saved to: {lines_path}")
    return True


def main():
    # Get all image files
    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    original_images = []

    for filename in os.listdir(ORIGINAL_IMG_DIR):
        if any(filename.lower().endswith(ext) for ext in img_extensions):
            original_images.append(filename)

    if not original_images:
        print(f"No image files found in directory {ORIGINAL_IMG_DIR}")
        return

    print(f"Found {len(original_images)} images to process")

    # Process each image
    for img_filename in original_images:
        print(f"\nProcessing image: {img_filename}")

        # Build full paths
        original_img_path = os.path.join(ORIGINAL_IMG_DIR, img_filename)

        # Find corresponding mask file (assume same filename, different extension)
        img_name_without_ext = os.path.splitext(img_filename)[0]
        mask_filename = None

        # Try different mask file extensions
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            potential_mask_path = os.path.join(MASK_DIR, img_name_without_ext + ext)
            if os.path.exists(potential_mask_path):
                mask_filename = img_name_without_ext + ext
                break

        if mask_filename is None:
            print(f"Warning: Mask file for {img_filename} not found, skipping")
            continue

        mask_path = os.path.join(MASK_DIR, mask_filename)

        # Create separate output directory for this image
        image_output_dir = os.path.join(OUTPUT_PARENT_DIR, img_name_without_ext)
        os.makedirs(image_output_dir, exist_ok=True)

        # Process single image
        success = process_single_image(original_img_path, mask_path, image_output_dir)

        if success:
            print(f"Successfully processed image: {img_filename}")
        else:
            print(f"Failed to process image: {img_filename}")

    print(f"\nBatch processing completed! All results saved in: {OUTPUT_PARENT_DIR}")


if __name__ == "__main__":
    main()
