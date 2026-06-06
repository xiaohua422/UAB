import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import csv

plt.rcParams['font.family'] = ['Times New Roman', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# -------------------------- Configuration Parameters --------------------------
ORIGINAL_IMG_DIR = "img/"
MASK_DIR = "mask/"
OUTPUT_PARENT_DIR = "cobb12_amace_cobb_results"
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

# -------------------------- Line Detection and Processing Functions --------------------------
def extend_line_to_border(p1, p2, img_shape):
    """Extend a line segment to the image borders and return the two intersection points."""
    h, w = img_shape[:2]
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:  # vertical line
        return (x1, 0), (x1, h-1)
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    intersections = []
    # left border x=0
    y_left = b
    if 0 <= y_left <= h-1:
        intersections.append((0, y_left))
    # right border x=w-1
    y_right = m * (w-1) + b
    if 0 <= y_right <= h-1:
        intersections.append((w-1, y_right))
    # top border y=0
    if m != 0:
        x_top = -b / m
        if 0 <= x_top <= w-1:
            intersections.append((x_top, 0))
    # bottom border y=h-1
    if m != 0:
        x_bottom = (h-1 - b) / m
        if 0 <= x_bottom <= w-1:
            intersections.append((x_bottom, h-1))
    if len(intersections) >= 2:
        # Choose the two points farthest from the original segment endpoints
        distances = [np.linalg.norm(np.array(p) - np.array(p1)) + np.linalg.norm(np.array(p) - np.array(p2)) for p in intersections]
        sorted_indices = np.argsort(distances)[-2:]
        return intersections[sorted_indices[0]], intersections[sorted_indices[1]]
    return p1, p2  # fallback: return original endpoints

def calculate_cobb_angle(line1, line2):
    """Calculate the acute angle between two lines (Cobb angle)."""
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
    For a single vertebra mask, use Canny edge detection and Hough transform
    to detect the upper and lower endplate lines.
    Returns two lines, each represented by two points (the endpoints of the segment).
    """
    # Ensure mask is binary
    mask = vertebra_mask.copy().astype(np.uint8)
    if np.sum(mask) == 0:
        print(f"Warning: {label_name} mask is empty, skipping")
        return None, None

    # Get vertebra bounding box for later filtering
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    center_y = (y_min + y_max) // 2

    # Edge detection (Canny)
    edges = cv2.Canny(mask, 50, 150)  # thresholds can be adjusted

    # Hough transform to detect line segments (probabilistic Hough)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=30,
                            minLineLength=max(20, (x_max-x_min)//3),
                            maxLineGap=10)

    if lines is None or len(lines) == 0:
        print(f"Warning: No lines detected for {label_name}, using bounding box edges instead")
        # Fallback: use top and bottom edges of bounding box
        top_line = [(x_min, y_min), (x_max, y_min)]
        bottom_line = [(x_min, y_max), (x_max, y_max)]
        return top_line, bottom_line

    # Collect all segments
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append(((x1, y1), (x2, y2)))

    # Separate segments into top/bottom candidates based on midpoint y-coordinate
    top_candidates = []
    bottom_candidates = []
    for seg in segments:
        mid_y = (seg[0][1] + seg[1][1]) / 2
        if mid_y < center_y:
            top_candidates.append(seg)
        else:
            bottom_candidates.append(seg)

    # If no candidates, put all segments into both sets
    if len(top_candidates) == 0:
        top_candidates = segments
    if len(bottom_candidates) == 0:
        bottom_candidates = segments

    # Choose the longest segment as representative (could also merge multiple segments)
    def select_longest(candidates):
        if not candidates:
            return None
        longest = max(candidates, key=lambda s: np.hypot(s[1][0]-s[0][0], s[1][1]-s[0][1]))
        return longest

    top_line = select_longest(top_candidates)
    bottom_line = select_longest(bottom_candidates)

    # If one side still has no line, use bounding box edge
    if top_line is None:
        top_line = [(x_min, y_min), (x_max, y_min)]
        print(f"Warning: {label_name} upper endplate not detected, using bounding box top edge")
    if bottom_line is None:
        bottom_line = [(x_min, y_max), (x_max, y_max)]
        print(f"Warning: {label_name} lower endplate not detected, using bounding box bottom edge")

    return top_line, bottom_line

# -------------------------- Visualization Function --------------------------
def visualize_endplates(original_img, mask, vertebra_lines, output_path):
    """
    vertebra_lines: dict, key = vertebra name, value = (top_line, bottom_line)
    Each line is a tuple of two points ((x1,y1),(x2,y2)).
    """
    if original_img.ndim == 3 and original_img.shape[-1] == 3:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(12, 15))
    plt.imshow(img_rgb)

    # Draw endplate lines for each vertebra (extended to borders)
    extended_lines = {}  # store extended lines for Cobb angle calculation
    for name, (top_line, bottom_line) in vertebra_lines.items():
        color = VERTEBRA_COLORS[name]
        # Extend to image borders
        top_ext = extend_line_to_border(top_line[0], top_line[1], img_rgb.shape)
        bottom_ext = extend_line_to_border(bottom_line[0], bottom_line[1], img_rgb.shape)
        extended_lines[name] = (top_ext, bottom_ext)

        # Draw extended lines (red)
        plt.plot([top_ext[0][0], top_ext[1][0]], [top_ext[0][1], top_ext[1][1]],
                 color='red', linewidth=2, zorder=3)
        plt.plot([bottom_ext[0][0], bottom_ext[1][0]], [bottom_ext[0][1], bottom_ext[1][1]],
                 color='red', linewidth=2, zorder=3)

        # Optionally draw original segments (thinner dashed lines) to show detected lines
        plt.plot([top_line[0][0], top_line[1][0]], [top_line[0][1], top_line[1][1]],
                 color=color, linewidth=1, linestyle='--', zorder=2)
        plt.plot([bottom_line[0][0], bottom_line[1][0]], [bottom_line[0][1], bottom_line[1][1]],
                 color=color, linewidth=1, linestyle='--', zorder=2)

        # Label vertebra name
        mid_x = (top_line[0][0] + top_line[1][0] + bottom_line[0][0] + bottom_line[1][0]) / 4
        mid_y = (top_line[0][1] + top_line[1][1] + bottom_line[0][1] + bottom_line[1][1]) / 4
        plt.text(mid_x, mid_y, name, color=color, fontsize=12, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.5, pad=1))

    # Draw Cobb angle annotations for each adjacent pair
    for i in range(1, 5):
        upper = extended_lines[f"L{i}"][1]   # lower endplate of upper vertebra
        lower = extended_lines[f"L{i+1}"][0] # upper endplate of lower vertebra
        angle = calculate_cobb_angle(upper, lower)
        # Place angle label between the two lines
        mid1 = ((upper[0][0] + upper[1][0]) / 2, (upper[0][1] + upper[1][1]) / 2)
        mid2 = ((lower[0][0] + lower[1][0]) / 2, (lower[0][1] + lower[1][1]) / 2)
        center = ((mid1[0] + mid2[0]) / 2, (mid1[1] + mid2[1]) / 2)
        plt.text(center[0], center[1], f"L{i}-L{i+1} Cobb:{angle:.1f}°",
                 color='yellow', fontsize=10, weight='bold',
                 bbox=dict(facecolor='black', alpha=0.6, pad=1))

    # Calculate overall L1-L5 Cobb angle
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
    print(f"Cobb angle result saved to: {output_path}")

# -------------------------- Batch Processing Function (returns angle data) --------------------------
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

    vertebra_lines = {}  # store (top_line, bottom_line) for each vertebra
    for label, name in VERTEBRA_LABELS.items():
        print(f"Processing {name}...")
        vertebra_mask = (mask_array == label).astype(np.uint8) * 255
        top_line, bottom_line = detect_endplate_lines(vertebra_mask, name)
        if top_line is None or bottom_line is None:
            print(f"Error: {name} endplate detection failed, skipping this image.")
            return None
        vertebra_lines[name] = (top_line, bottom_line)

    # Calculate segment Cobb angles (need to extend lines to borders first)
    img_shape = original_img.shape[:2]
    extended_lines = {}
    for name, (top_line, bottom_line) in vertebra_lines.items():
        top_ext = extend_line_to_border(top_line[0], top_line[1], img_shape)
        bottom_ext = extend_line_to_border(bottom_line[0], bottom_line[1], img_shape)
        extended_lines[name] = (top_ext, bottom_ext)

    segment_angles = {}
    for i in range(1, 5):
        upper = extended_lines[f"L{i}"][1]      # lower endplate of L_i
        lower = extended_lines[f"L{i+1}"][0]    # upper endplate of L_{i+1}
        angle = calculate_cobb_angle(upper, lower)
        segment_angles[f"L{i}_L{i+1}"] = angle

    # Calculate overall L1-L5 Cobb angle (L1 lower endplate vs L5 upper endplate)
    L1_line = extended_lines["L1"][1]
    L5_line = extended_lines["L5"][0]
    total_angle = calculate_cobb_angle(L1_line, L5_line)

    # Save visualization result
    output_filename = os.path.splitext(os.path.basename(original_img_path))[0] + "_cobb_hough.png"
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
        **segment_angles
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
