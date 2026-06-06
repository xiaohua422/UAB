import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm

# ----------------- Configuration (Only modify here) -----------------
img_dir = "output4_CSF"
csv_file = os.path.join(img_dir, "CSF_highlight_statistics.csv")
zoom_scale = 2.5
max_offset_ratio = 0.18
overlay_color = (0, 255, 0)
alpha = 0.7
center_smooth_sigma = 3.0
mask_blur_k = 21
# -------------------------------------------------------------------

def process_csf_signal_mapping(base_name, img_dir, csv_file, zoom_scale, max_offset_ratio,
                               alpha, center_smooth_sigma, mask_blur_k):
    """Process CSF signal mapping for a single image"""

    mri_img_path = os.path.join(img_dir, base_name + ".png")
    curve_img_path = os.path.join(img_dir, base_name + "_CSF_signal_curve_compare.png")

    # ---- Read bounding box ----
    df = pd.read_csv(csv_file)
    row = df.loc[df['Filename'] == base_name + ".png"]
    if row.shape[0] == 0:
        print(f"[Warning] No data found for {base_name}.png in CSV, skipping")
        return False

    bbox_raw = row['Bounding_Box'].values[0]
    if bbox_raw == 'None':
        print(f"[Info] No CSF region found for {base_name}.png, skipping")
        return False

    nums = ''.join(ch if ch.isdigit() else ',' for ch in str(bbox_raw))
    parts = [p for p in nums.split(',') if p != '']

    if len(parts) == 4:
        x, y, w, h = map(int, parts)
    elif len(parts) == 2:
        a, b = parts
        x = int(a[:3])
        y = int(a[3:]) if len(a) > 3 else 0
        w = int(b[:2])
        h = int(b[2:]) if len(b) > 2 else 0
    else:
        print(f"[Warning] Invalid Bounding_Box format for {base_name}: {bbox_raw}, skipping")
        return False

    # ---- Read original image and curve image ----
    img = cv2.imread(mri_img_path)
    if img is None:
        print(f"[Error] MRI image not found: {mri_img_path}")
        return False

    curve_img = cv2.imread(curve_img_path)
    if curve_img is None:
        print(f"[Error] Curve image not found: {curve_img_path}")
        return False

    # ---- Crop CSF region ----
    H_img, W_img = img.shape[:2]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(W_img, x + w)
    y1 = min(H_img, y + h)
    csf = img[y0:y1, x0:x1].copy()
    if csf.size == 0:
        print(f"[Warning] Empty CSF region for {base_name}, skipping")
        return False

    # ---- Extract 1D signal from CSV or curve image ----
    sig = None
    for col in df.columns:
        name = col.lower()
        if name.startswith("signal") or name.startswith("profile") or name.startswith("mean"):
            v = row[col].values[0]
            if isinstance(v, str):
                try:
                    sig = np.array(eval(v), dtype=float)
                    break
                except:
                    try:
                        sig = np.array(
                            [float(p) for p in v.replace('[', '').replace(']', '').split(',') if p.strip() != ''])
                        break
                    except:
                        sig = None

    if sig is None:
        # Extract from curve image
        ch, cw = curve_img.shape[:2]
        if cw > ch:
            curve_v = cv2.rotate(curve_img, cv2.ROTATE_90_CLOCKWISE)
        else:
            curve_v = curve_img.copy()
        gray = cv2.cvtColor(curve_v, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)

        h_c, w_c = edges.shape
        xs = []
        for r in range(h_c):
            cols = np.where(edges[r] > 0)[0]
            if cols.size == 0:
                rowvals = gray[r]
                cols = np.where(rowvals >= rowvals.mean() + rowvals.std())[0]
            xs.append(cols.mean() if cols.size > 0 else w_c / 2)
        sig = np.array(xs, dtype=float)

    # ---- Extract centerline ----
    csf_gray = cv2.cvtColor(csf, cv2.COLOR_BGR2GRAY)
    csf_blur = cv2.GaussianBlur(csf_gray, (7, 7), 0)
    _, mask = cv2.threshold(csf_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    H, W = mask.shape
    center_xs = []
    prev = W // 2
    for r in range(H):
        cols = np.where(mask[r] > 0)[0]
        if cols.size > 0:
            vals = csf_blur[r, cols].astype(float) + 1e-6
            cx = (cols * vals).sum() / vals.sum()
            prev = cx
        else:
            cx = prev
        center_xs.append(cx)
    center_xs = np.array(center_xs)

    # ---- Gaussian smooth centerline ----
    def gaussian_smooth(a, s):
        rad = int(3 * s)
        x = np.arange(-rad, rad + 1)
        k = np.exp(-(x ** 2) / (2 * s * s))
        k /= k.sum()
        return np.convolve(np.pad(a, (rad, rad), 'edge'), k, 'valid')

    center_xs_s = gaussian_smooth(center_xs, center_smooth_sigma)

    # ---- Resample and normalize signal ----
    sig = np.asarray(sig, float)
    sig = np.interp(np.linspace(0, 1, len(center_xs_s)), np.linspace(0, 1, len(sig)), sig)
    lo, hi = np.percentile(sig, [2, 98])
    sig = np.clip(sig, lo, hi)
    sig = (sig - lo) / (hi - lo + 1e-9)

    # ---- Zoom CSF region ----
    zoom_w = int(W * zoom_scale)
    zoom_h = int(H * zoom_scale)
    csf_zoom = cv2.resize(csf, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)

    center_ys = np.arange(H)
    center_xs_zoom = center_xs_s * zoom_scale
    center_ys_zoom = (center_ys * zoom_scale).astype(int)

    max_offset = int(zoom_w * max_offset_ratio)
    sig_smooth = gaussian_smooth(sig, 3.0)
    offsets = (sig_smooth * max_offset).astype(int)

    pts = np.vstack([center_xs_zoom, center_ys_zoom]).T
    normals = []
    for i in range(len(pts)):
        if i == 0:
            p0, p1 = pts[i], pts[i + 1]
        elif i == len(pts) - 1:
            p0, p1 = pts[i - 1], pts[i]
        else:
            p0, p1 = pts[i - 1], pts[i + 1]
        t = p1 - p0
        n = np.linalg.norm(t)
        if n == 0:
            normals.append((1, 0))
        else:
            t /= n
            normals.append((-t[1], t[0]))
    normals = np.array(normals)

    pts_off = []
    for (cx, cy), (nx, ny), off in zip(pts, normals, offsets):
        pts_off.append((int(cx + nx * off), int(cy + ny * off)))
    pts_off = np.array(pts_off, np.int32)

    band_mask = np.zeros((zoom_h, zoom_w), np.uint8)
    if len(pts_off) >= 2:
        cv2.polylines(band_mask, [pts_off], False, 255, 3, cv2.LINE_AA)
        band_mask = cv2.dilate(band_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), 1)
    band_mask = cv2.GaussianBlur(band_mask, (mask_blur_k, mask_blur_k), 0)
    band_mask_f = (band_mask.astype(float) / 255.0) * alpha

    # === Gradient coloring (Blue→Cyan→Yellow→Red) ===
    colormap = cm.get_cmap('jet')

    # Normalize offsets (consistent with signal)
    offset_norm = (offsets - offsets.min()) / (offsets.max() - offsets.min() + 1e-9)

    # Original colors (H, 3)
    raw_colors = (colormap(offset_norm)[:, :3] * 255).astype(np.uint8)

    # Interpolate colors to zoom_h
    ys_src = np.linspace(0, len(raw_colors) - 1, len(raw_colors))
    ys_dst = np.linspace(0, len(raw_colors) - 1, zoom_h)
    raw_colors_interp = np.array([
        raw_colors[int(y)] for y in ys_dst
    ], dtype=np.uint8)

    # Expand to full image
    color_img = np.zeros_like(csf_zoom)
    for y in range(zoom_h):
        color_img[y, :, :] = raw_colors_interp[y]

    # Blend (band region → gradient color)
    blend = (csf_zoom * (1 - band_mask_f[:, :, None]) + color_img * band_mask_f[:, :, None]).astype(np.uint8)

    # White centerline
    for (cxz, cyz) in zip(center_xs_zoom.astype(int), center_ys_zoom):
        cv2.circle(blend, (cxz, cyz), 1, (255, 255, 255), -1)

    # === Text annotation (High signal region) ===
    text = "High CSF Signal Region"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    # Auto position: near center of band region
    text_x = int(center_xs_zoom[int(len(center_xs_zoom) * 0.3)])
    text_y = int(center_ys_zoom[int(len(center_ys_zoom) * 0.3)]) - 25
    text_x = max(10, min(text_x, blend.shape[1] - 200))
    text_y = max(30, text_y)

    # Semi-transparent background for text
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    overlay = blend.copy()
    cv2.rectangle(overlay, (text_x - 8, text_y - text_h - 8), (text_x + text_w + 8, text_y + 8), (0, 0, 0), -1)
    blend = cv2.addWeighted(overlay, 0.4, blend, 0.6, 0)

    # White text
    cv2.putText(blend, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # ---- Side-by-side comparison ----
    left = csf_zoom
    right = blend
    side_by_side = np.hstack((left, right))

    out1 = os.path.join(img_dir, base_name + "_CSF_zoom_band_compare.png")
    cv2.imwrite(out1, side_by_side)

    # ---- Thumbnail back to original MRI ----
    thumb_h = h
    thumb_w = int(side_by_side.shape[1] * (thumb_h / side_by_side.shape[0]))
    thumb = cv2.resize(side_by_side, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    insert_x = x1 + 8
    insert_y = y0
    need_w = insert_x + thumb_w - img.shape[1]
    if need_w > 0:
        img = cv2.copyMakeBorder(img, 0, 0, 0, need_w + 30, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img[insert_y:insert_y + thumb_h, insert_x:insert_x + thumb_w] = thumb

    out2 = os.path.join(img_dir, base_name + "_CSF_MRI_with_thumb.png")
    cv2.imwrite(out2, img)

    return True


def main():
    """Main function: Batch process all files"""

    # Check if CSV exists
    if not os.path.exists(csv_file):
        print(f"[Error] CSV file not found: {csv_file}")
        return

    # Read CSV and get valid files
    df = pd.read_csv(csv_file)
    valid_files = df[df['Bounding_Box'] != 'None']['Filename'].tolist()
    base_names = [os.path.splitext(f)[0] for f in valid_files]

    print(f"Found {len(base_names)} images with CSF regions to process")

    # Process with progress bar
    success_count = 0
    for base_name in tqdm(base_names, desc="Processing CSF Signal Mapping"):
        try:
            if process_csf_signal_mapping(base_name, img_dir, csv_file, zoom_scale,
                                          max_offset_ratio, alpha, center_smooth_sigma, mask_blur_k):
                success_count += 1
            else:
                print(f"[Info] Failed to process {base_name}")
        except Exception as e:
            print(f"[Error] Error processing {base_name}: {str(e)}")
            continue

    print(f"\n✅ Batch processing completed!")
    print(f"📊 Successfully processed: {success_count}/{len(base_names)} files")
    print(f"📁 Output directory: {img_dir}")


if __name__ == "__main__":
    main()
