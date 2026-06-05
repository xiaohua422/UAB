import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def compute_group_mIoU(hist, group_indices, name_classes):
    """
    Calculate mIoU for a group of specified classes

    Parameters:
        hist: confusion matrix
        group_indices: list of class indices to calculate
        name_classes: list of class names

    Return:
        group_mIoU: mean IoU of the group
        group_IoUs: IoU for each class
    """
    group_IoUs = []

    for idx in group_indices:
        if idx == 0:
            continue

        iou = hist[idx, idx] / (np.sum(hist[idx, :]) + np.sum(hist[:, idx]) - hist[idx, idx] + 1e-8)
        group_IoUs.append(iou)

    group_mIoU = np.mean(group_IoUs) if group_IoUs else 0

    return group_mIoU, group_IoUs


if __name__ == "__main__":
    miou_mode = 0
    num_classes = 13
    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5", "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]

    # Define class indices for different anatomical regions
    vertebral_indices = [1, 2, 3, 4, 5, 11]
    disc_indices = [6, 7, 8, 9, 10]
    lumbar_indices = list(set(vertebral_indices + disc_indices))
    csf_indices = [12]

    VOCdevkit_path = 'VOCdevkit'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        # Vertebral mIoU
        print("\nCalculating vertebral mIoU...")
        vertebral_mIoU, vertebral_IoUs = compute_group_mIoU(hist, vertebral_indices, name_classes)
        print(f"Vertebral mIoU: {vertebral_mIoU:.4f}")
        for i, idx in enumerate(vertebral_indices):
            if idx != 0:
                print(f"  {name_classes[idx]}: {vertebral_IoUs[i]:.4f}")

        # Disc mIoU
        print("\nCalculating disc mIoU...")
        disc_mIoU, disc_IoUs = compute_group_mIoU(hist, disc_indices, name_classes)
        print(f"Disc mIoU: {disc_mIoU:.4f}")
        for i, idx in enumerate(disc_indices):
            if idx != 0:
                print(f"  {name_classes[idx]}: {disc_IoUs[i]:.4f}")

        # Lumbar mIoU
        print("\nCalculating lumbar mIoU...")
        lumbar_mIoU, lumbar_IoUs = compute_group_mIoU(hist, lumbar_indices, name_classes)
        print(f"Lumbar mIoU: {lumbar_mIoU:.4f}")
        for i, idx in enumerate(lumbar_indices):
            if idx != 0:
                print(f"  {name_classes[idx]}: {lumbar_IoUs[i]:.4f}")

        # CSF mIoU
        print("\nCalculating CSF mIoU...")
        csf_mIoU, csf_IoUs = compute_group_mIoU(hist, csf_indices, name_classes)
        print(f"CSF mIoU: {csf_mIoU:.4f}")
        print(f"  {name_classes[csf_indices[0]]}: {csf_IoUs[0]:.4f}")

        # Save results
        result_file = os.path.join(miou_out_path, "miou_group_results.txt")
        with open(result_file, "w") as f:
            f.write("Overall mIoU Results:\n")
            for i in range(1, num_classes):
                f.write("  {}: {:.4f}\n".format(name_classes[i], IoUs[i - 1]))

            f.write("\nVertebral mIoU: {:.4f}\n".format(vertebral_mIoU))
            for i, idx in enumerate(vertebral_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], vertebral_IoUs[i]))

            f.write("\nDisc mIoU: {:.4f}\n".format(disc_mIoU))
            for i, idx in enumerate(disc_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], disc_IoUs[i]))

            f.write("\nLumbar mIoU: {:.4f}\n".format(lumbar_mIoU))
            for i, idx in enumerate(lumbar_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], lumbar_IoUs[i]))

            f.write("\nCSF mIoU: {:.4f}\n".format(csf_mIoU))
            f.write("  {}: {:.4f}\n".format(name_classes[csf_indices[0]], csf_IoUs[0]))

        print(f"\nGroup mIoU results saved to {result_file}")
