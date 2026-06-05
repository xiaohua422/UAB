import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def compute_group_metrics(hist, group_indices, name_classes, metric_type='iou'):
    """
    Calculate average metrics (IoU or Dice) for a specified class group

    Parameters:
        hist: confusion matrix
        group_indices: list of class indices (excluding background 0)
        name_classes: class names
        metric_type: 'iou' or 'dice'

    Return:
        group_mean: mean value of the group
        group_values: list of individual values
    """
    group_values = []
    for idx in group_indices:
        if idx == 0:
            continue
        tp = hist[idx, idx]
        fp = np.sum(hist[:, idx]) - tp
        fn = np.sum(hist[idx, :]) - tp
        if metric_type == 'iou':
            union = tp + fp + fn
            value = tp / (union + 1e-8)
        else:
            value = 2 * tp / (2 * tp + fp + fn + 1e-8)
        group_values.append(value)
    group_mean = np.mean(group_values) if group_values else 0
    return group_mean, group_values


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    # miou_mode: 0 = full pipeline (predict + compute metrics)
    #            1 = predict only
    #            2 = compute metrics only (IoU + Dice)
    # ---------------------------------------------------------------------------#
    miou_mode = 0

    num_classes = 13

    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5",
                    "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]

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
        deeplabv3 = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            image = Image.open(image_path)
            image = deeplabv3.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")

        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        all_ious = []
        all_recalls = []
        all_precisions = []
        all_dices = []

        for c in range(num_classes):
            tp = hist[c, c]
            fn = np.sum(hist[c, :]) - tp
            fp = np.sum(hist[:, c]) - tp
            union = tp + fn + fp
            iou = tp / (union + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            dice = 2 * tp / (2 * tp + fp + fn + 1e-8)

            all_ious.append(iou)
            all_recalls.append(recall)
            all_precisions.append(precision)
            all_dices.append(dice)

        miou_with_bg = np.mean(all_ious)
        mpa_with_bg = np.mean(all_recalls)
        mean_precision_with_bg = np.mean(all_precisions)
        mdice_with_bg = np.mean(all_dices)

        miou_without_bg = np.mean(all_ious[1:])
        mpa_without_bg = np.mean(all_recalls[1:])
        mean_precision_without_bg = np.mean(all_precisions[1:])
        mdice_without_bg = np.mean(all_dices[1:])

        print("\n" + "=" * 60)
        print("Overall Results (including background):")
        print("  mIoU: {:.4f}".format(miou_with_bg))
        print("  MPA (mean recall): {:.4f}".format(mpa_with_bg))
        print("  Mean Precision: {:.4f}".format(mean_precision_with_bg))
        print("  mDice: {:.4f}".format(mdice_with_bg))

        print("\nOverall Results (excluding background):")
        print("  mIoU: {:.4f}".format(miou_without_bg))
        print("  MPA (mean recall): {:.4f}".format(mpa_without_bg))
        print("  Mean Precision: {:.4f}".format(mean_precision_without_bg))
        print("  mDice: {:.4f}".format(mdice_without_bg))
        print("=" * 60)

        print("\nPer-class metrics (including background):")
        print("{:<12} {:>8} {:>8} {:>10} {:>8}".format("Class", "IoU", "PA", "Precision", "Dice"))
        for c, name in enumerate(name_classes):
            print("{:<12} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f}".format(
                name, all_ious[c], all_recalls[c], all_precisions[c], all_dices[c]))

        print("\nPer-class metrics (excluding background):")
        print("{:<12} {:>8} {:>8} {:>10} {:>8}".format("Class", "IoU", "PA", "Precision", "Dice"))
        for c in range(1, num_classes):
            print("{:<12} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f}".format(
                name_classes[c], all_ious[c], all_recalls[c], all_precisions[c], all_dices[c]))

        print("\n--- Group IoU (excluding background) ---")
        vertebral_mIoU, vertebral_IoUs = compute_group_metrics(hist, vertebral_indices, name_classes, 'iou')
        print("Vertebral mIoU: {:.4f}".format(vertebral_mIoU))
        for i, idx in enumerate(vertebral_indices):
            if idx != 0:
                print("  {}: {:.4f}".format(name_classes[idx], vertebral_IoUs[i]))

        disc_mIoU, disc_IoUs = compute_group_metrics(hist, disc_indices, name_classes, 'iou')
        print("\nDisc mIoU: {:.4f}".format(disc_mIoU))
        for i, idx in enumerate(disc_indices):
            print("  {}: {:.4f}".format(name_classes[idx], disc_IoUs[i]))

        lumbar_mIoU, lumbar_IoUs = compute_group_metrics(hist, lumbar_indices, name_classes, 'iou')
        print("\nLumbar mIoU: {:.4f}".format(lumbar_mIoU))
        for i, idx in enumerate(lumbar_indices):
            if idx != 0:
                print("  {}: {:.4f}".format(name_classes[idx], lumbar_IoUs[i]))

        csf_mIoU, csf_IoUs = compute_group_metrics(hist, csf_indices, name_classes, 'iou')
        print("\nCSF mIoU: {:.4f}".format(csf_mIoU))
        print("  {}: {:.4f}".format(name_classes[csf_indices[0]], csf_IoUs[0]))

        print("\n--- Group Dice (excluding background) ---")
        vertebral_mDice, vertebral_Dices = compute_group_metrics(hist, vertebral_indices, name_classes, 'dice')
        print("Vertebral mDice: {:.4f}".format(vertebral_mDice))
        for i, idx in enumerate(vertebral_indices):
            if idx != 0:
                print("  {}: {:.4f}".format(name_classes[idx], vertebral_Dices[i]))

        disc_mDice, disc_Dices = compute_group_metrics(hist, disc_indices, name_classes, 'dice')
        print("\nDisc mDice: {:.4f}".format(disc_mDice))
        for i, idx in enumerate(disc_indices):
            print("  {}: {:.4f}".format(name_classes[idx], disc_Dices[i]))

        lumbar_mDice, lumbar_Dices = compute_group_metrics(hist, lumbar_indices, name_classes, 'dice')
        print("\nLumbar mDice: {:.4f}".format(lumbar_mDice))
        for i, idx in enumerate(lumbar_indices):
            if idx != 0:
                print("  {}: {:.4f}".format(name_classes[idx], lumbar_Dices[i]))

        csf_mDice, csf_Dices = compute_group_metrics(hist, csf_indices, name_classes, 'dice')
        print("\nCSF mDice: {:.4f}".format(csf_mDice))
        print("  {}: {:.4f}".format(name_classes[csf_indices[0]], csf_Dices[0]))

        result_file = os.path.join(miou_out_path, "miou_dice_results.txt")
        with open(result_file, "w") as f:
            f.write("=== Overall Results ===\n\n")
            f.write("Including background:\n")
            f.write("  mIoU: {:.4f}\n".format(miou_with_bg))
            f.write("  MPA (mean recall): {:.4f}\n".format(mpa_with_bg))
            f.write("  Mean Precision: {:.4f}\n".format(mean_precision_with_bg))
            f.write("  mDice: {:.4f}\n\n".format(mdice_with_bg))

            f.write("Excluding background:\n")
            f.write("  mIoU: {:.4f}\n".format(miou_without_bg))
            f.write("  MPA (mean recall): {:.4f}\n".format(mpa_without_bg))
            f.write("  Mean Precision: {:.4f}\n".format(mean_precision_without_bg))
            f.write("  mDice: {:.4f}\n\n".format(mdice_without_bg))

            f.write("=== Per-class Metrics (including background) ===\n")
            f.write("{:<12} {:>8} {:>8} {:>10} {:>8}\n".format("Class", "IoU", "PA", "Precision", "Dice"))
            for c, name in enumerate(name_classes):
                f.write("{:<12} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f}\n".format(
                    name, all_ious[c], all_recalls[c], all_precisions[c], all_dices[c]))

            f.write("\n=== Per-class Metrics (excluding background) ===\n")
            f.write("{:<12} {:>8} {:>8} {:>10} {:>8}\n".format("Class", "IoU", "PA", "Precision", "Dice"))
            for c in range(1, num_classes):
                f.write("{:<12} {:>8.4f} {:>8.4f} {:>10.4f} {:>8.4f}\n".format(
                    name_classes[c], all_ious[c], all_recalls[c], all_precisions[c], all_dices[c]))

            f.write("\n=== Group IoU (excluding background) ===\n")
            f.write("Vertebral mIoU: {:.4f}\n".format(vertebral_mIoU))
            for i, idx in enumerate(vertebral_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], vertebral_IoUs[i]))
            f.write("\nDisc mIoU: {:.4f}\n".format(disc_mIoU))
            for i, idx in enumerate(disc_indices):
                f.write("  {}: {:.4f}\n".format(name_classes[idx], disc_IoUs[i]))
            f.write("\nLumbar mIoU: {:.4f}\n".format(lumbar_mIoU))
            for i, idx in enumerate(lumbar_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], lumbar_IoUs[i]))
            f.write("\nCSF mIoU: {:.4f}\n".format(csf_mIoU))
            f.write("  {}: {:.4f}\n".format(name_classes[csf_indices[0]], csf_IoUs[0]))

            f.write("\n=== Group Dice (excluding background) ===\n")
            f.write("Vertebral mDice: {:.4f}\n".format(vertebral_mDice))
            for i, idx in enumerate(vertebral_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], vertebral_Dices[i]))
            f.write("\nDisc mDice: {:.4f}\n".format(disc_mDice))
            for i, idx in enumerate(disc_indices):
                f.write("  {}: {:.4f}\n".format(name_classes[idx], disc_Dices[i]))
            f.write("\nLumbar mDice: {:.4f}\n".format(lumbar_mDice))
            for i, idx in enumerate(lumbar_indices):
                if idx != 0:
                    f.write("  {}: {:.4f}\n".format(name_classes[idx], lumbar_Dices[i]))
            f.write("\nCSF mDice: {:.4f}\n".format(csf_mDice))
            f.write("  {}: {:.4f}\n".format(name_classes[csf_indices[0]], csf_Dices[0]))

        print("\nAll detailed results have been saved to: {}".format(result_file))
