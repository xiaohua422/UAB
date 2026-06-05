import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


def compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes):
    """
    Calculate mean Dice coefficient
    """
    print("Calculate mDice...")
    hist = np.zeros((num_classes, num_classes))
    dice_per_class = np.zeros(num_classes)

    for image_id in tqdm(image_ids):
        label_path = os.path.join(gt_dir, image_id + ".png")
        pred_path = os.path.join(pred_dir, image_id + ".png")

        label = Image.open(label_path)
        pred = Image.open(pred_path)

        label = np.array(label)
        pred = np.array(pred)

        if len(label.shape) == 3:
            label = label[:, :, 0]
        if len(pred.shape) == 3:
            pred = pred[:, :, 0]

        label = label.flatten()
        pred = pred.flatten()

        for cls in range(num_classes):
            gt_mask = (label == cls).astype(np.int32)
            pred_mask = (pred == cls).astype(np.int32)

            intersection = np.sum(gt_mask * pred_mask)
            union = np.sum(gt_mask) + np.sum(pred_mask)

            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
            dice_per_class[cls] += dice

    dice_per_class /= len(image_ids)
    mdice = np.mean(dice_per_class)

    return mdice, dice_per_class


def show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True):
    """
    Show and save Dice results to TXT and PNG
    """
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    if include_background:
        dice_data = dice_per_class
        class_names = name_classes
        title_suffix = " (with background)"
        filename_suffix = ""
    else:
        dice_data = dice_per_class[1:]
        class_names = name_classes[1:]
        title_suffix = " (without background)"
        filename_suffix = "_no_bg"

    mdice = np.mean(dice_data)

    print(f"Dice coefficients for each class{title_suffix}:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {dice_data[i]:.4f}")

    print(f"mDice{title_suffix}: {mdice:.4f}")

    with open(os.path.join(miou_out_path, f"dice_results{filename_suffix}.txt"), "w") as f:
        f.write(f"Dice coefficients for each class{title_suffix}:\n")
        for i in range(len(class_names)):
            f.write(f"{class_names[i]}: {dice_data[i]:.4f}\n")
        f.write(f"mDice{title_suffix}: {mdice:.4f}\n")

    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    bars = plt.bar(range(len(class_names)), dice_data, color=colors)

    plt.title(f'Dice Coefficients by Class{title_suffix} (mDice: {mdice:.4f})', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)

    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')

    for i, v in enumerate(dice_data):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    plt.axhline(y=mdice, color='r', linestyle='--', alpha=0.7, label=f'mDice: {mdice:.4f}')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(miou_out_path, f"dice_results{filename_suffix}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Results saved to {os.path.join(miou_out_path, f'dice_results{filename_suffix}.txt')}")


if __name__ == "__main__":
    miou_mode = 3

    num_classes = 13
    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5",
                    "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1", "CSF"]

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

        IoUs_no_bg = IoUs[1:]
        PA_Recall_no_bg = PA_Recall[1:]
        Precision_no_bg = Precision[1:]
        name_classes_no_bg = name_classes[1:]
        miou_no_bg = np.nanmean(IoUs_no_bg)

        print("=" * 50)
        print("mIoU including background:")
        for i in range(num_classes):
            print(f"{name_classes[i]}: {IoUs[i]:.4f}")
        print(f"Mean mIoU: {np.nanmean(IoUs):.4f}")

        print("\n" + "=" * 50)
        print("mIoU excluding background:")
        for i in range(len(name_classes_no_bg)):
            print(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}")
        print(f"Mean mIoU (no bg): {miou_no_bg:.4f}")

        with open(os.path.join(miou_out_path, "miou_results.txt"), "w") as f:
            f.write("mIoU including background:\n")
            for i in range(num_classes):
                f.write(f"{name_classes[i]}: {IoUs[i]:.4f}\n")
            f.write(f"Mean mIoU: {np.nanmean(IoUs):.4f}\n\n")

            f.write("mIoU excluding background:\n")
            for i in range(len(name_classes_no_bg)):
                f.write(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}\n")
            f.write(f"Mean mIoU (no bg): {miou_no_bg:.4f}\n")

        print("Get miou done.")

    if miou_mode == 3:
        print("Get mDice.")
        mdice, dice_per_class = compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get mDice done.")

        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True)
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=False)

    if miou_mode == 4:
        print("Get miou and mDice.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        IoUs_no_bg = IoUs[1:]
        PA_Recall_no_bg = PA_Recall[1:]
        Precision_no_bg = Precision[1:]
        name_classes_no_bg = name_classes[1:]
        miou_no_bg = np.nanmean(IoUs_no_bg)

        print("=" * 50)
        print("mIoU including background:")
        for i in range(num_classes):
            print(f"{name_classes[i]}: {IoUs[i]:.4f}")
        print(f"Mean mIoU: {np.nanmean(IoUs):.4f}")

        print("\n" + "=" * 50)
        print("mIoU excluding background:")
        for i in range(len(name_classes_no_bg)):
            print(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}")
        print(f"Mean mIoU (no bg): {miou_no_bg:.4f}")

        with open(os.path.join(miou_out_path, "miou_results.txt"), "w") as f:
            f.write("mIoU including background:\n")
            for i in range(num_classes):
                f.write(f"{name_classes[i]}: {IoUs[i]:.4f}\n")
            f.write(f"Mean mIoU: {np.nanmean(IoUs):.4f}\n\n")

            f.write("mIoU excluding background:\n")
            for i in range(len(name_classes_no_bg)):
                f.write(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}\n")
            f.write(f"Mean mIoU (no bg): {miou_no_bg:.4f}\n")

        mdice, dice_per_class = compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True)
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=False)

        print("Get miou and mDice done.")
