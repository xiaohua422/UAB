import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
#from model.pspnet import PSPNet
from utils.utils_metrics import compute_mIoU, show_results

if __name__ == "__main__":
    miou_mode = 0
    # 保持原始类别数量，包括背景
    num_classes = 13
    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5", "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1",
                    "CSF"]

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
        # 计算包含背景的mIoU
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        # 计算不包含背景的mIoU
        IoUs_no_bg = IoUs[1:]  # 排除背景（索引0）
        PA_Recall_no_bg = PA_Recall[1:]
        Precision_no_bg = Precision[1:]
        name_classes_no_bg = name_classes[1:]

        # 计算平均mIoU（不含背景）
        miou_no_bg = np.nanmean(IoUs_no_bg)

        print("=" * 50)
        print("包含背景的mIoU:")
        for i in range(num_classes):
            print(f"{name_classes[i]}: {IoUs[i]:.4f}")
        print(f"平均mIoU: {np.nanmean(IoUs):.4f}")

        print("\n" + "=" * 50)
        print("不包含背景的mIoU:")
        for i in range(len(name_classes_no_bg)):
            print(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}")
        print(f"平均mIoU (不含背景): {miou_no_bg:.4f}")

        # 保存结果到文件
        with open(os.path.join(miou_out_path, "miou_results.txt"), "w") as f:
            f.write("包含背景的mIoU:\n")
            for i in range(num_classes):
                f.write(f"{name_classes[i]}: {IoUs[i]:.4f}\n")
            f.write(f"平均mIoU: {np.nanmean(IoUs):.4f}\n\n")

            f.write("不包含背景的mIoU:\n")
            for i in range(len(name_classes_no_bg)):
                f.write(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}\n")
            f.write(f"平均mIoU (不含背景): {miou_no_bg:.4f}\n")

        print("Get miou done.")