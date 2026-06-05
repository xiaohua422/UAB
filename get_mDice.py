import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results


# 新增函数：计算mDice系数
def compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes):
    """
    计算mDice系数
    """
    print("Calculate mDice...")
    hist = np.zeros((num_classes, num_classes))

    # 创建存储每个类别Dice系数的数组
    dice_per_class = np.zeros(num_classes)

    for image_id in tqdm(image_ids):
        # 从文件读取真实标签和预测结果
        label_path = os.path.join(gt_dir, image_id + ".png")
        pred_path = os.path.join(pred_dir, image_id + ".png")

        label = Image.open(label_path)
        pred = Image.open(pred_path)

        # 转换为numpy数组
        label = np.array(label)
        pred = np.array(pred)

        # 如果图像是单通道，确保形状一致
        if len(label.shape) == 3:
            label = label[:, :, 0]
        if len(pred.shape) == 3:
            pred = pred[:, :, 0]

        # 展平数组
        label = label.flatten()
        pred = pred.flatten()

        # 计算每个类别的Dice系数
        for cls in range(num_classes):
            # 获取当前类别的二值掩码
            gt_mask = (label == cls).astype(np.int32)
            pred_mask = (pred == cls).astype(np.int32)

            # 计算交集和并集
            intersection = np.sum(gt_mask * pred_mask)
            union = np.sum(gt_mask) + np.sum(pred_mask)

            # 计算Dice系数（添加平滑项避免除零）
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

            # 累加当前类别的Dice系数
            dice_per_class[cls] += dice

    # 计算每个类别的平均Dice系数
    dice_per_class /= len(image_ids)

    # 计算mDice（所有类别的平均值）
    mdice = np.mean(dice_per_class)

    return mdice, dice_per_class


# 修改后的函数：显示mDice结果并保存为TXT和PNG
def show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True):
    """
    显示mDice结果并保存为TXT和PNG格式
    """
    # 确保输出目录存在
    if not os.path.exists(miou_out_path):
        os.makedirs(miou_out_path)

    # 根据是否包含背景选择数据
    if include_background:
        dice_data = dice_per_class
        class_names = name_classes
        title_suffix = " (包含背景)"
        filename_suffix = ""
    else:
        dice_data = dice_per_class[1:]  # 排除背景
        class_names = name_classes[1:]  # 排除背景
        title_suffix = " (不包含背景)"
        filename_suffix = "_no_bg"

    # 计算mDice
    mdice = np.mean(dice_data)

    # 打印每个类别的Dice系数
    print(f"Dice coefficients for each class{title_suffix}:")
    for i in range(len(class_names)):
        print(f"{class_names[i]}: {dice_data[i]:.4f}")

    # 打印mDice
    print(f"mDice{title_suffix}: {mdice:.4f}")

    # 将结果保存到TXT文件
    with open(os.path.join(miou_out_path, f"dice_results{filename_suffix}.txt"), "w") as f:
        f.write(f"Dice coefficients for each class{title_suffix}:\n")
        for i in range(len(class_names)):
            f.write(f"{class_names[i]}: {dice_data[i]:.4f}\n")
        f.write(f"mDice{title_suffix}: {mdice:.4f}\n")

    # 创建并保存柱状图PNG
    plt.figure(figsize=(12, 8))

    # 设置颜色 - 为不同类别使用不同颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))

    # 创建柱状图
    bars = plt.bar(range(len(class_names)), dice_data, color=colors)

    # 设置标题和标签
    plt.title(f'Dice Coefficients by Class{title_suffix} (mDice: {mdice:.4f})', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Dice Coefficient', fontsize=14)

    # 设置x轴标签
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')

    # 在每个柱子上方添加数值标签
    for i, v in enumerate(dice_data):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

    # 添加水平参考线
    plt.axhline(y=mdice, color='r', linestyle='--', alpha=0.7, label=f'mDice: {mdice:.4f}')

    # 设置y轴范围
    plt.ylim(0, 1.1)

    # 添加图例
    plt.legend()

    # 调整布局以防止标签被截断
    plt.tight_layout()

    # 保存图像
    plt.savefig(os.path.join(miou_out_path, f"dice_results{filename_suffix}.png"), dpi=300, bbox_inches='tight')

    # 关闭图形以释放内存
    plt.close()

    print(
        f"Results saved to {os.path.join(miou_out_path, f'dice_results{filename_suffix}.txt')} and {os.path.join(miou_out_path, f'dice_results{filename_suffix}.png')}")


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    #   新增：miou_mode为3代表计算mDice。
    # ---------------------------------------------------------------------------#
    miou_mode = 3
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 13
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["_background_", "L1", "L2", "L3", "L4", "L5", "L1/L2", "L2/L3", "L3/L4", "L4/L5", "L5/S1", "S1",
                    "CSF"]

    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
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

    # 新增：计算mDice模式
    if miou_mode == 3:
        print("Get mDice.")
        mdice, dice_per_class = compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get mDice done.")

        # 显示包含背景的mDice结果
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True)

        # 显示不包含背景的mDice结果
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=False)

    # 同时计算mIoU和mDice
    if miou_mode == 4:
        print("Get miou and mDice.")
        # 计算mIoU
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        # 计算不包含背景的mIoU
        IoUs_no_bg = IoUs[1:]
        PA_Recall_no_bg = PA_Recall[1:]
        Precision_no_bg = Precision[1:]
        name_classes_no_bg = name_classes[1:]
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

        # 保存mIoU结果到文件
        with open(os.path.join(miou_out_path, "miou_results.txt"), "w") as f:
            f.write("包含背景的mIoU:\n")
            for i in range(num_classes):
                f.write(f"{name_classes[i]}: {IoUs[i]:.4f}\n")
            f.write(f"平均mIoU: {np.nanmean(IoUs):.4f}\n\n")

            f.write("不包含背景的mIoU:\n")
            for i in range(len(name_classes_no_bg)):
                f.write(f"{name_classes_no_bg[i]}: {IoUs_no_bg[i]:.4f}\n")
            f.write(f"平均mIoU (不含背景): {miou_no_bg:.4f}\n")

        # 计算mDice
        mdice, dice_per_class = compute_mDice(gt_dir, pred_dir, image_ids, num_classes, name_classes)

        # 显示包含背景的mDice结果
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=True)

        # 显示不包含背景的mDice结果
        show_dice_results(miou_out_path, dice_per_class, name_classes, include_background=False)

        print("Get miou and mDice done.")