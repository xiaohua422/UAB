'''
# 训练完整模型
python train_ablation.py --config_name full_model --gpu_id 0

# 训练无注意力
python train_ablation.py --config_name no_attention --gpu_id 0

# 训练仅通道注意力
python train_ablation.py --config_name channel_only --gpu_id 0

# 训练仅空间注意力
python train_ablation.py --config_name spatial_only --gpu_id 0

# 训练无 shortcut conv
python train_ablation.py --config_name no_shortcut --gpu_id 0

# 训练无特征融合
python train_ablation.py --config_name no_fusion --gpu_id 0
'''
import datetime
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from nets.deeplabv3_plus import DeepLab
from nets.combined_loss import CombinedLoss
from utils.callbacks import LossHistory, ImprovedEvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="消融实验训练脚本")
    parser.add_argument("--config_name", type=str, required=True,
                        choices=['full_model', 'no_attention', 'channel_only',
                                 'spatial_only', 'no_shortcut', 'no_fusion'],
                        help="消融配置名称")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    return parser.parse_args()


# 根据配置名称返回对应的模型参数和损失权重
def get_config_by_name(config_name):
    base_config = {
        "use_attention": True,
        "attention_type": "unified",
        "use_shortcut_conv": True,
        "use_feature_fusion": True,
        "use_cls_conv": True,
        "use_auxiliary_head": True,
        "use_boundary_guidance": True,
        "loss_boundary_weight": 0.3,
    }

    if config_name == "full_model":
        pass  # 使用默认
    elif config_name == "no_attention":
        base_config.update({
            "use_attention": False,
            "attention_type": "none",
            "use_boundary_guidance": False,
            "loss_boundary_weight": 0.0,
        })
    elif config_name == "channel_only":
        base_config.update({
            "use_attention": True,
            "attention_type": "channel_only",
            "use_boundary_guidance": False,
            "loss_boundary_weight": 0.0,
        })
    elif config_name == "spatial_only":
        base_config.update({
            "use_attention": True,
            "attention_type": "spatial_only",
            "use_boundary_guidance": False,
            "loss_boundary_weight": 0.0,
        })
    elif config_name == "no_shortcut":
        base_config.update({
            "use_shortcut_conv": False,
        })
    elif config_name == "no_fusion":
        base_config.update({
            "use_feature_fusion": False,
        })
    else:
        raise ValueError(f"Unknown config: {config_name}")

    return base_config


def main():
    args = parse_args()
    config_name = args.config_name
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # 获取模型配置
    model_config = get_config_by_name(config_name)

    # ========== 基本训练参数（与原 Phase1TrainingConfig 保持一致） ==========
    num_classes = 13
    backbone = "resnet50"
    pretrained = True
    model_path = ''  # 如果要从某个预训练继续，可设置
    downsample_factor = 16
    input_shape = [512, 512]

    Init_Epoch = 0
    Freeze_Epoch = 50
    UnFreeze_Epoch = 200
    Freeze_batch_size = 8
    Unfreeze_batch_size = 4
    Freeze_Train = False

    Init_lr = 1e-4
    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 5e-5
    lr_decay_type = 'cos'

    save_period = 10
    eval_flag = True
    eval_period = 5

    use_combined_loss = True
    # 损失权重（根据配置动态调整 boundary 权重）
    loss_weights = {
        'ce': 0.5,
        'dice': 2.0,
        'boundary': model_config['loss_boundary_weight'],
    }

    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True
    focal_loss = False
    cls_weights = np.array([
        0.042, 1.008, 0.985, 0.982, 0.989, 0.995,
        1.269, 1.239, 1.221, 1.214, 1.232, 1.045, 0.779
    ], dtype=np.float32)
    num_workers = 4

    patience_earlystop = 150
    min_improvement = 0.0005
    use_warmup = False
    warmup_epochs = 5
    warmup_factor = 0.1

    # 保存目录
    save_dir = f"logs_ablation/{config_name}"
    os.makedirs(save_dir, exist_ok=True)

    # ========== 设置设备 ==========
    Cuda = True
    distributed = False
    sync_bn = False
    fp16 = False
    seed = 11
    seed_everything(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_rank = 0
    rank = 0

    # ========== 创建模型 ==========
    model = DeepLab(
        num_classes=num_classes,
        backbone=backbone,
        downsample_factor=downsample_factor,
        pretrained=pretrained,
        use_auxiliary=model_config['use_auxiliary_head'],
        use_attention=model_config['use_attention'],
        attention_type=model_config['attention_type'],
        use_shortcut_conv=model_config['use_shortcut_conv'],
        use_feature_fusion=model_config['use_feature_fusion'],
        use_cls_conv=model_config['use_cls_conv'],
        use_boundary_guidance=model_config['use_boundary_guidance'],
    )

    # 加载预训练权重（如果指定）
    if model_path != '':
        print(f'加载权重: {model_path}')
        state_dict = torch.load(model_path, map_location=device)
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict, strict=False)

    # 记录 Loss
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, f"ablation_{config_name}_{time_str}")
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    # 混合精度
    scaler = None
    if fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    # 多GPU
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

    # ========== 数据加载 ==========
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    nbs = 16
    lr_limit_max = 5e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True,
                              weight_decay=weight_decay)

    if lr_decay_type == 'cos':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=UnFreeze_Epoch, eta_min=Min_lr_fit)
    elif lr_decay_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=Min_lr_fit,
                                         verbose=True)
    else:
        lr_scheduler = None

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = DeeplabDataset(train_lines, input_shape, num_classes, True, VOCdevkit_path)
    val_dataset = DeeplabDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)

    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)
    gen_val = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=deeplab_dataset_collate)

    eval_callback = ImprovedEvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path,
                                         log_dir, Cuda, eval_flag=eval_flag, period=eval_period)

    # ========== 训练循环 ==========
    best_avg_miou = 0.0
    early_stop_counter = 0
    EARLY_STOP_START = max(80, Freeze_Epoch + 10)
    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_MIN_DELTA = 0.005
    from collections import deque
    miou_queue = deque(maxlen=5)

    UnFreeze_flag = False
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        # 动态调整边界损失权重（可选，与原脚本一致）
        if epoch < 15:
            loss_weights['boundary'] = 0.0
        elif epoch < 40:
            loss_weights['boundary'] = 0.2
        else:
            loss_weights['boundary'] = model_config['loss_boundary_weight']

        # 解冻逻辑
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            # ... (与原脚本相同，此处省略，可参考原脚本)
            pass

        train_results = fit_one_epoch(
            model_train=model_train, model=model, loss_history=loss_history, eval_callback=eval_callback,
            optimizer=optimizer, epoch=epoch, epoch_step=epoch_step, epoch_step_val=epoch_step_val,
            gen=gen, gen_val=gen_val, Epoch=UnFreeze_Epoch, cuda=Cuda,
            dice_loss=dice_loss, focal_loss=focal_loss, cls_weights=cls_weights,
            num_classes=num_classes, fp16=fp16, scaler=scaler, save_period=save_period,
            save_dir=log_dir, local_rank=local_rank, use_combined_loss=use_combined_loss,
            loss_weights=loss_weights
        )

        # 学习率调度
        if lr_scheduler is not None:
            if lr_decay_type == 'plateau':
                lr_scheduler.step(train_results.get('val_loss', 0))
            else:
                lr_scheduler.step()

        # 早停逻辑
        current_miou = train_results.get('miou', None)
        if current_miou is not None:
            miou_queue.append(current_miou)
            avg_miou = sum(miou_queue) / len(miou_queue)
            if epoch >= EARLY_STOP_START:
                if avg_miou - best_avg_miou > EARLY_STOP_MIN_DELTA:
                    best_avg_miou = avg_miou
                    early_stop_counter = 0
                    print(f"[EarlyStop] Avg mIoU 提升至 {best_avg_miou:.4f}")
                else:
                    early_stop_counter += 1
                    print(f"[EarlyStop] 无显著提升 ({early_stop_counter}/{EARLY_STOP_PATIENCE}), Avg mIoU={avg_miou:.4f}")
                if early_stop_counter >= EARLY_STOP_PATIENCE:
                    print(f"\n[EarlyStop Triggered] Epoch {epoch}, Best Avg mIoU: {best_avg_miou:.4f}")
                    break

    # 保存最佳模型
    best_model_path = os.path.join(log_dir, "best_model.pth")
    torch.save(model.state_dict(), best_model_path)
    print(f"训练完成，最佳模型保存至 {best_model_path}")
    loss_history.writer.close()


if __name__ == "__main__":
    main()