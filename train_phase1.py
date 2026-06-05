#改进早停策略
import datetime
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, StepLR

from nets.deeplabv3_plus import DeepLab
from nets.combined_loss import CombinedLoss
from utils.callbacks import LossHistory, ImprovedEvalCallback
from utils.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,
                         worker_init_fn)
from utils.utils_fit import fit_one_epoch


class Phase1TrainingConfig:
    """第一阶段训练配置"""

    # 基础配置
    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False

    # 模型配置
    num_classes = 13
    backbone = "resnet50"  # 第一阶段：使用ResNet50
    pretrained = True
    #pretrained = False
    #model_path = ""  # 从头开始训练
    model_path = 'logs_phase1/phase1_resnet50_2025_12_15_22_02_18/best_model.pth'
    model_path = ''
    downsample_factor = 16
    input_shape = [512, 512]

    # 训练参数
    #Init_Epoch = 0
    Init_Epoch =400  #400轮继续训练到600轮
    #Freeze_Epoch = 50
    Freeze_Epoch =400  #400轮继续训练到600轮
    Freeze_batch_size = 8
    UnFreeze_Epoch = 600  # ResNet收敛更快，减少训练轮数
    Unfreeze_batch_size = 4
    #Freeze_Train = True
    Freeze_Train = False  #400轮继续训练到600轮

    # 优化器参数（ResNet需要更小的学习率）
    if backbone == "resnet50":
        Init_lr = 1e-4  #400轮继续训练到600轮
        # Init_lr = 2e-4  #400轮训练
        # Init_lr = 5e-4 # ResNet学习率较小
        #Init_lr = 1e-3  # ResNet学习率原来设置
    elif backbone == "mobilenet":
        Init_lr = 5e-3
    else:
        Init_lr = 1e-3

    Min_lr = Init_lr * 0.01
    # optimizer_type = "sgd"
    optimizer_type = "adam"
    momentum = 0.9
    # weight_decay = 1e-4
    weight_decay = 5e-5
    lr_decay_type = 'cos'

    # 保存和评估
    save_period = 10
    save_dir = 'logs_phase1'
    eval_flag = True
    eval_period = 5

    # 第一阶段改进：综合损失函数
    use_combined_loss = True
    loss_weights = {
         # 'ce': 1.0,
         # 'dice': 1.0,      # 启用Dice损失
         # 'boundary': 0.5   # 边界损失   这是第二个权重文件配置
    #     'ce':1,
    #     'dice': 0.8,      # 启用Dice损失
    #     'boundary': 0.3   # 边界损失
         # 'ce': 1.0,
         # 'dice': 1.0,      # 启用Dice损失
         # 'boundary': 0.3
         'ce': 0.5 ,
         'dice': 2.0,      # 启用Dice损失
         'boundary': 0.3
     }


    # 数据集路径
    VOCdevkit_path = 'VOCdevkit'
    dice_loss = True      # 启用Dice损失
    focal_loss = False    # 第一阶段不启用Focal Loss
    # cls_weights = np.ones([num_classes], np.float32)
    cls_weights = np.array([
    0.042,   # background
    1.008,
    0.985,
    0.982,
    0.989,
    0.995,
    1.269,
    1.239,
    1.221,
    1.214,
    1.232,
    1.045,
    0.779,
], dtype=np.float32)

    num_workers = 4

    # 自适应早停
    patience_earlystop = 150
    min_improvement = 0.0005

    # 学习率预热
    # use_warmup = True
    use_warmup = False  #400轮继续训练到600轮
    warmup_epochs = 5
    warmup_factor = 0.1




def main():
    """第一阶段训练主函数"""
    config = Phase1TrainingConfig()

    print("=" * 70)
    print("第一阶段改进训练")
    print("改进内容：")
    print("  1. 骨干网络：ResNet50（替代MobileNetV2）")
    print("  2. 统一注意力模块（整合DFF、边界优化、EGA）")
    print("  3. 综合损失函数（CE + Dice + 边界损失）")
    print("=" * 70)

    # 设置随机种子
    seed_everything(config.seed)

    # 设备设置
    ngpus_per_node = torch.cuda.device_count()
    if config.distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # 下载预训练权重
    if config.pretrained and config.backbone != "resnet50":  # ResNet50通过torchvision自动下载
        if config.distributed:
            if local_rank == 0:
                download_weights(config.backbone)
            dist.barrier()
        else:
            download_weights(config.backbone)

    # 创建模型
    model = DeepLab(
        num_classes=config.num_classes,
        backbone=config.backbone,
        downsample_factor=config.downsample_factor,
        pretrained=config.pretrained,
        use_auxiliary=True

    )

    # if not config.pretrained:
    #     from nets.deeplabv3_training import weights_init
    #     weights_init(model)
#======================================400轮训练时使用400-600不使用=================================================
    # 修复：只有当没有预训练权重且没有加载模型时才初始化
    if not config.pretrained and config.model_path == '':
        from nets.deeplabv3_training import weights_init
        weights_init(model)
    else:
        print("使用预训练权重，跳过随机初始化")

    # 加载预训练权重（如果指定）
    if config.model_path != '':
        if local_rank == 0:
            print(f'加载权重: {config.model_path}')

        model_dict = model.state_dict()
        pretrained_dict = torch.load(config.model_path, map_location=device)

        if 'model' in pretrained_dict:
            pretrained_dict = pretrained_dict['model']

        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)

        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print(f"\n成功加载参数: {len(load_key)}个")
            print(f"失败加载参数: {len(no_load_key)}个")
            if len(no_load_key) > 0:
                print("失败参数示例:", str(no_load_key[:5]))

#==========================================================================================================


#======================================训练400-600时使用=================================================
# 加载预训练权重（这里加载你之前训练好的模型）
    if config.model_path != '':
        if local_rank == 0:
            print(f'加载权重: {config.model_path}')

        try:
            # 尝试加载完整的模型状态
            checkpoint = torch.load(config.model_path, map_location=device)

            # 检查checkpoint的类型
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    # 如果是包含模型和优化器状态的checkpoint
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"加载模型状态 (包含在checkpoint中)")

                    # 可选：也加载优化器状态（如果保存了的话）
                    if 'optimizer_state_dict' in checkpoint and config.Init_Epoch > 0:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"加载优化器状态 (从第{config.Init_Epoch}轮)")

                    if 'epoch' in checkpoint:
                        print(f"上次训练结束轮数: {checkpoint['epoch']}")

                elif 'state_dict' in checkpoint:
                    # 另一种常见的checkpoint格式
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"加载模型状态 (state_dict格式)")

                else:
                    # 假设整个文件就是模型状态字典
                    model.load_state_dict(checkpoint)
                    print(f"加载模型状态 (直接状态字典)")

            else:
                # 文件直接就是模型状态字典
                model.load_state_dict(checkpoint)
                print(f"加载模型状态 (直接状态字典)")

        except Exception as e:
            print(f"加载模型时出错: {e}")
            print("尝试使用旧格式加载...")

            # 旧格式的加载方式
            model_dict = model.state_dict()
            pretrained_dict = torch.load(config.model_path, map_location=device)

            # 如果checkpoint有'model'键
            if 'model' in pretrained_dict:
                pretrained_dict = pretrained_dict['model']

            load_key, no_load_key, temp_dict = [], [], {}
            for k, v in pretrained_dict.items():
                if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                    temp_dict[k] = v
                    load_key.append(k)
                else:
                    no_load_key.append(k)

            model_dict.update(temp_dict)
            model.load_state_dict(model_dict)

            if local_rank == 0:
                print(f"\n成功加载参数: {len(load_key)}个")
                print(f"失败加载参数: {len(no_load_key)}个")
                if len(no_load_key) > 0:
                    print("失败参数示例:", str(no_load_key[:5]))
#==============================================================================================================


    # 记录Loss
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(config.save_dir, f"phase1_{config.backbone}_{time_str}")
        loss_history = LossHistory(log_dir, model, input_shape=config.input_shape)
    else:
        loss_history = None

    # 混合精度训练
    if config.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    # 模型训练模式
    model_train = model.train()

    if config.sync_bn and ngpus_per_node > 1 and config.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif config.sync_bn:
        print("Sync_bn在单GPU或非分布式模式下不支持。")

    if config.Cuda:
        if config.distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train,
                device_ids=[local_rank],
                find_unused_parameters=True
            )
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # 读取数据集
    with open(os.path.join(config.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(config.VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"), "r") as f:
        val_lines = f.readlines()

    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=config.num_classes,
            backbone=config.backbone,
            model_path=config.model_path,
            input_shape=config.input_shape,
            Init_Epoch=config.Init_Epoch,
            Freeze_Epoch=config.Freeze_Epoch,
            UnFreeze_Epoch=config.UnFreeze_Epoch,
            Freeze_batch_size=config.Freeze_batch_size,
            Unfreeze_batch_size=config.Unfreeze_batch_size,
            Freeze_Train=config.Freeze_Train,
            Init_lr=config.Init_lr,
            Min_lr=config.Min_lr,
            optimizer_type=config.optimizer_type,
            momentum=config.momentum,
            lr_decay_type=config.lr_decay_type,
            save_period=config.save_period,
            save_dir=config.save_dir,
            num_workers=config.num_workers,
            num_train=num_train,
            num_val=num_val
        )

    # 冻结训练
    UnFreeze_flag = False

    if config.Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = config.Freeze_batch_size if config.Freeze_Train else config.Unfreeze_batch_size

    # 自适应调整学习率
    nbs = 16
    lr_limit_max = 5e-4 if config.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if config.optimizer_type == 'adam' else 5e-4

    if config.backbone == "xception":
        lr_limit_max = 1e-4 if config.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if config.optimizer_type == 'adam' else 5e-4

    Init_lr_fit = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 选择优化器
    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            Init_lr_fit,
            betas=(config.momentum, 0.999),
            weight_decay=config.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            Init_lr_fit,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=config.weight_decay
        )

    # 学习率调度器
    if config.lr_decay_type == 'cos':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.UnFreeze_Epoch, eta_min=Min_lr_fit)
    elif config.lr_decay_type == 'plateau':
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=Min_lr_fit,
            verbose=True
        )
    else:
        lr_scheduler = None

    # 判断每个epoch的长度
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

    # 创建数据集
    train_dataset = DeeplabDataset(
        train_lines,
        config.input_shape,
        config.num_classes,
        True,
        config.VOCdevkit_path
    )
    val_dataset = DeeplabDataset(
        val_lines,
        config.input_shape,
        config.num_classes,
        False,
        config.VOCdevkit_path
    )

    if config.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle = True

    # 创建数据加载器
    gen = DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=train_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=config.seed)
    )

    gen_val = DataLoader(
        val_dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=deeplab_dataset_collate,
        sampler=val_sampler,
        worker_init_fn=partial(worker_init_fn, rank=rank, seed=config.seed)
    )

    # 创建回调
    if local_rank == 0:
        eval_callback = ImprovedEvalCallback(
            model,
            config.input_shape,
            config.num_classes,
            val_lines,
            config.VOCdevkit_path,
            log_dir,
            config.Cuda,
            eval_flag=config.eval_flag,
            period=config.eval_period
        )
    else:
        eval_callback = None

    # # 训练监控变量
    # best_miou = 0.0
    # patience_counter = 0

    from collections import deque
    #修改早停策略新加的
    # ================= Early Stopping 配置 =================
    EARLY_STOP_START = max(80, config.Freeze_Epoch + 10)
    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_MIN_DELTA = 0.005
    MIOU_SMOOTH_WINDOW = 5

    miou_queue = deque(maxlen=MIOU_SMOOTH_WINDOW)
    best_avg_miou = 0.0
    early_stop_counter = 0
    # ======================================================

    # 开始训练
    for epoch in range(config.Init_Epoch, config.UnFreeze_Epoch):

        # ================== 动态调整 Boundary Loss 权重 ==================
        if epoch < 15:
            config.loss_weights['boundary'] = 0.0
        elif epoch < 40:
            config.loss_weights['boundary'] = 0.2
        else:
            config.loss_weights['boundary'] = 0.3
        # =================================================================
        # 解冻逻辑
        if epoch >= config.Freeze_Epoch and not UnFreeze_flag and config.Freeze_Train:
            batch_size = config.Unfreeze_batch_size

            # 重新计算学习率
            Init_lr_fit = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            # 更新优化器学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = Init_lr_fit

            # 重新创建学习率调度器
            if config.lr_decay_type == 'cos':
                lr_scheduler = CosineAnnealingLR(optimizer, T_max=config.UnFreeze_Epoch - epoch, eta_min=Min_lr_fit)
            elif config.lr_decay_type == 'plateau':
                lr_scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=10,
                    min_lr=Min_lr_fit,
                    verbose=True
                )

            # 解冻主干网络
            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if config.distributed:
                batch_size = batch_size // ngpus_per_node

            # 重新创建数据加载器
            gen = DataLoader(
                train_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=train_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=config.seed)
            )

            gen_val = DataLoader(
                val_dataset,
                shuffle=shuffle,
                batch_size=batch_size,
                num_workers=config.num_workers,
                pin_memory=True,
                drop_last=True,
                collate_fn=deeplab_dataset_collate,
                sampler=val_sampler,
                worker_init_fn=partial(worker_init_fn, rank=rank, seed=config.seed)
            )

            UnFreeze_flag = True
            print(f"\n[解冻] 第{epoch}轮解冻主干网络，学习率重置为{Init_lr_fit:.6f}")

        if config.distributed:
            train_sampler.set_epoch(epoch)

        # 学习率预热
        if config.use_warmup and epoch < config.warmup_epochs:
            warmup_lr = Init_lr_fit * config.warmup_factor * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # 训练一轮
        train_results = fit_one_epoch(
            model_train=model_train,
            model=model,
            loss_history=loss_history,
            eval_callback=eval_callback,
            optimizer=optimizer,
            epoch=epoch,
            epoch_step=epoch_step,
            epoch_step_val=epoch_step_val,
            gen=gen,
            gen_val=gen_val,
            Epoch=config.UnFreeze_Epoch,
            cuda=config.Cuda,
            dice_loss=config.dice_loss,
            focal_loss=config.focal_loss,
            cls_weights=config.cls_weights,
            num_classes=config.num_classes,
            fp16=config.fp16,
            scaler=scaler,
            save_period=config.save_period,
            save_dir=log_dir if local_rank == 0 else config.save_dir,
            local_rank=local_rank,
            use_combined_loss=config.use_combined_loss,
            loss_weights=config.loss_weights
        )

        # 监控逻辑
        if local_rank == 0:
            # current_miou = train_results.get('val_miou', 0.0)
            current_miou = train_results.get('miou', None)


            # 更新学习率调度器
            if config.lr_decay_type == 'plateau' and lr_scheduler is not None:
                lr_scheduler.step(train_results.get('val_loss', 0))
            elif lr_scheduler is not None:
                lr_scheduler.step()

            # # 早停判断
            # if current_miou - best_miou > config.min_improvement:
            #     best_miou = current_miou
            #     patience_counter = 0
            #     print(f"[改进] mIoU提升至 {best_miou:.4f}")
            # else:
            #     patience_counter += 1
            #     print(f"[耐心] 连续 {patience_counter}/{config.patience_earlystop} 轮无改进")

            # # 触发早停
            # if patience_counter >= config.patience_earlystop:
            #     print(f"\n[早停] 连续{config.patience_earlystop}轮无显著改进，停止训练")
            #     print(f"最佳mIoU: {best_miou:.4f}")
            #     break
            # ================= 改进 Early Stopping（医学分割适配） =================
            if local_rank == 0 and current_miou is not None:
                miou_queue.append(current_miou)
                avg_miou = sum(miou_queue) / len(miou_queue)

                if epoch >= EARLY_STOP_START:
                    if avg_miou - best_avg_miou > EARLY_STOP_MIN_DELTA:
                        best_avg_miou = avg_miou
                        early_stop_counter = 0
                        print(f"[EarlyStop] Avg mIoU 提升至 {best_avg_miou:.4f}")
                    else:
                        early_stop_counter += 1
                        print(
                            f"[EarlyStop] 无显著提升 "
                            f"({early_stop_counter}/{EARLY_STOP_PATIENCE}), "
                            f"Avg mIoU={avg_miou:.4f}"
                        )

                    if early_stop_counter >= EARLY_STOP_PATIENCE:
                        print(
                            f"\n[EarlyStop Triggered] "
                            f"Epoch {epoch}, Best Avg mIoU: {best_avg_miou:.4f}"
                        )
                        break
            # =====================================================================

    # 训练结束
    if local_rank == 0:
        print("\n" + "=" * 70)
        print("第一阶段训练完成!")
        print(f"总训练轮数: {epoch + 1}")
        print(f"最佳mIoU: {best_avg_miou:.4f}")
        print(f"最终学习率: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 70)

        # 保存唯一的最佳模型（覆盖旧文件）
        best_model_path = os.path.join(log_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"[保存] best_model.pth 已更新（mIoU={best_avg_miou:.4f}）")

        loss_history.writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)