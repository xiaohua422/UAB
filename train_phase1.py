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

    Cuda = True
    seed = 11
    distributed = False
    sync_bn = False
    fp16 = False
    num_classes = 13
    backbone = "resnet50"
    pretrained = True
    model_path = ''
    downsample_factor = 16
    input_shape = [512, 512]

    Init_Epoch = 400
    Freeze_Epoch = 400
    Freeze_batch_size = 8
    UnFreeze_Epoch = 600
    Unfreeze_batch_size = 4
    Freeze_Train = False

    if backbone == "resnet50":
        Init_lr = 1e-4
    elif backbone == "mobilenet":
        Init_lr = 5e-3
    else:
        Init_lr = 1e-3

    Min_lr = Init_lr * 0.01
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 5e-5
    lr_decay_type = 'cos'

    save_period = 10
    save_dir = 'logs_phase1'
    eval_flag = True
    eval_period = 5

    use_combined_loss = True
    loss_weights = {
        'ce': 0.5,
        'dice': 2.0,
        'boundary': 0.3
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


def main():
    config = Phase1TrainingConfig()

    print("=" * 70)
    print("Enhanced Training Pipeline")
    print("Improvements:")
    print("  1. Backbone: ResNet50")
    print("  2. Unified attention module")
    print("  3. Combined loss (CE + Dice + Boundary)")
    print("=" * 70)

    seed_everything(config.seed)

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

    if config.pretrained and config.backbone != "resnet50":
        if config.distributed:
            if local_rank == 0:
                download_weights(config.backbone)
            dist.barrier()
        else:
            download_weights(config.backbone)

    model = DeepLab(
        num_classes=config.num_classes,
        backbone=config.backbone,
        downsample_factor=config.downsample_factor,
        pretrained=config.pretrained,
        use_auxiliary=True
    )

    if not config.pretrained and config.model_path == '':
        from nets.deeplabv3_training import weights_init
        weights_init(model)
    else:
        print("Using pretrained weights, skip random initialization")

    if config.model_path != '':
        if local_rank == 0:
            print(f'Loading weights: {config.model_path}')

        try:
            checkpoint = torch.load(config.model_path, map_location=device)

            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model state from checkpoint")
                    if 'optimizer_state_dict' in checkpoint and config.Init_Epoch > 0:
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print(f"Loaded optimizer state")
                    if 'epoch' in checkpoint:
                        print(f"Last trained epoch: {checkpoint['epoch']}")
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                    print(f"Loaded model state (state_dict format)")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"Loaded model state (direct dict)")
            else:
                model.load_state_dict(checkpoint)
                print(f"Loaded model state (direct state dict)")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Fallback to legacy loading...")

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
                print(f"\nSuccessfully loaded: {len(load_key)} keys")
                print(f"Failed to load: {len(no_load_key)} keys")
                if len(no_load_key) > 0:
                    print("Failed keys example:", str(no_load_key[:5]))

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(config.save_dir, f"phase1_{config.backbone}_{time_str}")
        loss_history = LossHistory(log_dir, model, input_shape=config.input_shape)
    else:
        loss_history = None

    if config.fp16:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if config.sync_bn and ngpus_per_node > 1 and config.distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif config.sync_bn:
        print("Sync_bn is not supported in single GPU or non-distributed mode.")

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

    UnFreeze_flag = False

    if config.Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = config.Freeze_batch_size if config.Freeze_Train else config.Unfreeze_batch_size

    nbs = 16
    lr_limit_max = 5e-4 if config.optimizer_type == 'adam' else 1e-1
    lr_limit_min = 3e-4 if config.optimizer_type == 'adam' else 5e-4

    if config.backbone == "xception":
        lr_limit_max = 1e-4 if config.optimizer_type == 'adam' else 1e-1
        lr_limit_min = 1e-4 if config.optimizer_type == 'adam' else 5e-4

    Init_lr_fit = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    if config.optimizer_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            Init_lr_fit,
            betas=(config.momentum, 0.999),
            weight_decay=config.weight_decay
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            Init_lr_fit,
            momentum=config.momentum,
            nesterov=True,
            weight_decay=config.weight_decay
        )

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

    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset is too small for training.")

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

    from collections import deque

    EARLY_STOP_START = max(80, config.Freeze_Epoch + 10)
    EARLY_STOP_PATIENCE = 100
    EARLY_STOP_MIN_DELTA = 0.005
    MIOU_SMOOTH_WINDOW = 5

    miou_queue = deque(maxlen=MIOU_SMOOTH_WINDOW)
    best_avg_miou = 0.0
    early_stop_counter = 0

    for epoch in range(config.Init_Epoch, config.UnFreeze_Epoch):

        if epoch < 15:
            config.loss_weights['boundary'] = 0.0
        elif epoch < 40:
            config.loss_weights['boundary'] = 0.2
        else:
            config.loss_weights['boundary'] = 0.3

        if epoch >= config.Freeze_Epoch and not UnFreeze_flag and config.Freeze_Train:
            batch_size = config.Unfreeze_batch_size

            Init_lr_fit = min(max(batch_size / nbs * config.Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * config.Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            for param_group in optimizer.param_groups:
                param_group['lr'] = Init_lr_fit

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

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size

            if config.distributed:
                batch_size = batch_size // ngpus_per_node

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
            print(f"\n[Unfreeze] Backbone unfrozen at epoch {epoch}, lr={Init_lr_fit:.6f}")

        if config.distributed:
            train_sampler.set_epoch(epoch)

        if config.use_warmup and epoch < config.warmup_epochs:
            warmup_lr = Init_lr_fit * config.warmup_factor * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

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

        if local_rank == 0:
            current_miou = train_results.get('miou', None)

            if config.lr_decay_type == 'plateau' and lr_scheduler is not None:
                lr_scheduler.step(train_results.get('val_loss', 0))
            elif lr_scheduler is not None:
                lr_scheduler.step()

            if local_rank == 0 and current_miou is not None:
                miou_queue.append(current_miou)
                avg_miou = sum(miou_queue) / len(miou_queue)

                if epoch >= EARLY_STOP_START:
                    if avg_miou - best_avg_miou > EARLY_STOP_MIN_DELTA:
                        best_avg_miou = avg_miou
                        early_stop_counter = 0
                        print(f"[EarlyStop] Avg mIoU improved to {best_avg_miou:.4f}")
                    else:
                        early_stop_counter += 1
                        print(
                            f"[EarlyStop] No improvement "
                            f"({early_stop_counter}/{EARLY_STOP_PATIENCE}), "
                            f"Avg mIoU={avg_miou:.4f}"
                        )

                    if early_stop_counter >= EARLY_STOP_PATIENCE:
                        print(
                            f"\n[EarlyStop Triggered] "
                            f"Epoch {epoch}, Best Avg mIoU: {best_avg_miou:.4f}"
                        )
                        break

    if local_rank == 0:
        print("\n" + "=" * 70)
        print("Phase 1 training finished!")
        print(f"Total epochs: {epoch + 1}")
        print(f"Best mIoU: {best_avg_miou:.4f}")
        print(f"Final lr: {optimizer.param_groups[0]['lr']:.6f}")
        print("=" * 70)

        best_model_path = os.path.join(log_dir, "best_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"Best model saved to: {best_model_path}")

        loss_history.writer.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
