import os
import torch
import numpy as np
from tqdm import tqdm
from nets.deeplabv3_training import (
    CE_Loss, Dice_loss, Focal_Loss, 
    get_loss_function, calculate_metrics, LossRecorder
)
from utils.utils import get_lr

def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                  epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, 
                  dice_loss, focal_loss, cls_weights, num_classes, 
                  fp16, scaler, save_period, save_dir, local_rank=0,
                  use_combined_loss=True, loss_weights=None):
    """
    改进的训练和验证循环
    Args:
        use_combined_loss: 是否使用综合损失函数
        loss_weights: 综合损失函数的权重
    """
    total_loss = 0
    total_f_score = 0
    train_recorder = LossRecorder()
    
    val_loss = 0
    val_f_score = 0
    val_recorder = LossRecorder()
    
    # 获取损失函数
    loss_function = get_loss_function(
        num_classes=num_classes,
        use_combined_loss=use_combined_loss,
        loss_weights=loss_weights,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=cls_weights
    )
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    model_train.train()
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        imgs, pngs, labels = batch

        with torch.no_grad():
            weights = torch.from_numpy(cls_weights) if cls_weights is not None else None
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                if weights is not None:
                    weights = weights.cuda(local_rank)
        
        # 清零梯度
        optimizer.zero_grad()
        
        if not fp16:
            # 前向传播
            outputs = model_train(imgs)
            
            # 计算损失
            if isinstance(outputs, tuple):
                # 处理主输出和辅助输出
                main_output, aux_output = outputs
                
                if use_combined_loss and hasattr(loss_function, '__call__'):
                    # 使用综合损失函数
                    total_main_loss, main_loss_dict = loss_function(main_output, pngs)
                    
                    # 辅助损失
                    aux_loss = CE_Loss(aux_output, pngs, weights, num_classes=num_classes)
                    total_loss_value = total_main_loss + 0.4 * aux_loss
                    
                    # 记录损失
                    loss_dict = main_loss_dict.copy()
                    loss_dict['auxiliary'] = aux_loss.item()
                    loss_dict['total'] = total_loss_value.item()
                else:
                    # 使用旧版损失函数
                    if focal_loss:
                        main_loss = Focal_Loss(main_output, pngs, weights, num_classes=num_classes)
                        aux_loss = Focal_Loss(aux_output, pngs, weights, num_classes=num_classes)
                    else:
                        main_loss = CE_Loss(main_output, pngs, weights, num_classes=num_classes)
                        aux_loss = CE_Loss(aux_output, pngs, weights, num_classes=num_classes)
                    
                    total_loss_value = main_loss + 0.4 * aux_loss
                    loss_dict = {
                        'total': total_loss_value.item(),
                        'main': main_loss.item(),
                        'auxiliary': aux_loss.item()
                    }
                    
                    if dice_loss:
                        main_dice = Dice_loss(main_output, labels)
                        aux_dice = Dice_loss(aux_output, labels)
                        dice_loss_value = main_dice + 0.4 * aux_dice
                        total_loss_value = total_loss_value + dice_loss_value
                        loss_dict['dice'] = dice_loss_value.item()
            else:
                # 只有主输出
                if use_combined_loss and hasattr(loss_function, '__call__'):
                    total_loss_value, loss_dict = loss_function(outputs, pngs)
                else:
                    if focal_loss:
                        total_loss_value = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        total_loss_value = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    
                    loss_dict = {'total': total_loss_value.item()}
                    
                    if dice_loss:
                        dice_loss_value = Dice_loss(outputs, labels)
                        total_loss_value = total_loss_value + dice_loss_value
                        loss_dict['dice'] = dice_loss_value.item()

            # 反向传播
            total_loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                # 前向传播
                outputs = model_train(imgs)
                
                # 计算损失
                if isinstance(outputs, tuple):
                    main_output, aux_output = outputs
                    
                    if focal_loss:
                        main_loss = Focal_Loss(main_output, pngs, weights, num_classes=num_classes)
                        aux_loss = Focal_Loss(aux_output, pngs, weights, num_classes=num_classes)
                    else:
                        main_loss = CE_Loss(main_output, pngs, weights, num_classes=num_classes)
                        aux_loss = CE_Loss(aux_output, pngs, weights, num_classes=num_classes)
                    
                    total_loss_value = main_loss + 0.4 * aux_loss
                    loss_dict = {
                        'total': total_loss_value.item(),
                        'main': main_loss.item(),
                        'auxiliary': aux_loss.item()
                    }
                    
                    if dice_loss:
                        main_dice = Dice_loss(main_output, labels)
                        aux_dice = Dice_loss(aux_output, labels)
                        dice_loss_value = main_dice + 0.4 * aux_dice
                        total_loss_value = total_loss_value + dice_loss_value
                        loss_dict['dice'] = dice_loss_value.item()
                else:
                    if focal_loss:
                        total_loss_value = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                    else:
                        total_loss_value = CE_Loss(outputs, pngs, weights, num_classes=num_classes)
                    
                    loss_dict = {'total': total_loss_value.item()}
                    
                    if dice_loss:
                        dice_loss_value = Dice_loss(outputs, labels)
                        total_loss_value = total_loss_value + dice_loss_value
                        loss_dict['dice'] = dice_loss_value.item()

            # 反向传播
            scaler.scale(total_loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += total_loss_value.item()
        train_recorder.update(loss_dict)

        # 计算F-score
        from utils.utils_metrics import f_score
        if isinstance(outputs, tuple):
            _f_score = f_score(main_output, labels)
        else:
            _f_score = f_score(outputs, labels)
        
        total_f_score += _f_score.item()

        if local_rank == 0:
            # 更新进度条
            postfix_dict = {
                'total_loss': total_loss / (iteration + 1),
                'f_score': total_f_score / (iteration + 1),
                'lr': get_lr(optimizer)
            }
            
            # 添加具体的损失分量
            if 'ce' in loss_dict:
                postfix_dict['ce_loss'] = loss_dict['ce']
            if 'dice' in loss_dict:
                postfix_dict['dice_loss'] = loss_dict['dice']
            if 'boundary' in loss_dict:
                postfix_dict['boundary_loss'] = loss_dict['boundary']
            
            pbar.set_postfix(**postfix_dict)
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        train_recorder.print_summary(epoch + 1, 'Train')
        
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    model_train.eval()
    
    # 验证阶段
    val_metrics = {
        'miou': [],
        'accuracy': [],
        'dice': []
    }
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights) if cls_weights is not None else None
            if cuda:
                imgs = imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                if weights is not None:
                    weights = weights.cuda(local_rank)

            # 前向传播
            outputs = model_train(imgs)

            # 验证阶段：如果输出是元组，只使用主输出
            if isinstance(outputs, tuple):
                main_output = outputs[0]
            else:
                main_output = outputs

            # 计算损失
            if use_combined_loss and hasattr(loss_function, '__call__'):
                val_loss_value, val_loss_dict = loss_function(main_output, pngs)
            else:
                if focal_loss:
                    val_loss_value = Focal_Loss(main_output, pngs, weights, num_classes=num_classes)
                else:
                    val_loss_value = CE_Loss(main_output, pngs, weights, num_classes=num_classes)
                
                val_loss_dict = {'total': val_loss_value.item()}
                
                if dice_loss:
                    dice_loss_value = Dice_loss(main_output, labels)
                    val_loss_value = val_loss_value + dice_loss_value
                    val_loss_dict['dice'] = dice_loss_value.item()

            # 计算评估指标
            #metrics = calculate_metrics(main_output, torch.argmax(pngs, dim=3), num_classes)
            metrics = calculate_metrics(main_output, pngs.squeeze(1).long(), num_classes)
            for key in val_metrics:
                val_metrics[key].append(metrics[key])

            # 计算f_score
            from utils.utils_metrics import f_score
            _f_score = f_score(main_output, labels)

            val_loss += val_loss_value.item()
            val_f_score += _f_score.item()
            val_recorder.update(val_loss_dict)

            if local_rank == 0:
                # 更新验证进度条
                postfix_dict = {
                    'val_loss': val_loss / (iteration + 1),
                    'val_f_score': val_f_score / (iteration + 1),
                    'val_miou': np.mean(val_metrics['miou']),
                    'lr': get_lr(optimizer)
                }
                pbar.set_postfix(**postfix_dict)
                pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        # 计算平均指标
        avg_train_loss = total_loss / epoch_step
        avg_val_loss = val_loss / epoch_step_val
        avg_val_f_score = val_f_score / epoch_step_val
        
        avg_miou = np.mean(val_metrics['miou'])
        avg_accuracy = np.mean(val_metrics['accuracy'])
        avg_dice = np.mean(val_metrics['dice'])
        
        # 更新损失历史
        if loss_history is not None:
            loss_history.append_loss(epoch + 1, avg_train_loss, avg_val_loss)
        
        # 调用回调函数
        if eval_callback is not None:
            eval_callback.on_epoch_end(
                epoch + 1, 
                model_train,
                current_miou=avg_miou,
                val_loss=avg_val_loss,
                val_f_score=avg_val_f_score,
                val_dice=avg_dice,
                val_accuracy=avg_accuracy
            )
        
        # 打印训练信息
        print('\n' + '=' * 70)
        print(f'Epoch {epoch + 1}/{Epoch}')
        print(f'Total Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        print(f'Val mIoU: {avg_miou:.4f} | Val Dice: {avg_dice:.4f} | Val Acc: {avg_accuracy:.4f}')
        print('=' * 70)
        
        # 保存模型
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            checkpoint_path = os.path.join(save_dir, 
                f'epoch{epoch+1:03d}_loss{avg_train_loss:.3f}_valloss{avg_val_loss:.3f}_miou{avg_miou:.4f}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'[保存] 检查点已保存: {checkpoint_path}')
        
        # 保存最佳模型
        if hasattr(eval_callback, 'best_miou') and avg_miou > eval_callback.best_miou:
            best_model_path = os.path.join(save_dir, f'best_model_epoch{epoch+1}_miou{avg_miou:.4f}.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f'[最佳] 最佳模型已保存: {best_model_path}')
        
        # 总是保存最后一个epoch
        last_model_path = os.path.join(save_dir, "last_epoch_weights.pth")
        torch.save(model.state_dict(), last_model_path)
    
    # 返回训练结果
    # return {
    #     'total_loss': total_loss / epoch_step,
    #     'val_loss': val_loss / epoch_step_val if epoch_step_val > 0 else 0,
    #     'val_f_score': val_f_score / epoch_step_val if epoch_step_val > 0 else 0,
    #     'val_miou': avg_miou if local_rank == 0 else 0,
    #     'val_dice': avg_dice if local_rank == 0 else 0,
    #     'val_accuracy': avg_accuracy if local_rank == 0 else 0
    # }
    return {
        'loss': total_loss / epoch_step,
        'val_loss': val_loss / epoch_step_val if epoch_step_val > 0 else 0,
        'miou': avg_miou if local_rank == 0 else 0,
        'dice': avg_dice if local_rank == 0 else 0,
        'accuracy': avg_accuracy if local_rank == 0 else 0
    }


def fit_one_epoch_simple(model_train, model, loss_history, eval_callback, optimizer, epoch,
                         epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, 
                         dice_loss, focal_loss, cls_weights, num_classes, 
                         fp16, scaler, save_period, save_dir, local_rank=0):
    """
    简化版训练循环（兼容原有调用）
    """
    return fit_one_epoch(
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
        Epoch=Epoch,
        cuda=cuda,
        dice_loss=dice_loss,
        focal_loss=focal_loss,
        cls_weights=cls_weights,
        num_classes=num_classes,
        fp16=fp16,
        scaler=scaler,
        save_period=save_period,
        save_dir=save_dir,
        local_rank=local_rank,
        use_combined_loss=True,  # 默认使用综合损失
        loss_weights={
            'ce': 1.0,
            'dice': 1.0 if dice_loss else 0.0,
            'boundary': 0.5
        }
    )