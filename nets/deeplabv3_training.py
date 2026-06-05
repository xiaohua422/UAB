import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.combined_loss import CombinedLoss, MixedLoss


# def CE_Loss(inputs, target, cls_weights, num_classes=21):
#     """交叉熵损失（兼容原有接口）"""
#     n, c, h, w = inputs.size()
#     nt, ht, wt = target.size()
    
#     if h != ht and w != wt:
#         inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

#     temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
#     temp_target = target.view(-1)

#     # 应用类别权重
#     if cls_weights is not None:
#         weight_tensor = torch.tensor(cls_weights, device=inputs.device)
#         CE_loss = F.cross_entropy(temp_inputs, temp_target, weight=weight_tensor, reduction='mean')
#     else:
#         CE_loss = F.cross_entropy(temp_inputs, temp_target, reduction='mean')
    
#     return CE_loss

def CE_Loss(inputs, target, cls_weights=None, num_classes=21, ignore_index=255):
    """
    Cross Entropy Loss with class weights (医学分割推荐版)
    """
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    if h != ht or w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    # [N, C, H, W] -> [N*H*W, C]
    temp_inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, c)
    temp_target = target.view(-1)

    # ===== 核心修改点 =====
    if cls_weights is not None:
        if not isinstance(cls_weights, torch.Tensor):
            cls_weights = torch.as_tensor(
                cls_weights,
                dtype=temp_inputs.dtype,
                device=temp_inputs.device
            )

        loss = F.cross_entropy(
            temp_inputs,
            temp_target,
            weight=cls_weights,
            ignore_index=ignore_index,
            reduction='mean'
        )
    else:
        loss = F.cross_entropy(
            temp_inputs,
            temp_target,
            ignore_index=ignore_index,
            reduction='mean'
        )

    return loss

def Focal_Loss(inputs, target, cls_weights, num_classes=21, alpha=0.5, gamma=2):
    """Focal Loss（兼容原有接口）"""
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    logpt = -F.cross_entropy(temp_inputs, temp_target, weight=None, reduction='none')
    pt = torch.exp(logpt)
    
    if alpha is not None:
        alpha_tensor = torch.ones(c, device=inputs.device) * alpha
        alpha_tensor[temp_target] = alpha
        logpt = logpt * alpha_tensor[temp_target]
    
    focal_loss = -((1 - pt) ** gamma) * logpt
    
    if cls_weights is not None:
        weight_tensor = torch.tensor(cls_weights, device=inputs.device)
        focal_loss = focal_loss * weight_tensor[temp_target]
    
    focal_loss = focal_loss.mean()
    return focal_loss


def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    """Dice Loss（兼容原有接口）"""
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    tp = torch.sum(temp_target[..., :-1] * temp_inputs, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    
    return dice_loss


def weights_init(net, init_type='normal', init_gain=0.02):
    """权重初始化"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.batch_norm.bias.data, 0.0)
    
    print('initialize network with %s type' % init_type)
    net.apply(init_func)


def get_loss_function(num_classes, use_combined_loss=True, loss_weights=None, 
                     dice_loss=False, focal_loss=False, cls_weights=None):
    """
    获取损失函数（兼容新旧接口）
    Args:
        use_combined_loss: 是否使用新的综合损失函数
        loss_weights: 综合损失函数的权重
    """
    if use_combined_loss:
        # 使用新的综合损失函数
        weights = loss_weights or {
            'ce': 1.0,
            'dice': 1.0 if dice_loss else 0.0,
            'boundary': 0.5
        }
        
        # 过滤权重为0的损失
        filtered_weights = {k: v for k, v in weights.items() if v > 0}
        
        return CombinedLoss(
            num_classes=num_classes,
            weights=filtered_weights
        )
    else:
        # 使用旧的损失函数（兼容）
        def old_loss_function(pred, target, is_focal=focal_loss, weights=cls_weights):
            if is_focal:
                return Focal_Loss(pred, target, weights, num_classes)
            else:
                return CE_Loss(pred, target, weights, num_classes)
        
        return old_loss_function


class LossRecorder:
    """损失记录器，用于记录训练过程中的各种损失"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = {
            'total': [],
            'ce': [],
            'dice': [],
            'boundary': [],
            'auxiliary': []
        }
    
    def update(self, loss_dict):
        for key in loss_dict:
            if key in self.losses:
                self.losses[key].append(loss_dict[key])
    
    def get_average(self, last_n=None):
        averages = {}
        for key, values in self.losses.items():
            if values:
                if last_n:
                    avg_values = values[-last_n:]
                else:
                    avg_values = values
                averages[key] = sum(avg_values) / len(avg_values)
        return averages
    
    def print_summary(self, epoch=None, phase='Train'):
        averages = self.get_average()
        
        if epoch is not None:
            print(f"\n[{phase}] Epoch {epoch} 损失统计:")
        else:
            print(f"\n[{phase}] 损失统计:")
        
        print("-" * 60)
        for key, value in averages.items():
            print(f"  {key:15s}: {value:.6f}")
        print("-" * 60)


def calculate_metrics(pred, target, num_classes, ignore_index=255):
    """计算评估指标：mIoU, Accuracy, Dice等"""
    pred = torch.argmax(pred, dim=1)
    
    # 创建掩码
    mask = (target != ignore_index)
    
    # 计算混淆矩阵
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=pred.device)
    
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = torch.sum((pred == i) & (target == j) & mask)
    
    # 计算每个类别的IoU
    intersection = torch.diag(cm)
    union = cm.sum(dim=1) + cm.sum(dim=0) - torch.diag(cm)
    
    iou = intersection / (union + 1e-8)
    
    # 计算mIoU（忽略背景类0）
    miou = iou[1:].mean()
    
    # 计算准确率
    accuracy = torch.sum(torch.diag(cm)) / torch.sum(cm)
    
    # 计算Dice系数
    dice = (2 * intersection) / (cm.sum(dim=1) + cm.sum(dim=0) + 1e-8)
    dice_mean = dice[1:].mean()
    
    return {
        'miou': miou.item(),
        'accuracy': accuracy.item(),
        'dice': dice_mean.item(),
        'iou_per_class': iou.cpu().numpy(),
        'confusion_matrix': cm.cpu().numpy()
    }


def get_lr_scheduler(optimizer, lr_decay_type, epochs, lr):
    """获取学习率调度器"""
    if lr_decay_type == 'cos':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    elif lr_decay_type == 'step':
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    elif lr_decay_type == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    else:
        scheduler = None
    
    return scheduler


def set_optimizer_lr(optimizer, lr_scheduler, epoch):
    """设置优化器学习率"""
    if lr_scheduler is not None:
        lr_scheduler.step(epoch)
    
    current_lr = optimizer.param_groups[0]['lr']
    return current_lr


def test_training_functions():
    """测试训练函数"""
    print("测试训练函数...")
    
    # 创建测试数据
    batch_size = 2
    num_classes = 13
    H, W = 32, 32
    
    pred = torch.randn(batch_size, num_classes, H, W)
    target = torch.randint(0, num_classes, (batch_size, H, W))
    
    # 测试损失函数
    print("1. 测试损失函数:")
    ce_loss = CE_Loss(pred, target, None, num_classes)
    print(f"   CE Loss: {ce_loss.item():.4f}")
    
    focal_loss = Focal_Loss(pred, target, None, num_classes)
    print(f"   Focal Loss: {focal_loss.item():.4f}")
    
    # 测试综合损失函数
    combined_loss = get_loss_function(num_classes, use_combined_loss=True)
    if isinstance(combined_loss, CombinedLoss):
        total_loss, loss_dict = combined_loss(pred, target)
        print(f"   Combined Loss - Total: {total_loss.item():.4f}")
        print(f"   Combined Loss - Components: {loss_dict}")
    
    # 测试评估指标
    print("\n2. 测试评估指标:")
    metrics = calculate_metrics(pred, target, num_classes)
    print(f"   mIoU: {metrics['miou']:.4f}")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Dice: {metrics['dice']:.4f}")
    
    print("\n✅ 训练函数测试通过！")


if __name__ == "__main__":
    test_training_functions()