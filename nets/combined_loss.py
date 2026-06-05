import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """综合损失函数，包含CE、Dice、边界损失"""
    
    def __init__(self, num_classes, weights=None, ignore_index=255):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # 默认权重
        self.weights = weights or {
            'ce': 1.0,      # 交叉熵损失
            'dice': 1.0,    # Dice损失
            'boundary': 0.5, # 边界损失
        }
        
        # 交叉熵损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='mean')
        
        # Dice损失
        self.dice_loss = DiceLoss(num_classes, ignore_index)
        
        # 边界损失
        self.boundary_loss = BoundaryLoss(num_classes, ignore_index)
        
    def forward(self, pred, target):
        """
        Args:
            pred: 预测结果 [B, C, H, W]
            target: 真实标签 [B, H, W] 或 [B, C, H, W]
        Returns:
            total_loss: 总损失
            loss_dict: 各损失分量
        """
        # 处理target的不同格式
        if target.dim() == 4:
            target = torch.argmax(target, dim=1)
        elif target.dim() != 3:
            raise ValueError(f"target的维度应为3或4，当前维度为{target.dim()}")
    
        total_loss = 0.0
        loss_dict = {}
    
        # CE
        if 'ce' in self.weights and self.weights['ce'] > 0:
            ce = self.ce_loss(pred, target)
            total_loss += self.weights['ce'] * ce
            loss_dict['ce'] = ce.item()
    
        # Dice
        if 'dice' in self.weights and self.weights['dice'] > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.weights['dice'] * dice
            loss_dict['dice'] = dice.item()
    
        # ⭐ Boundary（关键修复：可选）
        if 'boundary' in self.weights and self.weights['boundary'] > 0:
            boundary = self.boundary_loss(pred, target)
            total_loss += self.weights['boundary'] * boundary
            loss_dict['boundary'] = boundary.item()
    
        loss_dict['total'] = total_loss.item()
        return total_loss, loss_dict



class DiceLoss(nn.Module):
    """Dice损失，处理类别不平衡"""
    
    def __init__(self, num_classes, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth
        
    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W]
            target: [B, H, W]
        """
        # 将预测转换为概率
        pred = F.softmax(pred, dim=1)
        
        # 创建one-hot编码的目标
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        
        # 创建掩码，忽略特定索引
        valid_mask = (target != self.ignore_index).float()
        
        dice = 0.0
        count = 0
        
        for cls in range(self.num_classes):
            if cls == self.ignore_index:
                continue
                
            pred_cls = pred[:, cls] * valid_mask
            target_cls = target_one_hot[:, cls] * valid_mask
            
            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum()
            
            if union > 0:
                dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice += (1.0 - dice_coeff)
                count += 1
        
        return dice / max(count, 1)




# class BoundaryLoss(nn.Module):
#     """边界损失，专门优化边界区域"""
    
#     def __init__(self, num_classes, ignore_index=255):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
        
#         # 拉普拉斯算子核 (使用正确命名 laplace_kernel)
#         self.laplace_kernel = nn.Parameter(
#             torch.tensor([
#                 [0, 1, 0],
#                 [1, -4, 1],
#                 [0, 1, 0]
#             ], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
#             requires_grad=False
#         )

#     def forward(self, pred, target):
#         # 获取边界区域
#         pred_boundary = self.get_boundary(pred)
#         target_boundary = self.get_boundary(target.unsqueeze(1))
        
#         # 拉平
#         pred_flat = pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
#         target_flat = target.view(-1)
#         boundary_mask = (target_boundary > 0).view(-1)

#         if boundary_mask.sum() > 0:
#             boundary_loss = F.cross_entropy(
#                 pred_flat[boundary_mask], 
#                 target_flat[boundary_mask],
#                 ignore_index=self.ignore_index,
#                 reduction='mean'
#             )
#         else:
#             boundary_loss = torch.tensor(0.0, device=pred.device)

#         return boundary_loss

#     def get_boundary(self, tensor):
#         """提取边界区域"""
#         if tensor.dim() == 4 and tensor.size(1) > 1:  
#             pred_class = torch.argmax(tensor, dim=1, keepdim=True).float()
#             boundary = self.apply_laplacian(pred_class)
#         else:
#             boundary = self.apply_laplacian(tensor.float())

#         boundary = (boundary.abs() > 0.1).float()
#         return boundary

#     def apply_laplacian(self, tensor):
#         """应用拉普拉斯算子"""
#         # 确保卷积核在与输入相同的设备上
#         laplace_kernel = self.laplace_kernel.to(tensor.device)  # 修正这里：使用正确的变量名
        
#         # 对每个通道应用卷积
#         if tensor.size(1) > 1:
#             boundary = torch.zeros_like(tensor[:, :1])
#             for i in range(tensor.size(1)):
#                 boundary += F.conv2d(
#                     tensor[:, i:i+1], 
#                     laplace_kernel,  # 修正这里：使用正确的变量名
#                     padding=1
#                 )
#         else:
#             boundary = F.conv2d(
#                 tensor, 
#                 laplace_kernel,  # 修正这里：使用正确的变量名
#                 padding=1
#             )
        
#         return boundary

# 替换你当前的BoundaryLoss类，损失值0.1，miou.:0.89，
# class BoundaryLoss(nn.Module):
#     """修复版边界损失 - 专为医学图像设计，更稳定"""
    
#     def __init__(self, num_classes, ignore_index=255):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
        
#         # 注册形态学操作核
#         self.register_buffer('morph_kernel', 
#             torch.ones(1, 1, 3, 3, dtype=torch.float32))
        
#         # 创建Sobel算子用于边缘检测
#         sobel_x = torch.tensor([[-1, 0, 1],
#                                 [-2, 0, 2],
#                                 [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1],
#                                 [0, 0, 0],
#                                 [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
#         self.register_buffer('sobel_x', sobel_x)
#         self.register_buffer('sobel_y', sobel_y)
        
#         print(f"[边界损失] 初始化: 使用Sobel算子检测边缘")
    
#     def extract_boundary_simple(self, target_onehot):
#         """
#         简化版边界提取 - 更稳定
#         target_onehot: [B, C, H, W]
#         """
#         B, C, H, W = target_onehot.shape
#         device = target_onehot.device
        
#         # 方法1：对每个类别单独提取边界，然后合并
#         all_boundaries = []
        
#         for cls in range(C):
#             if target_onehot[:, cls].sum() == 0:
#                 continue
                
#             mask = target_onehot[:, cls:cls+1, :, :]  # [B, 1, H, W]
            
#             # 使用Sobel算子检测边缘
#             grad_x = F.conv2d(mask, self.sobel_x.to(device), padding=1)
#             grad_y = F.conv2d(mask, self.sobel_y.to(device), padding=1)
#             grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
#             # 归一化
#             max_val = grad_magnitude.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
#             grad_magnitude = grad_magnitude / (max_val + 1e-8)
            
#             # 二值化
#             boundary = (grad_magnitude > 0.1).float()
#             all_boundaries.append(boundary)
        
#         if all_boundaries:
#             # 合并所有类别的边界
#             boundary = torch.cat(all_boundaries, dim=1)  # [B, N, H, W]
#             boundary = boundary.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
#         else:
#             boundary = torch.zeros(B, 1, H, W, device=device)
        
#         return boundary
    
#     def forward(self, pred, target):
#         """
#         简化且稳定的边界损失实现
#         """
#         B, C, H, W = pred.shape
        
#         # 将target转换为one-hot
#         target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
        
#         # 提取边界区域
#         with torch.no_grad():
#             boundary_mask = self.extract_boundary_simple(target_onehot)  # [B, 1, H, W]
        
#         # 计算每个像素的交叉熵损失
#         ce_per_pixel = F.cross_entropy(
#             pred, 
#             target, 
#             ignore_index=self.ignore_index, 
#             reduction='none'
#         ).view(B, 1, H, W)
        
#         # 创建简单的权重图
#         # 边界区域权重为1.0，非边界区域权重为0.1
#         weights = torch.where(
#             boundary_mask > 0.5,
#             torch.tensor(1.0, device=pred.device),
#             torch.tensor(0.1, device=pred.device)
#         )
        
#         # 计算加权损失
#         weighted_loss = ce_per_pixel * weights
        
#         # 有效区域掩码
#         valid_mask = (target != self.ignore_index).float().view(B, 1, H, W)
        
#         # 计算平均损失
#         if valid_mask.sum() > 0:
#             # 只计算有效区域的加权损失
#             boundary_loss = (weighted_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
#         else:
#             boundary_loss = torch.tensor(0.0, device=pred.device)
        
        
        
#         return boundary_loss

#修改2后：
# class BoundaryLoss(nn.Module):
#     def __init__(self, num_classes, ignore_index=255):
#         super().__init__()
#         self.num_classes = num_classes
#         self.ignore_index = ignore_index
        
#         # 关键修复：使用nn.Parameter而不是register_buffer
#         # Sobel算子
#         self.sobel_x = nn.Parameter(
#             torch.tensor([[-1, 0, 1],
#                           [-2, 0, 2],
#                           [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
#             requires_grad=False
#         )
        
#         self.sobel_y = nn.Parameter(
#             torch.tensor([[-1, -2, -1],
#                           [0, 0, 0],
#                           [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
#             requires_grad=False
#         )
        
#         # 拉普拉斯算子
#         self.laplace_kernel = nn.Parameter(
#             torch.tensor([
#                 [0, 1, 0],
#                 [1, -4, 1],
#                 [0, 1, 0]
#             ], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
#             requires_grad=False
#         )
        
#         # 形态学核
#         self.dilate_kernel = nn.Parameter(
#             torch.ones(1, 1, 3, 3, dtype=torch.float32),
#             requires_grad=False
#         )
        
#         print(f"[边界损失] 初始化: 使用双检测算子")
    
#     def extract_boundary(self, target_tensor, method='combined'):
#         """
#         边界提取方法
#         Args:
#             target_tensor: [B, 1, H, W]
#             method: 'sobel', 'laplace', 或 'combined'
#         """
#         device = target_tensor.device
#         B, C, H, W = target_tensor.shape
        
#         # 确保卷积核在正确的设备上
#         sobel_x = self.sobel_x.to(device)
#         sobel_y = self.sobel_y.to(device)
#         laplace_kernel = self.laplace_kernel.to(device)
#         dilate_kernel = self.dilate_kernel.to(device)
        
#         if method == 'sobel':
#             # Sobel边缘检测
#             grad_x = F.conv2d(target_tensor, sobel_x, padding=1)
#             grad_y = F.conv2d(target_tensor, sobel_y, padding=1)
#             grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
#             # 自适应阈值
#             max_val = grad_magnitude.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
#             threshold = 0.15 * max_val
#             boundary = (grad_magnitude > threshold).float()
            
#         elif method == 'laplace':
#             # 拉普拉斯边缘检测
#             laplace = F.conv2d(target_tensor, laplace_kernel, padding=1)
#             boundary = (laplace.abs() > 0.1).float()
            
#         elif method == 'combined':
#             # 结合两种方法
#             # Sobel
#             grad_x = F.conv2d(target_tensor, sobel_x, padding=1)
#             grad_y = F.conv2d(target_tensor, sobel_y, padding=1)
#             grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
#             max_val = grad_magnitude.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
#             threshold = 0.15 * max_val
#             sobel_boundary = (grad_magnitude > threshold).float()
            
#             # Laplace
#             laplace = F.conv2d(target_tensor, laplace_kernel, padding=1)
#             laplace_boundary = (laplace.abs() > 0.1).float()
            
#             # 合并
#             boundary = torch.max(sobel_boundary, laplace_boundary)
            
#             # 膨胀操作，使边界更连续
#             boundary = F.conv2d(boundary, dilate_kernel, padding=1)
#             boundary = (boundary > 0).float()
        
#         return boundary
    
#     def forward(self, pred, target):
#         """
#         前向传播
#         """
#         B, C, H, W = pred.shape
#         device = pred.device
        
#         # 将target转换为one-hot
#         target_onehot = F.one_hot(target, C).permute(0, 3, 1, 2).float()
        
#         # 提取边界区域
#         with torch.no_grad():
#             # 方法1：对每个类别单独提取边界
#             boundary_masks = []
#             # for cls in range(C):
#             for cls in range(1, C):  # ⬅ 跳过背景
#                 if target_onehot[:, cls].sum() == 0:
#                     continue
#                 mask = target_onehot[:, cls:cls+1, :, :]
#                 cls_boundary = self.extract_boundary(mask, method='combined')
#                 boundary_masks.append(cls_boundary)
            
#             if boundary_masks:
#                 boundary_mask = torch.cat(boundary_masks, dim=1)
#                 boundary_mask = boundary_mask.max(dim=1, keepdim=True)[0]
#             else:
#                 boundary_mask = torch.zeros(B, 1, H, W, device=device)
        
#         # 计算每个像素的交叉熵损失
#         ce_per_pixel = F.cross_entropy(
#             pred, 
#             target, 
#             ignore_index=self.ignore_index, 
#             reduction='none'
#         ).view(B, 1, H, W)
        
#         # 创建权重图
#         weights = torch.where(
#             boundary_mask > 0.5,
#             torch.tensor(3.0, device=device),  # 边界区域权重
#             torch.tensor(0.5, device=device)   # 非边界区域权重
#         )
        
#         # 计算加权损失
#         weighted_loss = ce_per_pixel * weights
        
#         # 有效区域掩码
#         valid_mask = (target != self.ignore_index).float().view(B, 1, H, W)
        
#         # 计算平均损失
#         if valid_mask.sum() > 0:
#             boundary_loss = (weighted_loss * valid_mask).sum() / (valid_mask.sum() + 1e-8)
#         else:
#             boundary_loss = torch.tensor(0.0, device=device)
        
#         return boundary_loss
class BoundaryLoss(nn.Module):
    """
    Boundary Dice Loss（医学图像推荐）
    只在 GT 边界区域计算 Dice
    """
    def __init__(self, num_classes, ignore_index=255, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

        # Sobel
        self.sobel_x = nn.Parameter(
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )
        self.sobel_y = nn.Parameter(
            torch.tensor([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3),
            requires_grad=False
        )

        # Laplace
        self.laplace_kernel = nn.Parameter(
            torch.tensor([[0, 1, 0],
                          [1, -4, 1],
                          [0, 1, 0]], dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )

        # dilation
        self.dilate_kernel = nn.Parameter(
            torch.ones(1, 1, 3, 3),
            requires_grad=False
        )

    def extract_boundary(self, mask):
        """
        mask: [B,1,H,W] binary
        """
        device = mask.device
        gx = F.conv2d(mask, self.sobel_x.to(device), padding=1)
        gy = F.conv2d(mask, self.sobel_y.to(device), padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        max_val = grad.view(mask.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
        sobel_edge = (grad > 0.15 * max_val).float()

        laplace = F.conv2d(mask, self.laplace_kernel.to(device), padding=1)
        laplace_edge = (laplace.abs() > 0.1).float()

        edge = torch.max(sobel_edge, laplace_edge)
        edge = F.conv2d(edge, self.dilate_kernel.to(device), padding=1)
        return (edge > 0).float()

    def forward(self, pred, target):
        """
        pred:   [B,C,H,W] (logits)
        target: [B,H,W]
        """
        B, C, H, W = pred.shape
        pred_prob = F.softmax(pred, dim=1)

        total_loss = 0.0
        count = 0

        # 只对前景类别算（医学关键）
        for cls in range(1, C):
            gt_mask = (target == cls).float().unsqueeze(1)  # [B,1,H,W]
            if gt_mask.sum() == 0:
                continue

            with torch.no_grad():
                boundary = self.extract_boundary(gt_mask)

            pred_cls = pred_prob[:, cls:cls+1] * boundary
            gt_cls = gt_mask * boundary

            inter = (pred_cls * gt_cls).sum()
            union = pred_cls.sum() + gt_cls.sum()

            dice = (2 * inter + self.smooth) / (union + self.smooth)
            total_loss += (1.0 - dice)
            count += 1

        return total_loss / max(count, 1)


class FocalLoss(nn.Module):
    """Focal Loss，处理难易样本不平衡"""
    
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, pred, target):
        ce_loss = self.ce_loss(pred, target)
        
        # 计算概率
        pred_prob = F.softmax(pred, dim=1)
        
        # 获取目标类别的概率
        target_one_hot = F.one_hot(target, pred.size(1)).permute(0, 3, 1, 2)
        target_prob = (pred_prob * target_one_hot).sum(dim=1)
        
        # 计算Focal Loss权重
        focal_weight = self.alpha * (1 - target_prob) ** self.gamma
        
        # 应用权重
        focal_loss = focal_weight * ce_loss
        
        # 忽略ignore_index的区域
        mask = (target != self.ignore_index).float()
        focal_loss = (focal_loss * mask).sum() / (mask.sum() + 1e-8)
        
        return focal_loss


class MixedLoss(nn.Module):
    """混合损失，可选择组合"""
    
    def __init__(self, num_classes, loss_types=None, weights=None):
        super().__init__()
        self.num_classes = num_classes
        
        # 默认损失类型和权重
        self.loss_types = loss_types or ['ce', 'dice', 'boundary']
        self.weights = weights or {'ce': 1.0, 'dice': 1.0, 'boundary': 0.5, 'focal': 0.5}
        
        # 初始化各个损失
        self.losses = nn.ModuleDict()
        
        if 'ce' in self.loss_types:
            self.losses['ce'] = nn.CrossEntropyLoss()
        
        if 'dice' in self.loss_types:
            self.losses['dice'] = DiceLoss(num_classes)
        
        if 'boundary' in self.loss_types:
            self.losses['boundary'] = BoundaryLoss(num_classes)
        
        if 'focal' in self.loss_types:
            self.losses['focal'] = FocalLoss()
    
    def forward(self, pred, target):
        total_loss = 0
        loss_dict = {}
        
        for loss_name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            total_loss += self.weights.get(loss_name, 1.0) * loss_value
            loss_dict[loss_name] = loss_value.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


def test_loss_functions():
    """测试损失函数"""
    print("测试损失函数...")
    
    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    num_classes = 13
    H, W = 32, 32
    
    pred = torch.randn(batch_size, num_classes, H, W).to(device)
    target = torch.randint(0, num_classes, (batch_size, H, W)).to(device)
    
    # 测试CombinedLoss
    combined_loss = CombinedLoss(num_classes).to(device)
    total_loss, loss_dict = combined_loss(pred, target)
    
    print(f"CombinedLoss:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  CE loss: {loss_dict['ce']:.4f}")
    print(f"  Dice loss: {loss_dict['dice']:.4f}")
    print(f"  Boundary loss: {loss_dict['boundary']:.4f}")
    
    print("\n✅ 损失函数测试通过！")


if __name__ == "__main__":
    test_loss_functions()