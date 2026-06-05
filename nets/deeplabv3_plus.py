#用于模型画图的模型代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from nets.xception import xception
from nets.mobilenetv2 import mobilenetv2


class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial

        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_level_features = self.features[:4](x)
        x = self.features[4:](low_level_features)
        return low_level_features, x


class ResNetBackbone(nn.Module):
    """ResNet50骨干网络"""

    def __init__(self, pretrained=True):
        super(ResNetBackbone, self).__init__()

        # 加载预训练的ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # 移除最后的全连接层和平均池化层
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 各个残差块
        self.layer1 = resnet.layer1  # 输出通道256，下采样4倍
        self.layer2 = resnet.layer2  # 输出通道512，下采样8倍
        self.layer3 = resnet.layer3  # 输出通道1024，下采样16倍
        self.layer4 = resnet.layer4  # 输出通道2048，下采样32倍

        # 调整stride以获得16倍下采样
        # 将layer4的第一个block的stride从2改为1
        for module in self.layer4.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.padding = (1, 1)
                    module.dilation = (1, 1)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # 各层特征
        layer1_out = self.layer1(x)  # [B, 256, H/4, W/4]
        layer2_out = self.layer2(layer1_out)  # [B, 512, H/8, W/8]
        layer3_out = self.layer3(layer2_out)  # [B, 1024, H/16, W/16]
        layer4_out = self.layer4(layer3_out)  # [B, 2048, H/16, W/16]

        # 返回浅层特征（layer1）和深层特征（layer4）
        return layer1_out, layer4_out


class ASPP(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(ASPP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        [b, c, row, col] = x.size()
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)

        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        result = self.conv_cat(feature_cat)
        return result


class UnifiedAttentionModule(nn.Module):
    """统一的注意力模块，整合DFF、边界优化和EGA的功能"""

    def __init__(self, channels=256, use_boundary_guidance=True):
        super().__init__()
        self.channels = channels

        # 1. 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

        # 2. 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )

        # 3. 边界引导注意力（可选）
        self.use_boundary_guidance = use_boundary_guidance
        if use_boundary_guidance:
            self.boundary_attention = nn.Sequential(
                nn.Conv2d(1, channels // 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid()
            )

        # 4. 特征融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

        # 5. 简化的小目标增强
        self.small_target_enhance = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, boundary_map=None):
        identity = x

        # 通道注意力
        ca = self.channel_attention(x)
        x_ca = x * ca

        # 空间注意力
        sa = self.spatial_attention(x_ca)
        x_sa = x_ca * sa

        # 边界引导注意力
        if self.use_boundary_guidance and boundary_map is not None:
            ba = self.boundary_attention(boundary_map)
            x_sa = x_sa * ba

        # 小目标增强
        x_enhanced = self.small_target_enhance(x_sa)

        # 融合
        x_fused = self.fusion(x_enhanced)

        # 残差连接
        return x_fused + identity


class FeatureFusion(nn.Module):
    """简化版特征融合模块"""

    def __init__(self, high_dim=256, low_dim=48, out_dim=256):
        super().__init__()

        # 浅层特征处理
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 高层特征处理
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_dim, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(128, out_dim, 3, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, high_feat, low_feat):
        low_processed = self.low_conv(low_feat)
        high_processed = self.high_conv(high_feat)

        # 特征拼接
        fused = torch.cat([high_processed, low_processed], dim=1)

        return self.fusion_conv(fused)


class DeepLab(nn.Module):
    def __init__(self, num_classes, backbone="resnet50", pretrained=True, downsample_factor=16, use_auxiliary=True,use_shortcut_conv=True, use_feature_fusion=True,
             use_attention=True, attention_type='unified',
             use_cls_conv=True, use_auxiliary_head=True,
             use_boundary_guidance=True):
        super(DeepLab, self).__init__()
        self.use_auxiliary = use_auxiliary
        # 新增：定义返回模式的默认值（避免未定义变量）
        self.return_intermediate = False
        self.return_boundary = False

        if backbone == "xception":
            self.backbone = xception(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        elif backbone == "mobilenet":
            self.backbone = MobileNetV2(downsample_factor=downsample_factor, pretrained=pretrained)
            in_channels = 320
            low_level_channels = 24
        elif backbone == "resnet50":
            self.backbone = ResNetBackbone(pretrained=pretrained)
            in_channels = 2048
            low_level_channels = 256
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, xception, resnet50.'.format(backbone))

        # ASPP特征提取模块
        self.aspp = ASPP(dim_in=in_channels, dim_out=256, rate=16 // downsample_factor)

        # 浅层特征处理
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # 特征融合模块
        self.feature_fusion = FeatureFusion(high_dim=256, low_dim=48, out_dim=256)

        # 统一注意力模块
        self.unified_attention = UnifiedAttentionModule(channels=256, use_boundary_guidance=True)

        # 分类卷积
        self.cls_conv = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, 1)
        )

        # 辅助分类头
        if self.use_auxiliary:
            self.auxiliary_head = nn.Sequential(
                nn.Conv2d(256, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, num_classes, 1)
            )

    def generate_boundary_map(self, low_level_features):
        """从浅层特征生成边界图"""
        with torch.no_grad():
            # Sobel算子
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                   dtype=torch.float32, device=low_level_features.device).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                   dtype=torch.float32, device=low_level_features.device).view(1, 1, 3, 3)

            # 计算梯度
            grad_x = F.conv2d(low_level_features, sobel_x.repeat(low_level_features.size(1), 1, 1, 1),
                              padding=1, groups=low_level_features.size(1))
            grad_y = F.conv2d(low_level_features, sobel_y.repeat(low_level_features.size(1), 1, 1, 1),
                              padding=1, groups=low_level_features.size(1))

            # 计算梯度幅值
            grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

            # 平均所有通道的梯度
            boundary_map = torch.mean(grad_magnitude, dim=1, keepdim=True)

            # 归一化
            boundary_map = (boundary_map - boundary_map.min()) / (boundary_map.max() - boundary_map.min() + 1e-8)

        return boundary_map

    def forward(self, x, return_intermediate=False, return_boundary=False):
        """前向传播
        
        Args:
            x: 输入张量 [B, C, H, W]
            return_intermediate: 是否返回中间结果（用于可视化）
            return_boundary: 是否返回边界图
        """
        H, W = x.size(2), x.size(3)

        # 1. Backbone提取特征
        low_level_features, x_deep = self.backbone(x)

        # 2. 深层特征经ASPP增强
        x_aspp = self.aspp(x_deep)

        # 3. 浅层特征处理
        low_level_processed = self.shortcut_conv(low_level_features)

        # 4. 生成边界图
        boundary_map = self.generate_boundary_map(low_level_processed)

        # 5. 深层特征上采样
        x_aspp_upsampled = F.interpolate(
            x_aspp,
            size=low_level_processed.size()[2:],
            mode='bilinear',
            align_corners=True
        )

        # 6. 特征融合
        x_cat = self.feature_fusion(x_aspp_upsampled, low_level_processed)

        # 7. 应用统一注意力模块
        x_attention = self.unified_attention(x_cat, boundary_map)

        # 8. 分类预测
        x_cls = self.cls_conv(x_attention)

        # 9. 上采样到原始图像尺寸
        x_final = F.interpolate(
            x_cls,
            size=(H, W),
            mode='bilinear',
            align_corners=True
        )

        # 10. 辅助输出（用于训练）
        if self.use_auxiliary and self.training:
            auxiliary_output = self.auxiliary_head(x_aspp)
            auxiliary_output = F.interpolate(
                auxiliary_output,
                size=(H, W),
                mode='bilinear',
                align_corners=True
            )
            if return_intermediate:
                return {
                    "pred": x_final,
                    "auxiliary": auxiliary_output,
                    "aspp": x_aspp,
                    "fusion": x_cat,
                    "attention": x_attention,
                    "boundary": boundary_map,
                    "low_level": low_level_features,
                    "low_level_processed": low_level_processed,
                    "x_deep": x_deep
                }
            elif return_boundary:
                return x_final, auxiliary_output, boundary_map
            else:
                return x_final, auxiliary_output
        
        # 11. 根据参数返回不同的结果（用于推理和可视化）
        if return_intermediate:
            return {
                "pred": x_final,
                "aspp": x_aspp,
                "fusion": x_cat,
                "attention": x_attention,
                "boundary": boundary_map,
                "low_level": low_level_features,
                "low_level_processed": low_level_processed,
                "x_deep": x_deep
            }
        
        if return_boundary:
            return x_final, boundary_map
        
        # 12. 默认返回（普通推理）
        return x_final



def print_model_blocks(model, indent=0):
    """递归打印模型的顶层模块"""
    indent_str = "  " * indent
    print(f"{indent_str}{model.__class__.__name__}")
    for name, child in model.named_children():
        print(f"{indent_str}  {name}: {child.__class__.__name__}")


if __name__ == "__main__":
    num_classes = 13
    model = DeepLab(num_classes=num_classes, backbone="resnet50")

    print("第一阶段改进后的模型架构：")
    print("=" * 50)
    print_model_blocks(model)
    print("=" * 50)

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    print(f"\n输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")