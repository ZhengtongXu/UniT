"""
Residual Feature Extractor
==============================

用于下游Policy集成的特征提取API

使用示例:
    # 1. 加载模型
    extractor = ResidualFeatureExtractor(
        ckpt_path='representation_models/xxx/checkpoints/last.ckpt'
    )

    # 2. 提取特征
    z_fused, residual = extractor.extract_features(
        tactile_img=tactile,    # (B, 3, 128, 160)
        visual_img=visual,      # (B, 3, 224, 224)
        robot_state=state       # (B, 8)
    )

    # 3. 池化为1D向量 (用于policy)
    fused_feat, residual_feat = extractor.to_1d_features(z_fused, residual)
"""

import torch
import torch.nn as nn
from typing import Tuple
import os


class ResidualFeatureExtractor:
    """
    封装ResidualVQModel用于策略学习

    提供两种接口:
    1. extract_features: 提取latent表示 (B, C, H, W)
    2. to_1d_features: 将latent池化为1D向量 (B, D)
    """

    def __init__(self, ckpt_path: str, device: str = 'cuda'):
        """
        Args:
            ckpt_path: Residual VQ-VAE checkpoint路径
            device: 运行设备 (cuda/cpu)
        """
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint不存在: {ckpt_path}")

        self.device = torch.device(device)

        # 加载模型
        from UniT.taming.models.residual_vqmodel import ResidualVQModel
        self.model = ResidualVQModel.load_from_checkpoint(ckpt_path)
        self.model.eval()
        self.model.to(self.device)

        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"✓ Residual Feature Extractor加载成功")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Device: {self.device}")

        # 延迟初始化pooling head (首次调用to_1d_features时创建)
        self.pooling_head = None

    @torch.no_grad()
    def extract_features(self,
                        tactile_img: torch.Tensor,
                        visual_img: torch.Tensor,
                        robot_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取触觉特征

        Args:
            tactile_img: (B, 3, 128, 160) 触觉图像
            visual_img: (B, 3, 224, 224) 视觉图像
            robot_state: (B, 8) 机器人状态

        Returns:
            z_fused: (B, C, H, W) 完整触觉表征 (预期 + 残差)
            residual: (B, C, H, W) 纯意外信号
        """
        # 确保在正确的设备上
        tactile_img = tactile_img.to(self.device)
        visual_img = visual_img.to(self.device)
        robot_state = robot_state.to(self.device)

        # 构建batch
        batch = {
            'tactile_image': tactile_img,
            'visual_image': visual_img,
            'robot_state': robot_state
        }

        # 前向传播
        outputs = self.model(batch)

        # 融合latent
        z_fused = outputs['z_exp'] + outputs['residual']
        residual = outputs['residual']

        return z_fused, residual

    @torch.no_grad()
    def to_1d_features(self,
                      z_fused: torch.Tensor,
                      residual: torch.Tensor,
                      output_dim: int = 512) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将空间latent池化为1D向量 (用于policy输入)

        使用ConvPoolingHead进行池化 (与UniT保持一致)

        Args:
            z_fused: (B, C, H, W) 融合latent
            residual: (B, C, H, W) 残差latent
            output_dim: 输出维度 (默认512)

        Returns:
            fused_feat: (B, output_dim) 融合特征
            residual_feat: (B, output_dim) 残差特征
        """
        # 延迟初始化pooling head
        if self.pooling_head is None:
            from UniT.model.common.position_encodings import ConvPoolingHead

            C, H, W = z_fused.shape[1:]
            self.pooling_head = ConvPoolingHead(
                in_channels=C,
                output_dim=output_dim,
                spatial_size=(H, W)
            )
            self.pooling_head.to(self.device)
            self.pooling_head.eval()

            print(f"  初始化ConvPoolingHead: ({C}, {H}, {W}) → {output_dim}")

        # 池化
        fused_feat = self.pooling_head(z_fused)       # (B, output_dim)
        residual_feat = self.pooling_head(residual)   # (B, output_dim)

        return fused_feat, residual_feat

    @torch.no_grad()
    def get_residual_statistics(self,
                                tactile_img: torch.Tensor,
                                visual_img: torch.Tensor,
                                robot_state: torch.Tensor) -> dict:
        """
        获取residual统计量 (用于分析)

        Args:
            tactile_img: (B, 3, 128, 160)
            visual_img: (B, 3, 224, 224)
            robot_state: (B, 8)

        Returns:
            stats: dict包含residual的各种统计量
        """
        _, residual = self.extract_features(tactile_img, visual_img, robot_state)

        return {
            'mean': residual.mean().item(),
            'std': residual.std().item(),
            'min': residual.min().item(),
            'max': residual.max().item(),
            'l1_norm': residual.abs().mean().item(),
            'l2_norm': residual.pow(2).mean().sqrt().item(),
            'sparsity': (residual.abs() < 0.01).float().mean().item()  # <1%的值接近0
        }


# 用于DiffusionPolicy集成的便捷wrapper
class ResidualTactileEncoder(nn.Module):
    """
    用于DiffusionPolicy的触觉编码器

    替换原有的VQModel.to_latent(),提供residual-aware特征

    使用:
        # 在DiffusionPolicy中
        self.tactile_encoder = ResidualTactileEncoder(
            residual_ckpt='path/to/residual_vqvae.ckpt',
            use_residual=True  # 是否使用残差特征
        )

        # 前向传播
        tactile_feat = self.tactile_encoder(
            tactile=obs_dict['tactile'],
            visual=obs_dict['camera_0'],
            robot_state=obs_dict['robot_state']
        )
    """

    def __init__(self,
                 residual_ckpt: str,
                 use_residual: bool = True,
                 fusion_mode: str = 'concat',
                 output_dim: int = 512):
        """
        Args:
            residual_ckpt: Residual VQ-VAE checkpoint路径
            use_residual: 是否使用residual特征 (False则只用z_fused)
            fusion_mode: 特征融合方式 ('concat'/'add'/'residual_only')
            output_dim: 输出维度
        """
        super().__init__()

        self.use_residual = use_residual
        self.fusion_mode = fusion_mode
        self.output_dim = output_dim

        # 初始化extractor
        self.extractor = ResidualFeatureExtractor(residual_ckpt)

        # 根据fusion_mode确定最终输出维度
        if fusion_mode == 'concat' and use_residual:
            self.final_dim = output_dim * 2  # 拼接fused + residual
        else:
            self.final_dim = output_dim

        print(f"ResidualTactileEncoder初始化:")
        print(f"  use_residual: {use_residual}")
        print(f"  fusion_mode: {fusion_mode}")
        print(f"  output_dim: {self.final_dim}")

    def forward(self,
                tactile: torch.Tensor,
                visual: torch.Tensor,
                robot_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            tactile: (B, 3, 128, 160)
            visual: (B, 3, 224, 224)
            robot_state: (B, 8)

        Returns:
            feat: (B, final_dim) 触觉特征
        """
        # 提取latent
        z_fused, residual = self.extractor.extract_features(tactile, visual, robot_state)

        # 池化为1D
        fused_feat, residual_feat = self.extractor.to_1d_features(
            z_fused, residual, output_dim=self.output_dim
        )

        # 融合策略
        if not self.use_residual:
            return fused_feat

        if self.fusion_mode == 'concat':
            return torch.cat([fused_feat, residual_feat], dim=-1)
        elif self.fusion_mode == 'add':
            return fused_feat + residual_feat
        elif self.fusion_mode == 'residual_only':
            return residual_feat
        else:
            raise ValueError(f"未知融合模式: {self.fusion_mode}")


# 测试代码
if __name__ == "__main__":
    print("测试Residual Feature Extractor\n")

    # 创建虚拟checkpoint (需要实际训练后的模型)
    # extractor = ResidualFeatureExtractor(
    #     ckpt_path='representation_models/xxx/checkpoints/last.ckpt'
    # )

    # 虚拟输入
    # batch_size = 4
    # tactile = torch.randn(batch_size, 3, 128, 160)
    # visual = torch.randn(batch_size, 3, 224, 224)
    # robot_state = torch.randn(batch_size, 8)

    # # 提取特征
    # z_fused, residual = extractor.extract_features(tactile, visual, robot_state)
    # print(f"Latent features:")
    # print(f"  z_fused: {z_fused.shape}")
    # print(f"  residual: {residual.shape}")

    # # 池化为1D
    # fused_feat, residual_feat = extractor.to_1d_features(z_fused, residual)
    # print(f"\n1D features:")
    # print(f"  fused_feat: {fused_feat.shape}")
    # print(f"  residual_feat: {residual_feat.shape}")

    # # 统计量
    # stats = extractor.get_residual_statistics(tactile, visual, robot_state)
    # print(f"\nResidual statistics:")
    # for k, v in stats.items():
    #     print(f"  {k}: {v:.4f}")

    print("提示: 需要实际训练好的checkpoint才能运行测试")
