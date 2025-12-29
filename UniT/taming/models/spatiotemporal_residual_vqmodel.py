"""
Spatiotemporal Residual VQ-VAE: "Feeling the Unexpected" (感受那份意外)
==================================================================

核心理念:
Prophet预测"预期的平滑趋势", Residual捕捉"意外事件"
VQ将连续的意外信号压缩成离散的事件码(Event Code)

时空架构:
1. Prophet Stream (Causal TCN):
   (Visual_Context + State + Action_Seq) → Z^_seq (预期触觉序列)

2. Observation Stream (TimeDistributed 2D CNN):
   Tactile_Video → Z_real_seq (真实触觉序列)

3. Residual Stream (Transformer + VQ):
   R_seq = Z_real_seq - Z^_seq
   → Transformer Encoder → Global Pool → VQ → Event Code (e_k)

4. Dynamic FiLM Decoder (Style Expander):
   Z^_seq (Content) + e_k (Style) → FiLM Modulation → Reconstructed Video

输入维度:
- Tactile Video: (B, C=3, T=16, H=128, W=160)
- Visual Context: (B, C_v, H_v, W_v)
- State: (B, T, state_dim)
- Action: (B, T, action_dim)

输出:
- Reconstructed Video: (B, C=3, T=16, H=128, W=160)
- Event Code: (B, codebook_size) one-hot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional, Tuple
import torchvision.models as models
from UniT.taming.modules.vqvae.quantize import VectorQuantizer2
import math


class TimeDistributed(nn.Module):
    """
    TimeDistributed包装器: 将2D CNN应用到视频的每一帧

    节省显存: 不使用3D CNN, 而是复用2D CNN处理空间特征

    Input: (B, C, T, H, W)
    Output: (B, C_out, T, H_out, W_out)
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)
        Returns:
            (B, C_out, T, H_out, W_out)
        """
        B, C, T, H, W = x.shape

        # Reshape: (B, C, T, H, W) → (B*T, C, H, W)
        x_reshaped = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, T, C, H, W)
        x_reshaped = x_reshaped.view(B * T, C, H, W)  # (B*T, C, H, W)

        # Apply 2D module
        y = self.module(x_reshaped)  # (B*T, C_out, H_out, W_out)

        # Reshape back: (B*T, C_out, H_out, W_out) → (B, C_out, T, H_out, W_out)
        C_out, H_out, W_out = y.shape[1:]
        y = y.view(B, T, C_out, H_out, W_out)  # (B, T, C_out, H_out, W_out)
        y = y.permute(0, 2, 1, 3, 4).contiguous()  # (B, C_out, T, H_out, W_out)

        return y


class CausalTCN(nn.Module):
    """
    Causal Temporal Convolutional Network (因果一维卷积)

    用于Prophet Stream, 处理时间维度
    确保t时刻的预测只依赖于过去和当前的信息 (≤t)

    Input: (B, T, D_in)
    Output: (B, T, D_out)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dilation_base: int = 2):
        super().__init__()

        layers = []
        for i in range(num_layers):
            dilation = dilation_base ** i
            padding = (kernel_size - 1) * dilation  # Causal padding

            layers.append(nn.Conv1d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            ))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(0.1))

        self.network = nn.Sequential(*layers)
        self.causal_shift = sum([(kernel_size - 1) * (dilation_base ** i) for i in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D_in)
        Returns:
            (B, T, D_out)
        """
        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)  # (B, D_in, T)

        y = self.network(x)  # (B, D_out, T + padding)

        # Remove future information (causal)
        if self.causal_shift > 0:
            y = y[:, :, :-self.causal_shift]  # (B, D_out, T)

        y = y.transpose(1, 2)  # (B, T, D_out)
        return y


class ProphetEncoder(nn.Module):
    """
    Prophet Stream: 预测预期的触觉序列

    Input:
        - Visual Context: (B, 3, H_v, W_v) - 单帧上下文图像
        - State Sequence: (B, T, state_dim) - 机器人状态序列
        - Action Sequence: (B, T, action_dim) - 未来动作序列

    Output:
        - Z^_seq: (B, T, latent_dim) - 预期触觉latent序列

    Architecture:
        Visual CNN → Global Pool → Broadcast to T
        + State + Action → Causal TCN → Z^_seq
    """

    def __init__(self,
                 visual_backbone: str = 'resnet18',
                 visual_pretrained: bool = True,
                 state_dim: int = 8,
                 action_dim: int = 8,
                 latent_dim: int = 256,
                 seq_length: int = 16,
                 tcn_layers: int = 3):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length

        # 1. Visual Encoder (单帧上下文)
        if visual_backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if visual_pretrained else None
            resnet = models.resnet18(weights=weights)
            visual_feat_dim = 512
        elif visual_backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if visual_pretrained else None
            resnet = models.resnet34(weights=weights)
            visual_feat_dim = 512
        else:
            raise ValueError(f"不支持的backbone: {visual_backbone}")

        # 移除最后的FC层
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # 2. Visual特征投影
        self.visual_proj = nn.Linear(visual_feat_dim, 128)

        # 3. Causal TCN (处理时序)
        # Input: visual_feat (broadcasted) + state + action
        tcn_input_dim = 128 + state_dim + action_dim
        self.tcn = CausalTCN(
            in_channels=tcn_input_dim,
            out_channels=latent_dim,
            num_layers=tcn_layers,
            kernel_size=3,
            dilation_base=2
        )

    def forward(self,
                visual_context: torch.Tensor,  # (B, 3, H, W)
                state_seq: torch.Tensor,       # (B, T, state_dim)
                action_seq: torch.Tensor) -> torch.Tensor:  # (B, T, action_dim)
        """
        Returns:
            z_pred_seq: (B, T, latent_dim)
        """
        B, T, _ = state_seq.shape

        # 1. Visual feature (单帧)
        v_feat = self.visual_encoder(visual_context)  # (B, 512, 1, 1)
        v_feat = v_feat.flatten(1)  # (B, 512)
        v_feat = self.visual_proj(v_feat)  # (B, 128)

        # 2. Broadcast visual feature to sequence
        v_feat_seq = v_feat.unsqueeze(1).expand(B, T, 128)  # (B, T, 128)

        # 3. Concatenate all features
        fused = torch.cat([v_feat_seq, state_seq, action_seq], dim=-1)  # (B, T, 128+state+action)

        # 4. Causal TCN
        z_pred_seq = self.tcn(fused)  # (B, T, latent_dim)

        return z_pred_seq


class ResidualTransformerAggregator(nn.Module):
    """
    Residual Stream Aggregator: 将残差序列聚合为单个Event Code

    Pipeline:
        R_seq (B, T, D) → Transformer Encoder → Global Pool → VQ → e_k

    核心思想:
        - Transformer捕捉整个序列的时序依赖
        - Global Pool得到整体事件表征
        - VQ将连续表征离散化为事件码
    """

    def __init__(self,
                 latent_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 ff_dim: int = 512):
        super().__init__()

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, residual_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            residual_seq: (B, T, latent_dim)
        Returns:
            pooled: (B, latent_dim) - 聚合后的事件表征
        """
        # Transformer
        attended = self.transformer(residual_seq)  # (B, T, latent_dim)

        # Global Pool: (B, T, D) → (B, D, T) → Pool → (B, D, 1) → (B, D)
        attended_t = attended.transpose(1, 2)  # (B, latent_dim, T)
        pooled = self.pool(attended_t).squeeze(-1)  # (B, latent_dim)

        return pooled


class StyleExpander(nn.Module):
    """
    Style Expander: 将静态Event Code展开为动态调制参数序列

    Input: e_k (B, codebook_dim) - 静态事件码
    Output: γ_seq, β_seq (B, T, C) - 时序FiLM参数

    Architecture: GRU展开
    """

    def __init__(self,
                 event_dim: int,      # VQ codebook embedding dim
                 film_dim: int,       # FiLM modulation dim (通常是decoder的channel数)
                 seq_length: int = 16,
                 hidden_dim: int = 256):
        super().__init__()

        self.seq_length = seq_length
        self.film_dim = film_dim

        # GRU: 从静态码生成时序参数
        self.gru = nn.GRU(
            input_size=event_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 投影到FiLM参数 (γ, β)
        self.to_gamma = nn.Linear(hidden_dim, film_dim)
        self.to_beta = nn.Linear(hidden_dim, film_dim)

        # 初始化: γ接近1, β接近0 (identity初始化)
        nn.init.constant_(self.to_gamma.bias, 0.0)
        nn.init.constant_(self.to_beta.bias, 0.0)
        nn.init.xavier_uniform_(self.to_gamma.weight, gain=0.01)
        nn.init.xavier_uniform_(self.to_beta.weight, gain=0.01)

    def forward(self, event_code: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            event_code: (B, event_dim)
        Returns:
            gamma_seq: (B, T, film_dim)
            beta_seq: (B, T, film_dim)
        """
        B = event_code.shape[0]

        # Expand event_code to sequence: (B, event_dim) → (B, T, event_dim)
        event_seq = event_code.unsqueeze(1).expand(B, self.seq_length, -1)  # (B, T, event_dim)

        # GRU
        gru_out, _ = self.gru(event_seq)  # (B, T, hidden_dim)

        # Project to FiLM parameters
        gamma_seq = self.to_gamma(gru_out) + 1.0  # (B, T, film_dim), bias towards 1
        beta_seq = self.to_beta(gru_out)  # (B, T, film_dim), bias towards 0

        return gamma_seq, beta_seq


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation Layer

    F'_t = γ_t ⊙ F_t + β_t

    支持force_identity模式: γ=1, β=0 (不调制)
    """

    def __init__(self, num_channels: int):
        super().__init__()
        self.num_channels = num_channels

    def forward(self,
                features: torch.Tensor,  # (B, C, T, H, W)
                gamma: torch.Tensor,     # (B, T, C)
                beta: torch.Tensor,      # (B, T, C)
                force_identity: bool = False) -> torch.Tensor:
        """
        Args:
            features: (B, C, T, H, W)
            gamma: (B, T, C)
            beta: (B, T, C)
            force_identity: 如果True, 忽略γ和β

        Returns:
            modulated: (B, C, T, H, W)
        """
        if force_identity:
            return features

        # Reshape gamma/beta for broadcasting: (B, T, C) → (B, C, T, 1, 1)
        gamma = gamma.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)  # (B, C, T, 1, 1)
        beta = beta.transpose(1, 2).unsqueeze(-1).unsqueeze(-1)    # (B, C, T, 1, 1)

        # FiLM modulation
        modulated = gamma * features + beta

        return modulated


class DynamicFiLMDecoder(nn.Module):
    """
    Dynamic FiLM Decoder: 使用Event Code动态调制解码过程

    Input:
        - z_pred_seq: (B, T, latent_dim) - Prophet预测序列 (Content/Base)
        - event_code: (B, event_dim) - VQ事件码 (Style)

    Output:
        - recon_video: (B, C=3, T, H, W) - 重建视频

    Architecture:
        1. Project z_pred_seq to spatial: (B, T, D) → (B, C, T, H, W)
        2. Style Expander: event_code → γ_seq, β_seq
        3. 多层FiLM调制的Conv Decoder
    """

    def __init__(self,
                 latent_dim: int = 256,
                 event_dim: int = 256,
                 seq_length: int = 16,
                 spatial_size: tuple = (16, 20),  # latent spatial size
                 out_channels: int = 3):
        super().__init__()

        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.spatial_size = spatial_size

        # 1. Project temporal latent to spatial-temporal
        # (B, T, latent_dim) → (B, T, C*H*W) → (B, C, T, H, W)
        H, W = spatial_size
        self.to_spatial = nn.Linear(latent_dim, 128 * H * W)

        # 2. Style Expander
        self.style_expander = StyleExpander(
            event_dim=event_dim,
            film_dim=128,  # Match the channel dim
            seq_length=seq_length
        )

        # 3. FiLM-modulated Decoder (TimeDistributed 2D Conv)
        # 使用TimeDistributed避免3D Conv的显存开销

        # Layer 1: 128 → 128
        self.conv1 = TimeDistributed(nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ))
        self.film1 = FiLMLayer(128)

        # Layer 2: 128 → 64, Upsample ×2
        self.conv2 = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ))
        self.film2 = FiLMLayer(64)

        # Layer 3: 64 → 32, Upsample ×2
        self.conv3 = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        ))
        self.film3 = FiLMLayer(32)

        # Layer 4: 32 → 3, Upsample ×2
        self.conv_out = TimeDistributed(nn.Sequential(
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        ))

    def forward(self,
                z_pred_seq: torch.Tensor,  # (B, T, latent_dim)
                event_code: torch.Tensor,  # (B, event_dim)
                force_identity: bool = False) -> torch.Tensor:
        """
        Returns:
            recon_video: (B, C=3, T, H, W)
        """
        B, T, _ = z_pred_seq.shape
        H, W = self.spatial_size

        # 1. Project to spatial-temporal
        z_spatial_flat = self.to_spatial(z_pred_seq)  # (B, T, 128*H*W)
        z_spatial = z_spatial_flat.view(B, T, 128, H, W)  # (B, T, 128, H, W)
        z_spatial = z_spatial.permute(0, 2, 1, 3, 4).contiguous()  # (B, 128, T, H, W)

        # 2. Expand style to temporal FiLM parameters
        gamma_seq, beta_seq = self.style_expander(event_code)  # (B, T, 128)

        # 3. FiLM-modulated decoding
        x = z_spatial  # (B, 128, T, H, W)

        # Layer 1: FiLM + Conv
        x = self.film1(x, gamma_seq, beta_seq, force_identity)  # (B, 128, T, H, W)
        x = self.conv1(x)  # (B, 128, T, H, W)

        # Layer 2: FiLM + Conv (upsample)
        # 注意: gamma/beta需要重新project到新的channel数
        # 为简化, 这里只在第一层使用FiLM (通常足够)
        x = self.conv2(x)  # (B, 64, T, 2H, 2W)

        # Layer 3: Conv (upsample)
        x = self.conv3(x)  # (B, 32, T, 4H, 4W)

        # Layer 4: Conv out (upsample to target size)
        recon_video = self.conv_out(x)  # (B, 3, T, 8H, 8W)

        return recon_video


class SpatiotemporalResidualVQModel(pl.LightningModule):
    """
    Spatiotemporal Residual VQ-VAE: "Feeling the Unexpected"

    时空架构, 将意外事件压缩为离散Event Code
    """

    def __init__(self,
                 # Prophet配置
                 prophet_config: dict,
                 # Observation Encoder配置 (复用2D VAE encoder)
                 obs_encoder_ckpt: Optional[str] = None,
                 # Residual Aggregator配置
                 residual_config: Optional[dict] = None,
                 # VQ配置
                 vq_config: Optional[dict] = None,
                 # Decoder配置
                 decoder_config: Optional[dict] = None,
                 # 损失权重
                 lambda_sparse: float = 0.01,
                 # 其他
                 learning_rate: float = 4.5e-6):
        super().__init__()

        self.save_hyperparameters()

        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate

        # 1. Prophet Encoder (Causal TCN)
        self.prophet = ProphetEncoder(**prophet_config)

        # 2. Observation Encoder (TimeDistributed 2D CNN)
        # TODO: 从预训练的2D VAE加载encoder
        # 暂时使用placeholder
        print("⚠️  Warning: Observation Encoder 需要从预训练VQ-VAE加载")
        print(f"   预训练checkpoint: {obs_encoder_ckpt}")

        # 3. Residual Aggregator (Transformer + Pool)
        if residual_config is None:
            residual_config = {
                'latent_dim': 256,
                'num_heads': 4,
                'num_layers': 2,
                'ff_dim': 512
            }
        self.residual_aggregator = ResidualTransformerAggregator(**residual_config)

        # 4. VQ Quantizer
        if vq_config is None:
            vq_config = {
                'n_e': 512,      # Codebook size
                'e_dim': 256,    # Embedding dim
                'beta': 0.25
            }
        self.vq = VectorQuantizer2(**vq_config)

        # 5. Dynamic FiLM Decoder
        if decoder_config is None:
            decoder_config = {
                'latent_dim': 256,
                'event_dim': 256,
                'seq_length': 16,
                'spatial_size': (16, 20),
                'out_channels': 3
            }
        self.decoder = DynamicFiLMDecoder(**decoder_config)

        print("✓ Spatiotemporal Residual VQ-VAE 初始化完成")

    def forward(self, batch: Dict[str, torch.Tensor], force_identity: bool = False) -> Dict:
        """
        Args:
            batch: {
                'tactile_video': (B, 3, T=16, H, W),
                'visual_context': (B, 3, H_v, W_v),
                'state_seq': (B, T, state_dim),
                'action_seq': (B, T, action_dim)
            }
            force_identity: 如果True, Decoder不使用VQ调制 (训练阶段1)

        Returns:
            {
                'recon_video': (B, 3, T, H, W),
                'z_pred_seq': (B, T, latent_dim),
                'z_real_seq': (B, T, latent_dim) [TODO],
                'event_code': (B, event_dim),
                'vq_loss': scalar
            }
        """
        # 1. Prophet Stream: 预测预期序列
        z_pred_seq = self.prophet(
            batch['visual_context'],
            batch['state_seq'],
            batch['action_seq']
        )  # (B, T, latent_dim)

        # 2. Observation Stream: 编码真实触觉序列
        # TODO: 实现TimeDistributed observation encoder
        # z_real_seq = self.obs_encoder(batch['tactile_video'])  # (B, T, latent_dim)
        # 暂时用placeholder
        B, _, T, _, _ = batch['tactile_video'].shape
        latent_dim = z_pred_seq.shape[-1]
        z_real_seq = torch.randn(B, T, latent_dim, device=z_pred_seq.device)

        # 3. Residual Stream: 聚合意外信号
        residual_seq = z_real_seq.detach() - z_pred_seq  # (B, T, latent_dim)
        residual_pooled = self.residual_aggregator(residual_seq)  # (B, latent_dim)

        # 4. VQ: 离散化为Event Code
        # VectorQuantizer2 expects (B, C, H, W), so reshape (B, D) → (B, D, 1, 1)
        residual_4d = residual_pooled.unsqueeze(-1).unsqueeze(-1)  # (B, latent_dim, 1, 1)
        event_code_4d, vq_loss, _ = self.vq(residual_4d)  # (B, latent_dim, 1, 1)
        event_code = event_code_4d.squeeze(-1).squeeze(-1)  # (B, latent_dim)

        # 5. Dynamic FiLM Decoder: 重建
        recon_video = self.decoder(z_pred_seq, event_code, force_identity)  # (B, 3, T, H, W)

        return {
            'recon_video': recon_video,
            'z_pred_seq': z_pred_seq,
            'z_real_seq': z_real_seq,
            'residual_seq': residual_seq,
            'event_code': event_code,
            'vq_loss': vq_loss
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9))
        return optimizer


if __name__ == "__main__":
    # 测试代码
    print("Testing Spatiotemporal Residual VQ-VAE...")

    model = SpatiotemporalResidualVQModel(
        prophet_config={
            'visual_backbone': 'resnet18',
            'state_dim': 8,
            'action_dim': 8,
            'latent_dim': 256,
            'seq_length': 16
        }
    )

    batch = {
        'tactile_video': torch.randn(2, 3, 16, 128, 160),
        'visual_context': torch.randn(2, 3, 224, 224),
        'state_seq': torch.randn(2, 16, 8),
        'action_seq': torch.randn(2, 16, 8)
    }

    outputs = model(batch)
    print(f"✓ Recon video shape: {outputs['recon_video'].shape}")
    print(f"✓ Event code shape: {outputs['event_code'].shape}")
    print("✓ Forward pass successful!")
