"""
Residual VQ-VAE: 残差触觉表征学习模型
==========================================

核心架构:
1. Prophet Stream: (Visual + Robot_State) → z_exp (预期触觉latent)
2. Observation Stream: Tactile_Image → z_real (真实触觉latent)
3. Residual: r = z_real.detach() - z_exp (意外信号，且不让z_real迁就预测)
4. Quantization: VQ(r) → quantized_residual (信息瓶颈)
5. Reconstruction (Hybrid): Decoder(z_exp + quantized_residual) → T_recon
6. Reconstruction (Aux): Decoder(z_real) → T_recon_aux (保证z_real质量)

损失函数:
L_total = L_recon_hybrid + L_recon_aux + L_sparse + L_vq
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Optional
import torchvision.models as models
from UniT.taming.modules.vqvae.quantize import VectorQuantizer2


class ProphetEncoder(nn.Module):
    """
    Prophet Stream: 从视觉(和可选的机器人状态)预测预期的触觉latent

    支持配置:
    - use_state: 是否使用robot_state输入 (默认True, 可设为False降低过拟合)
    - visual_backbone: resnet18/resnet34 (可选更小的网络)
    - hidden_dims: MLP隐藏层维度 (可调整网络容量)
    """

    def __init__(self,
                 visual_backbone: str = 'resnet34',
                 visual_pretrained: bool = True,
                 state_dim: int = 8,
                 use_state: bool = True,  # 新增: 是否使用state
                 latent_shape: tuple = (3, 16, 20),
                 hidden_dims: list = [512, 256, 128]):
        super().__init__()

        self.latent_shape = latent_shape
        self.hidden_dims = hidden_dims
        self.use_state = use_state

        # 1. Visual Encoder (ResNet)
        if visual_backbone == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if visual_pretrained else None
            resnet = models.resnet34(weights=weights)
            visual_feat_dim = 512
        elif visual_backbone == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if visual_pretrained else None
            resnet = models.resnet18(weights=weights)
            visual_feat_dim = 512
        else:
            raise ValueError(f"不支持的backbone: {visual_backbone}")

        # 移除最后的FC层,保留全局平均池化
        self.visual_encoder = nn.Sequential(*list(resnet.children())[:-1])

        # 2. State Encoder (可选)
        if self.use_state:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(inplace=True)
            )
            state_feat_dim = hidden_dims[1]
        else:
            state_feat_dim = 0

        # 3. Fusion MLP
        fusion_input_dim = visual_feat_dim + state_feat_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dims[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 4. 投影到空间latent
        C, H, W = latent_shape
        self.to_spatial = nn.Linear(hidden_dims[2], C * H * W)

    def forward(self, visual_img: torch.Tensor, robot_state: torch.Tensor = None) -> torch.Tensor:
        # Visual branch
        v_feat = self.visual_encoder(visual_img)  # (B, 512, 1, 1)
        v_feat = v_feat.flatten(1)  # (B, 512)

        # State branch (可选)
        if self.use_state:
            assert robot_state is not None, "robot_state is required when use_state=True"
            s_feat = self.state_encoder(robot_state)  # (B, 256)
            fused = torch.cat([v_feat, s_feat], dim=1)  # (B, 768)
        else:
            fused = v_feat  # (B, 512)

        # Fusion
        fused = self.fusion_mlp(fused)  # (B, 128)

        # Project to spatial latent
        z_flat = self.to_spatial(fused)  # (B, C*H*W)

        C, H, W = self.latent_shape
        z_exp = z_flat.reshape(-1, C, H, W)  # (B, C, H, W)

        return z_exp


class ResidualVQModel(pl.LightningModule):
    """
    Residual VQ-VAE (v2 Bottleneck) 主模型
    """

    def __init__(self,
                 ddconfig: dict,
                 lossconfig: dict,
                 prophet_config: dict,
                 lambda_sparse: float = 0.01,
                 z_exp_dropout: float = 0.0,  # 新增: z_exp dropout率
                 residual_vq_config: Optional[dict] = None,
                 vq_model_ckpt: Optional[str] = None,
                 freeze_obs_encoder: bool = False):
        super().__init__()

        # 保存超参数 (用于checkpoint)
        self.save_hyperparameters()

        self.lambda_sparse = lambda_sparse
        self.z_exp_dropout = z_exp_dropout  # 新增: 保存dropout率
        self.aux_loss_weight = 1.0
        self.hybrid_loss_weight = 1.0  # v4新增: Hybrid重建Loss权重
        self.freeze_obs_encoder = freeze_obs_encoder  # v4新增: 是否冻结Obs Encoder
        self.learning_rate = 0.0

        # 1. Prophet Stream
        self.prophet_encoder = ProphetEncoder(**prophet_config)

        # 2. Observation Stream (复用VQModel的Encoder)
        from UniT.taming.modules.diffusionmodules.model import Encoder, Decoder
        self.obs_encoder = Encoder(**ddconfig)

        # 通道调整层 (z_channels → latent_dim)
        latent_dim = prophet_config['latent_shape'][0]  # 3
        z_channels = ddconfig['z_channels']

        self.quant_conv = nn.Conv2d(z_channels, latent_dim, kernel_size=1)

        # 3. 共享Decoder
        self.post_quant_conv = nn.Conv2d(latent_dim, z_channels, kernel_size=1)
        self.decoder = Decoder(**ddconfig)

        # 4. Residual VQ (新增: 故事的核心 - 量化触觉细节)
        if residual_vq_config is None:
            residual_vq_config = {
                'n_e': 512,             # 默认512，config会覆盖为256
                'e_dim': latent_dim,    # 必须匹配latent维度
                'beta': 0.25,           # Commitment loss权重
                'legacy': False
            }
        print(f"初始化Residual VQ: {residual_vq_config}")
        self.residual_vq = VectorQuantizer2(**residual_vq_config)

        # 5. Loss模块
        from UniT.taming.logging_util import instantiate_from_config
        self.recon_loss = instantiate_from_config(lossconfig)

        # 6. 加载预训练权重 (可选)
        if vq_model_ckpt is not None:
            self._load_pretrained_weights(vq_model_ckpt)

        # 7. 冻结Observation Stream (可选 - 通常设为False)
        if freeze_obs_encoder:
            self._freeze_observation_stream()

    def _load_pretrained_weights(self, ckpt_path: str):
        print(f"从预训练模型加载权重: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict']

        # 加载obs_encoder
        encoder_sd = {k.replace('encoder.', ''): v
                     for k, v in state_dict.items() if k.startswith('encoder.')}
        missing, unexpected = self.obs_encoder.load_state_dict(encoder_sd, strict=False)
        if missing:
            print(f"  obs_encoder缺失键: {len(missing)}")

        # 加载decoder
        decoder_sd = {k.replace('decoder.', ''): v
                     for k, v in state_dict.items() if k.startswith('decoder.')}
        missing, unexpected = self.decoder.load_state_dict(decoder_sd, strict=False)
        if missing:
            print(f"  decoder缺失键: {len(missing)}")

        # 加载quant_conv
        if 'quant_conv.weight' in state_dict:
            try:
                quant_conv_sd = {k.replace('quant_conv.', ''): v
                                for k, v in state_dict.items() if k.startswith('quant_conv.')}
                self.quant_conv.load_state_dict(quant_conv_sd, strict=False)
                print("  ✓ 加载quant_conv权重")
            except Exception:
                pass

        # 加载post_quant_conv
        if 'post_quant_conv.weight' in state_dict:
            try:
                post_quant_conv_sd = {k.replace('post_quant_conv.', ''): v
                                     for k, v in state_dict.items() if k.startswith('post_quant_conv.')}
                self.post_quant_conv.load_state_dict(post_quant_conv_sd, strict=False)
                print("  ✓ 加载post_quant_conv权重")
            except Exception:
                pass
        print(f"  ✓ 预训练权重加载完成")

    def _freeze_observation_stream(self):
        """冻结Observation Stream (obs_encoder + quant_conv)"""
        for param in self.obs_encoder.parameters():
            param.requires_grad = False
        for param in self.quant_conv.parameters():
            param.requires_grad = False
        print("  ✓ 已冻结Observation Stream")

    def _unfreeze_observation_stream(self):
        """解冻Observation Stream (obs_encoder + quant_conv)"""
        for param in self.obs_encoder.parameters():
            param.requires_grad = True
        for param in self.quant_conv.parameters():
            param.requires_grad = True
        print("  ✓ 已解冻Observation Stream")

    def update_freeze_status(self, freeze: bool):
        """动态更新Obs Encoder冻结状态 (被Curriculum Callback调用)"""
        if freeze and not self.freeze_obs_encoder:
            self._freeze_observation_stream()
            self.freeze_obs_encoder = True
        elif not freeze and self.freeze_obs_encoder:
            self._unfreeze_observation_stream()
            self.freeze_obs_encoder = False

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        tactile = batch['tactile_image']
        visual = batch['visual_image']
        robot_state = batch['robot_state']

        # 1. Prophet Stream: V(+S) → z_exp
        z_exp = self.prophet_encoder(visual, robot_state)

        # 2. Observation Stream: T → z_real
        h_obs = self.obs_encoder(tactile)
        z_real = self.quant_conv(h_obs)

        # 3. 残差计算 (核心)
        # 修正: 仅detach z_real，保留z_exp的梯度
        # Prophet Encoder 必须依靠这里的梯度 (通过Sparse Loss和VQ Loss) 来学习
        raw_residual = z_real.detach() - z_exp

        # 4. 残差量化 (故事的关键: 离散化触觉细节)
        quantized_residual, vq_loss, _ = self.residual_vq(raw_residual)

        # 5. 融合 (Base + Quantized Detail)
        # 新增: 训练时对z_exp进行dropout，强制VQ路径学习
        if self.training and self.z_exp_dropout > 0:
            # Spatial dropout: 随机丢弃整个空间位置的z_exp
            dropout_mask = torch.rand(z_exp.shape[0], 1, z_exp.shape[2], z_exp.shape[3],
                                     device=z_exp.device) > self.z_exp_dropout
            z_exp_masked = z_exp * dropout_mask
        else:
            z_exp_masked = z_exp

        z_fused = z_exp_masked + quantized_residual

        # 6. 解码重建 (Hybrid)
        z_dec = self.post_quant_conv(z_fused)
        recon = self.decoder(z_dec)

        # 7. 辅助重建 (Pure Tactile)
        # 关键: 强制 z_real 包含完整信息, 防止其退化
        z_dec_aux = self.post_quant_conv(z_real)
        recon_aux = self.decoder(z_dec_aux)

        return {
            'recon': recon,
            'recon_aux': recon_aux,
            'z_exp': z_exp,
            'z_real': z_real,
            'residual': quantized_residual, # 用于可视化
            'raw_residual': raw_residual,   # 用于稀疏Loss
            'vq_loss': vq_loss
        }

    def training_step(self, batch: Dict[str, torch.Tensor],
                     batch_idx: int, optimizer_idx: int):
        outputs = self(batch)
        tactile = batch['tactile_image']

        if optimizer_idx == 0:  # Generator
            # 1. 重建Loss (Hybrid) - v4: 支持权重控制
            recon_loss, log_dict = self.recon_loss(
                codebook_loss=torch.tensor(0.0, device=self.device),  # 无VQ
                inputs=tactile,
                reconstructions=outputs['recon'],
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split='train'
            )

            # 1.5 辅助重建Loss (Auxiliary - Pure Tactile) - v4: 支持权重控制
            recon_loss_aux, log_dict_aux = self.recon_loss(
                codebook_loss=torch.tensor(0.0, device=self.device),
                inputs=tactile,
                reconstructions=outputs['recon_aux'],
                optimizer_idx=0,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split='train'
            )

            # 2. 稀疏Loss (L1范数)
            raw_residual = outputs['raw_residual']
            sparse_loss = self.lambda_sparse * raw_residual.abs().mean()

            # 3. VQ Loss (Commitment Loss)
            vq_loss = outputs['vq_loss']

            # 总损失 (v4: 使用hybrid_loss_weight和aux_loss_weight)
            # Backward: 包含所有 Loss
            total_loss = (self.hybrid_loss_weight * recon_loss +
                         self.aux_loss_weight * recon_loss_aux +
                         sparse_loss + vq_loss)
            # Log: 不包含 sparse_loss (避免震荡干扰视线)
            total_loss_log = (self.hybrid_loss_weight * recon_loss +
                             self.aux_loss_weight * recon_loss_aux +
                             vq_loss)

            self.log('train/total_loss', total_loss_log, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/aux_weight', self.aux_loss_weight, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/hybrid_weight', self.hybrid_loss_weight, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/recon_loss', recon_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/recon_loss_aux', recon_loss_aux, prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/sparse_loss', sparse_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/vq_loss', vq_loss, prog_bar=True, on_step=False, on_epoch=True)

            # 监控残差统计
            residual = outputs['residual']
            self.log('train/residual_norm', residual.abs().mean(), prog_bar=False, on_step=False, on_epoch=True)
            self.log('train/raw_residual_norm', raw_residual.abs().mean(), on_step=False, on_epoch=True)
            
            # 记录其他loss组件
            exclude_keys = {'train/total_loss', 'train/recon_loss', 'train/sparse_loss', 'train/vq_loss'}
            for k, v in log_dict.items():
                if k not in exclude_keys:
                    self.log(k, v, on_step=False, on_epoch=True)
            
            # 记录Aux loss组件
            for k, v in log_dict_aux.items():
                if k not in exclude_keys:
                    self.log(k.replace('train/', 'train/aux_'), v, on_step=False, on_epoch=True)

            return total_loss

        if optimizer_idx == 1:  # Discriminator
            # Discriminate Hybrid
            disc_loss, log_dict = self.recon_loss(
                codebook_loss=torch.tensor(0.0, device=self.device),
                inputs=tactile,
                reconstructions=outputs['recon'].detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split='train'
            )
            
            # Discriminate Aux
            disc_loss_aux, log_dict_aux = self.recon_loss(
                codebook_loss=torch.tensor(0.0, device=self.device),
                inputs=tactile,
                reconstructions=outputs['recon_aux'].detach(),
                optimizer_idx=1,
                global_step=self.global_step,
                last_layer=self.get_last_layer(),
                split='train'
            )
            
            total_disc_loss = (disc_loss + disc_loss_aux) / 2

            self.log('train/disc_loss', disc_loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log('train/disc_loss_aux', disc_loss_aux, prog_bar=False, on_step=False, on_epoch=True)

            exclude_keys = {'train/disc_loss', 'train/total_loss'}
            for k, v in log_dict.items():
                if k not in exclude_keys:
                    self.log(k, v, on_step=False, on_epoch=True)

            return total_disc_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        outputs = self(batch)
        tactile = batch['tactile_image']

        # 重建Loss
        recon_loss, log_dict_ae = self.recon_loss(
            codebook_loss=torch.tensor(0.0, device=self.device),
            inputs=tactile,
            reconstructions=outputs['recon'],
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split='val'
        )
        
        # 辅助重建Loss
        recon_loss_aux, log_dict_aux = self.recon_loss(
            codebook_loss=torch.tensor(0.0, device=self.device),
            inputs=tactile,
            reconstructions=outputs['recon_aux'],
            optimizer_idx=0,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split='val'
        )

        # Discriminator Loss
        disc_loss, log_dict_disc = self.recon_loss(
            codebook_loss=torch.tensor(0.0, device=self.device),
            inputs=tactile,
            reconstructions=outputs['recon'],
            optimizer_idx=1,
            global_step=self.global_step,
            last_layer=self.get_last_layer(),
            split='val'
        )

        # 稀疏Loss
        raw_residual = outputs['raw_residual']
        sparse_loss = self.lambda_sparse * raw_residual.abs().mean()
        
        vq_loss = outputs['vq_loss']

        total_loss = recon_loss + self.aux_loss_weight * recon_loss_aux + sparse_loss + vq_loss
        total_loss_log = recon_loss + self.aux_loss_weight * recon_loss_aux + vq_loss

        self.log('val/sparse_loss', sparse_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/vq_loss', vq_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/total_loss', total_loss_log, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/raw_residual_norm', raw_residual.abs().mean(), on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/recon_loss_aux', recon_loss_aux, on_step=False, on_epoch=True, sync_dist=True)
        
        log_dict_ae_filtered = {k: v for k, v in log_dict_ae.items() if k not in ['val/total_loss', 'val/recon_loss']}
        log_dict_disc_filtered = {k: v for k, v in log_dict_disc.items() if k not in ['val/total_loss', 'val/recon_loss']}

        self.log_dict(log_dict_ae_filtered, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(log_dict_disc_filtered, on_step=False, on_epoch=True, sync_dist=True)

        return total_loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        lr = self.learning_rate

        # Generator优化器
        opt_gen = torch.optim.Adam(
            list(self.prophet_encoder.parameters()) +
            list(self.obs_encoder.parameters()) +
            list(self.quant_conv.parameters()) +
            list(self.residual_vq.parameters()) + 
            list(self.post_quant_conv.parameters()) +
            list(self.decoder.parameters()),
            lr=lr,
            betas=(0.5, 0.9)
        )

        # Discriminator优化器
        opt_disc = torch.optim.Adam(
            self.recon_loss.discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.9)
        )

        return [opt_gen, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch: Dict[str, torch.Tensor], **kwargs):
        """
        记录可视化图像到wandb
        """
        outputs = self(batch)
        log = {}

        # 1. 输入与重建 (RGB)
        log['inputs_tactile'] = batch['tactile_image'][:4]
        log['inputs_visual'] = batch['visual_image'][:4]
        log['recon_hybrid'] = outputs['recon'][:4]
        log['recon_aux'] = outputs['recon_aux'][:4]

        # 2. Latent可视化 (RGB模式, 因为z_channels=3)
        z_exp = outputs['z_exp'][:4]
        z_real = outputs['z_real'][:4]
        
        # Helper: Fixed Range Norm (-3 to 3) -> 检查数值尺度是否坍塌
        def fixed_norm(z):
            return torch.clamp((z + 3.0) / 6.0, 0., 1.)
            
        log['z_real_fixed'] = fixed_norm(z_real)
        log['z_exp_fixed'] = fixed_norm(z_exp)
        
        # Helper: Auto Range Norm (Min-Max) -> 查看结构细节
        def auto_norm(z):
            B = z.shape[0]
            z_flat = z.view(B, -1)
            z_min = z_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            z_max = z_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            denom = z_max - z_min
            denom[denom < 1e-6] = 1.0
            return (z - z_min) / denom

        log['z_real_auto'] = auto_norm(z_real)
        log['z_exp_auto'] = auto_norm(z_exp)

        # 3. 残差可视化 (RGB Heatmap)
        # 取绝对值，展示误差在3个通道上的分布
        # 黑色=无误差，彩色=有误差
        raw_res = outputs['raw_residual'][:4].abs()     # (B, 3, H, W)
        vq_res = outputs['residual'][:4].abs()          # (B, 3, H, W)
        
        # 使用 Auto Norm 映射到 0-1
        log['residual_raw_rgb'] = auto_norm(raw_res)
        log['residual_vq_rgb'] = auto_norm(vq_res)
        
        # 移除旧的灰度可视化，避免混淆
        # log['residual_rgb_auto'] 也可以移除了，因为它和 residual_raw_rgb 一样

        return log

    @torch.no_grad()
    def decode_z_exp_only(self, z_exp: torch.Tensor) -> torch.Tensor:
        """
        Decode from z_exp (Prophet prediction) alone without residual correction.

        This serves as a baseline to demonstrate the contribution of VQ residuals.

        Args:
            z_exp: (B, 3, H, W) - Prophet encoder prediction

        Returns:
            recon: (B, 3, 128, 160) - Reconstructed tactile image
        """
        z_dec = self.post_quant_conv(z_exp)
        recon = self.decoder(z_dec)
        return recon