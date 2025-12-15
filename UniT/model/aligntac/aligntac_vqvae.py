"""
AlignTac VQ-VAE: 双流解耦的触觉表征学习模型

核心创新:
1. 双帧输入 (I_t, I_{t-1}) 捕捉动态信息
2. 双流解耦: 几何流 (Z_geo) + 动力学流 (Z_dyn)
3. 双Codebook: 分别学习几何和动力学的离散原子模式
4. 跨模态预测: 从RGB预测光流,内化物理规律

架构:
    Input(6ch) → Encoder → [Proj_geo, Proj_dyn] → [Quant_geo, Quant_dyn] → [Dec_RGB, Dec_Flow]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple, Optional

from UniT.taming.modules.diffusionmodules.model import Encoder, Decoder
from UniT.taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from UniT.taming.logging_util import instantiate_from_config


class AlignTacVQVAE(pl.LightningModule):
    """
    AlignTac: 基于VQ-VAE的几何-动力学解耦触觉表征
    
    Args:
        ddconfig: Encoder/Decoder配置
        lossconfig: 损失函数配置
        n_embed: Codebook大小 (如1024)
        embed_dim: Code维度 (如3)
        geo_embed_dim: 几何特征维度
        dyn_embed_dim: 动力学特征维度
        ckpt_path: 预训练模型路径
        freeze_encoder: 是否冻结encoder
    """
    
    def __init__(
        self,
        ddconfig: dict,
        lossconfig: dict,
        n_embed: int = 1024,
        embed_dim: int = 3,
        geo_embed_dim: int = 3,
        dyn_embed_dim: int = 3,
        ckpt_path: str = None,
        ignore_keys: list = [],
        colorize_nlabels: int = None,
        monitor: str = None,
        remap: str = None,
        sane_index_shape: bool = False,
        # AlignTac特有参数
        use_dual_quantizer: bool = True,
        ortho_weight: float = 0.1,
        flow_weight: float = 1.0,
        rgb_weight: float = 1.0,
        phase_classifier: bool = False,
        num_phases: int = 5,
    ):
        super().__init__()
        
        # 保存超参数
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.geo_embed_dim = geo_embed_dim
        self.dyn_embed_dim = dyn_embed_dim
        self.use_dual_quantizer = use_dual_quantizer
        self.ortho_weight = ortho_weight
        self.flow_weight = flow_weight
        self.rgb_weight = rgb_weight
        self.phase_classifier = phase_classifier
        self.num_phases = num_phases
        
        # ==================== 共享Encoder ====================
        # 修改输入通道: 3 → 6 (双帧堆叠)
        ddconfig_encoder = ddconfig.copy()
        ddconfig_encoder['in_channels'] = 6  # 双帧RGB输入
        self.encoder = Encoder(**ddconfig_encoder)
        
        # ==================== 双流投影头 ====================
        z_channels = ddconfig['z_channels']
        
        # 几何投影: 提取形状、深度等静态特征
        self.proj_geo = nn.Sequential(
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
            nn.GroupNorm(8, z_channels),
            nn.SiLU(),
            nn.Conv2d(z_channels, geo_embed_dim, 1),
        )
        
        # 动力学投影: 提取剪切、滑动等动态特征
        self.proj_dyn = nn.Sequential(
            nn.Conv2d(z_channels, z_channels, 3, padding=1),
            nn.GroupNorm(8, z_channels),
            nn.SiLU(),
            nn.Conv2d(z_channels, dyn_embed_dim, 1),
        )
        
        # ==================== 双Codebook (可选) ====================
        if use_dual_quantizer:
            # 几何Codebook: 学习形状、深度的离散模式
            self.quantize_geo = VectorQuantizer(
                n_embed, geo_embed_dim, beta=0.25,
                remap=remap, sane_index_shape=sane_index_shape
            )
            # 动力学Codebook: 学习剪切、滑动的离散模式
            self.quantize_dyn = VectorQuantizer(
                n_embed, dyn_embed_dim, beta=0.25,
                remap=remap, sane_index_shape=sane_index_shape
            )
        else:
            # 单Codebook模式 (用于消融实验)
            self.quantize_geo = VectorQuantizer(
                n_embed, geo_embed_dim, beta=0.25,
                remap=remap, sane_index_shape=sane_index_shape
            )
            self.quantize_dyn = self.quantize_geo  # 共享
        
        # ==================== 双解码器 ====================
        # RGB解码器: 重建当前帧
        ddconfig_dec_rgb = ddconfig.copy()
        ddconfig_dec_rgb['out_ch'] = 3  # RGB输出
        ddconfig_dec_rgb['z_channels'] = geo_embed_dim
        self.decoder_rgb = Decoder(**ddconfig_dec_rgb)
        self.post_quant_conv_geo = nn.Conv2d(geo_embed_dim, geo_embed_dim, 1)
        
        # Flow解码器: 预测光流
        ddconfig_dec_flow = ddconfig.copy()
        ddconfig_dec_flow['out_ch'] = 2  # 光流(u, v)
        ddconfig_dec_flow['z_channels'] = dyn_embed_dim
        self.decoder_flow = Decoder(**ddconfig_dec_flow)
        self.post_quant_conv_dyn = nn.Conv2d(dyn_embed_dim, dyn_embed_dim, 1)
        
        # ==================== 物理阶段分类器 (可选) ====================
        if phase_classifier:
            # 从共享特征预测装配阶段
            self.phase_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(z_channels, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_phases),
            )
        
        # ==================== 损失函数 ====================
        self.loss = instantiate_from_config(lossconfig)
        
        # 不确定性加权的可学习参数
        self.log_sigma_rgb = nn.Parameter(torch.zeros(1))
        self.log_sigma_flow = nn.Parameter(torch.zeros(1))
        if phase_classifier:
            self.log_sigma_phase = nn.Parameter(torch.zeros(1))
        
        # ==================== 加载预训练 ====================
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        
        # 训练阶段控制 (课程学习)
        self.training_phase = 3  # 1: RGB only, 2: +Flow+Ortho, 3: Full
        
        if monitor is not None:
            self.monitor = monitor
    
    def init_from_ckpt(self, path: str, ignore_keys: list = []):
        """从预训练模型加载权重"""
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print(f"Deleting key {k} from state_dict.")
                    del sd[k]
        # 处理输入通道不匹配的情况
        if 'encoder.conv_in.weight' in sd:
            old_weight = sd['encoder.conv_in.weight']
            if old_weight.shape[1] == 3:  # 原模型是3通道
                # 复制权重用于6通道输入
                new_weight = torch.cat([old_weight, old_weight], dim=1)
                sd['encoder.conv_in.weight'] = new_weight
                print("Expanded encoder input channels from 3 to 6")
        
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")
        if len(missing) > 0:
            print(f"Missing keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected keys: {unexpected}")
    
    def set_training_phase(self, phase: int):
        """设置课程学习阶段
        
        Phase 1: 只训练RGB重建 (视觉预热)
        Phase 2: 加入Flow预测 + 正交约束 (动力学注入)
        Phase 3: 全部损失 (全局微调)
        """
        assert phase in [1, 2, 3], f"Invalid phase: {phase}"
        self.training_phase = phase
        print(f"Set training phase to {phase}")
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        编码双帧输入
        
        Args:
            x: 双帧堆叠输入 (B, 6, H, W)
        
        Returns:
            h: 共享特征 (B, z_channels, h, w)
            h_geo: 几何特征 (B, geo_embed_dim, h, w)
            h_dyn: 动力学特征 (B, dyn_embed_dim, h, w)
        """
        h = self.encoder(x)
        h_geo = self.proj_geo(h)
        h_dyn = self.proj_dyn(h)
        return h, h_geo, h_dyn
    
    def quantize_features(
        self, h_geo: torch.Tensor, h_dyn: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        量化几何和动力学特征
        
        Returns:
            z_geo: 量化后的几何特征
            z_dyn: 量化后的动力学特征
            loss_geo: 几何量化损失
            loss_dyn: 动力学量化损失
        """
        z_geo, loss_geo, _ = self.quantize_geo(h_geo)
        z_dyn, loss_dyn, _ = self.quantize_dyn(h_dyn)
        return z_geo, z_dyn, loss_geo, loss_dyn
    
    def decode_rgb(self, z_geo: torch.Tensor) -> torch.Tensor:
        """从几何特征解码RGB"""
        z = self.post_quant_conv_geo(z_geo)
        return self.decoder_rgb(z)
    
    def decode_flow(self, z_dyn: torch.Tensor) -> torch.Tensor:
        """从动力学特征解码光流"""
        z = self.post_quant_conv_dyn(z_dyn)
        return self.decoder_flow(z)
    
    def forward(
        self, 
        stacked_input: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            stacked_input: 双帧堆叠 (B, 6, H, W) = concat(I_t, I_{t-1})
            return_features: 是否返回中间特征
        
        Returns:
            dict containing:
                - rgb_pred: RGB重建 (B, 3, H, W)
                - flow_pred: 光流预测 (B, 2, H, W)
                - quant_loss_geo: 几何量化损失
                - quant_loss_dyn: 动力学量化损失
                - (optional) h_geo, h_dyn, z_geo, z_dyn
        """
        # 编码
        h, h_geo, h_dyn = self.encode(stacked_input)
        
        # 量化
        z_geo, z_dyn, loss_geo, loss_dyn = self.quantize_features(h_geo, h_dyn)
        
        # 解码
        rgb_pred = self.decode_rgb(z_geo)
        flow_pred = self.decode_flow(z_dyn)
        
        outputs = {
            'rgb_pred': rgb_pred,
            'flow_pred': flow_pred,
            'quant_loss_geo': loss_geo,
            'quant_loss_dyn': loss_dyn,
        }
        
        if return_features:
            outputs.update({
                'h': h,
                'h_geo': h_geo,
                'h_dyn': h_dyn,
                'z_geo': z_geo,
                'z_dyn': z_dyn,
            })
        
        # 物理阶段分类
        if self.phase_classifier:
            phase_logits = self.phase_head(h)
            outputs['phase_logits'] = phase_logits
        
        return outputs
    
    def to_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取连续的latent特征 (用于下游策略学习)
        
        这是继承自UniT的核心方法，不进行量化，保留更多信息
        
        Args:
            x: 双帧输入 (B, 6, H, W)
        
        Returns:
            h_geo: 几何特征 (B, geo_embed_dim, h, w)
            h_dyn: 动力学特征 (B, dyn_embed_dim, h, w)
        """
        h = self.encoder(x)
        h_geo = self.proj_geo(h)
        h_dyn = self.proj_dyn(h)
        return h_geo, h_dyn
    
    def to_latent_combined(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取合并的latent特征 (兼容原有接口)
        
        Returns:
            h_combined: (B, geo_embed_dim + dyn_embed_dim, h, w)
        """
        h_geo, h_dyn = self.to_latent(x)
        return torch.cat([h_geo, h_dyn], dim=1)
    
    def compute_ortho_loss(self, h_geo: torch.Tensor, h_dyn: torch.Tensor) -> torch.Tensor:
        """
        计算正交约束损失
        
        强迫几何特征和动力学特征互不包含
        L_orth = CosineSimilarity(h_geo, h_dyn)^2
        """
        # 全局平均池化
        geo_flat = F.adaptive_avg_pool2d(h_geo, 1).flatten(1)
        dyn_flat = F.adaptive_avg_pool2d(h_dyn, 1).flatten(1)
        
        # 归一化
        geo_norm = F.normalize(geo_flat, dim=1)
        dyn_norm = F.normalize(dyn_flat, dim=1)
        
        # 余弦相似度的平方
        cos_sim = torch.sum(geo_norm * dyn_norm, dim=1)
        ortho_loss = torch.mean(cos_sim ** 2)
        
        return ortho_loss
    
    def get_input(self, batch: Dict, keys: list = ['stacked_image', 'image_t', 'flow_gt']):
        """从batch中提取输入数据"""
        outputs = {}
        for k in keys:
            if k in batch:
                x = batch[k]
                if len(x.shape) == 4 and x.shape[-1] in [2, 3, 6]:  # (B, H, W, C)
                    x = x.permute(0, 3, 1, 2).contiguous()
                outputs[k] = x.float()
        return outputs
    
    def training_step(self, batch: Dict, batch_idx: int, optimizer_idx: int = 0):
        """训练步骤"""
        # 获取输入
        inputs = self.get_input(batch)
        stacked = inputs.get('stacked_image')
        rgb_gt = inputs.get('image_t')
        flow_gt = inputs.get('flow_gt')
        
        # 前向传播
        outputs = self.forward(stacked, return_features=True)
        rgb_pred = outputs['rgb_pred']
        flow_pred = outputs['flow_pred']
        loss_geo = outputs['quant_loss_geo']
        loss_dyn = outputs['quant_loss_dyn']
        h_geo = outputs['h_geo']
        h_dyn = outputs['h_dyn']
        
        # ==================== 计算损失 ====================
        log_dict = {}
        
        # 1. RGB重建损失 (始终计算)
        if optimizer_idx == 0:
            # 自编码器损失
            loss_rgb, log_ae = self.loss(
                loss_geo, rgb_gt, rgb_pred, optimizer_idx, 
                self.global_step, last_layer=self.decoder_rgb.conv_out.weight,
                split="train"
            )
            log_dict.update(log_ae)
            
            total_loss = loss_rgb
            
            # 2. Flow预测损失 (Phase 2+)
            if self.training_phase >= 2 and flow_gt is not None:
                loss_flow = F.mse_loss(flow_pred, flow_gt)
                
                # 不确定性加权
                w_flow = torch.exp(-self.log_sigma_flow)
                loss_flow_weighted = w_flow * loss_flow + self.log_sigma_flow
                
                total_loss = total_loss + self.flow_weight * loss_flow_weighted
                log_dict['train/flow_loss'] = loss_flow.detach()
                log_dict['train/sigma_flow'] = torch.exp(self.log_sigma_flow).detach()
            
            # 3. 正交约束 (Phase 2+)
            if self.training_phase >= 2:
                loss_ortho = self.compute_ortho_loss(h_geo, h_dyn)
                total_loss = total_loss + self.ortho_weight * loss_ortho
                log_dict['train/ortho_loss'] = loss_ortho.detach()
            
            # 4. 动力学量化损失
            if self.training_phase >= 2:
                total_loss = total_loss + loss_dyn.mean()
                log_dict['train/quant_loss_dyn'] = loss_dyn.detach().mean()
            
            # 5. 物理阶段分类 (Phase 3)
            if self.training_phase >= 3 and self.phase_classifier and 'phase_label' in batch:
                phase_gt = batch['phase_label']
                phase_logits = outputs['phase_logits']
                loss_phase = F.cross_entropy(phase_logits, phase_gt)
                
                w_phase = torch.exp(-self.log_sigma_phase)
                loss_phase_weighted = w_phase * loss_phase + self.log_sigma_phase
                
                total_loss = total_loss + loss_phase_weighted
                
                acc = (phase_logits.argmax(1) == phase_gt).float().mean()
                log_dict['train/phase_loss'] = loss_phase.detach()
                log_dict['train/phase_acc'] = acc.detach()
            
            self.log("train/total_loss", total_loss, prog_bar=True, logger=True, 
                     on_step=True, on_epoch=True)
            self.log_dict(log_dict, prog_bar=False, logger=True, 
                         on_step=True, on_epoch=True)
            
            return total_loss
        
        elif optimizer_idx == 1:
            # 判别器损失
            disc_loss, log_disc = self.loss(
                loss_geo, rgb_gt, rgb_pred, optimizer_idx,
                self.global_step, last_layer=self.decoder_rgb.conv_out.weight,
                split="train"
            )
            self.log("train/disc_loss", disc_loss, prog_bar=True, logger=True,
                     on_step=True, on_epoch=True)
            self.log_dict(log_disc, prog_bar=False, logger=True,
                         on_step=True, on_epoch=True)
            return disc_loss
    
    def validation_step(self, batch: Dict, batch_idx: int):
        """验证步骤"""
        inputs = self.get_input(batch)
        stacked = inputs.get('stacked_image')
        rgb_gt = inputs.get('image_t')
        flow_gt = inputs.get('flow_gt')
        
        outputs = self.forward(stacked, return_features=True)
        rgb_pred = outputs['rgb_pred']
        flow_pred = outputs['flow_pred']
        loss_geo = outputs['quant_loss_geo']
        h_geo = outputs['h_geo']
        h_dyn = outputs['h_dyn']
        
        # RGB重建损失
        loss_rgb, log_ae = self.loss(
            loss_geo, rgb_gt, rgb_pred, 0,
            self.global_step, last_layer=self.decoder_rgb.conv_out.weight,
            split="val"
        )
        
        log_dict = {'val/rgb_loss': loss_rgb.detach()}
        log_dict.update(log_ae)
        
        # Flow预测损失
        if flow_gt is not None:
            loss_flow = F.mse_loss(flow_pred, flow_gt)
            log_dict['val/flow_loss'] = loss_flow.detach()
        
        # 正交损失
        loss_ortho = self.compute_ortho_loss(h_geo, h_dyn)
        log_dict['val/ortho_loss'] = loss_ortho.detach()
        
        self.log_dict(log_dict, prog_bar=True, logger=True,
                     on_step=False, on_epoch=True, sync_dist=True)
        
        return log_dict
    
    def configure_optimizers(self):
        """配置优化器"""
        lr = self.learning_rate
        
        # 自编码器参数
        ae_params = (
            list(self.encoder.parameters()) +
            list(self.proj_geo.parameters()) +
            list(self.proj_dyn.parameters()) +
            list(self.quantize_geo.parameters()) +
            list(self.post_quant_conv_geo.parameters()) +
            list(self.decoder_rgb.parameters()) +
            list(self.post_quant_conv_dyn.parameters()) +
            list(self.decoder_flow.parameters()) +
            [self.log_sigma_rgb, self.log_sigma_flow]
        )
        
        if self.use_dual_quantizer:
            ae_params += list(self.quantize_dyn.parameters())
        
        if self.phase_classifier:
            ae_params += list(self.phase_head.parameters())
            ae_params += [self.log_sigma_phase]
        
        opt_ae = torch.optim.Adam(ae_params, lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), 
            lr=lr, betas=(0.5, 0.9)
        )
        
        return [opt_ae, opt_disc], []
    
    def get_last_layer(self):
        """获取解码器最后一层权重 (用于自适应权重)"""
        return self.decoder_rgb.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, batch: Dict, **kwargs) -> Dict[str, torch.Tensor]:
        """记录图像用于可视化"""
        log = {}
        inputs = self.get_input(batch)
        stacked = inputs.get('stacked_image').to(self.device)
        rgb_gt = inputs.get('image_t').to(self.device)
        flow_gt = inputs.get('flow_gt')
        
        outputs = self.forward(stacked)
        rgb_pred = outputs['rgb_pred']
        flow_pred = outputs['flow_pred']
        
        log['inputs_t'] = rgb_gt
        log['inputs_t-1'] = stacked[:, 3:6, :, :]  # 前一帧
        log['rgb_reconstruction'] = rgb_pred
        
        # 光流可视化 (转换为伪彩色)
        if flow_gt is not None:
            flow_gt = flow_gt.to(self.device)
            log['flow_gt'] = self._flow_to_rgb(flow_gt)
        log['flow_pred'] = self._flow_to_rgb(flow_pred)
        
        return log
    
    def _flow_to_rgb(self, flow: torch.Tensor) -> torch.Tensor:
        """将光流转换为RGB伪彩色用于可视化"""
        B, _, H, W = flow.shape
        
        # 计算幅值和角度
        u, v = flow[:, 0], flow[:, 1]
        magnitude = torch.sqrt(u**2 + v**2)
        angle = torch.atan2(v, u)
        
        # 归一化
        max_mag = magnitude.flatten(1).max(1, keepdim=True)[0].unsqueeze(-1).unsqueeze(-1)
        max_mag = torch.clamp(max_mag, min=1e-5)
        magnitude = magnitude / max_mag
        
        # HSV转RGB (简化版)
        h = (angle + torch.pi) / (2 * torch.pi)  # [0, 1]
        s = magnitude
        v_channel = torch.ones_like(h)
        
        # 简化的HSV到RGB转换
        rgb = torch.stack([h, s, v_channel], dim=1)
        
        return rgb
