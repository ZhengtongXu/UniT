"""
AlignTac Loss Functions

包含:
1. AlignTacLoss: 多任务损失，含不确定性加权
2. FlowLoss: 光流预测损失
3. PhaseClassificationLoss: 物理阶段分类损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class FlowLoss(nn.Module):
    """
    光流预测损失
    
    支持多种损失类型:
    - L1: 绝对误差
    - L2/MSE: 均方误差
    - Smooth L1: Huber损失
    - Charbonnier: 鲁棒损失
    """
    
    def __init__(
        self, 
        loss_type: str = 'charbonnier',
        eps: float = 1e-6,
        q: float = 0.5,
        reduce: str = 'mean'
    ):
        super().__init__()
        self.loss_type = loss_type
        self.eps = eps
        self.q = q
        self.reduce = reduce
    
    def forward(
        self, 
        flow_pred: torch.Tensor, 
        flow_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            flow_pred: 预测光流 (B, 2, H, W)
            flow_gt: GT光流 (B, 2, H, W)
            mask: 有效区域mask (B, 1, H, W), 可选
        
        Returns:
            flow_loss: 光流损失标量
        """
        diff = flow_pred - flow_gt
        
        if self.loss_type == 'l1':
            loss = torch.abs(diff)
        elif self.loss_type in ['l2', 'mse']:
            loss = diff ** 2
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(flow_pred, flow_gt, reduction='none')
        elif self.loss_type == 'charbonnier':
            # Charbonnier loss: (x^2 + eps)^q
            loss = (diff ** 2 + self.eps) ** self.q
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用mask
        if mask is not None:
            loss = loss * mask
            if self.reduce == 'mean':
                return loss.sum() / (mask.sum() + 1e-8)
            elif self.reduce == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            if self.reduce == 'mean':
                return loss.mean()
            elif self.reduce == 'sum':
                return loss.sum()
            else:
                return loss


class PhaseClassificationLoss(nn.Module):
    """
    物理阶段分类损失
    
    支持:
    - 标准交叉熵
    - 标签平滑
    - Focal Loss (处理类别不平衡)
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,  # 0表示不使用focal loss
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.focal_gamma = focal_gamma
        
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            logits: 分类logits (B, num_classes)
            targets: 类别标签 (B,)
        
        Returns:
            loss: 分类损失
            acc: 分类准确率
        """
        # 标准交叉熵 + 标签平滑
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        
        # Focal Loss调制
        if self.focal_gamma > 0:
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss = focal_weight * ce_loss
        else:
            loss = ce_loss
        
        loss = loss.mean()
        
        # 计算准确率
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean()
        
        return loss, acc


class AlignTacLoss(nn.Module):
    """
    AlignTac多任务损失
    
    总损失 = w_rgb * L_rgb + w_flow * L_flow + w_phase * L_phase + L_quant
    
    使用基于不确定性的自动加权:
    L = sum_i [ exp(-log_sigma_i) * L_i + log_sigma_i ]
    """
    
    def __init__(
        self,
        # 损失权重
        rgb_weight: float = 1.0,
        flow_weight: float = 1.0,
        phase_weight: float = 0.5,
        quant_weight: float = 1.0,
        # 损失配置
        flow_loss_type: str = 'charbonnier',
        phase_label_smoothing: float = 0.1,
        phase_focal_gamma: float = 0.0,
        num_phases: int = 5,
        # 自动加权
        use_uncertainty_weighting: bool = True,
        # 重建损失 (复用VQ-VAE的perceptual loss)
        perceptual_weight: float = 1.0,
    ):
        super().__init__()
        
        # 固定权重
        self.rgb_weight = rgb_weight
        self.flow_weight = flow_weight
        self.phase_weight = phase_weight
        self.quant_weight = quant_weight
        
        # 子损失
        self.flow_loss = FlowLoss(loss_type=flow_loss_type)
        self.phase_loss = PhaseClassificationLoss(
            num_classes=num_phases,
            label_smoothing=phase_label_smoothing,
            focal_gamma=phase_focal_gamma,
        )
        
        # 可学习的不确定性参数 (log(sigma^2))
        self.use_uncertainty_weighting = use_uncertainty_weighting
        if use_uncertainty_weighting:
            self.log_var_rgb = nn.Parameter(torch.zeros(1))
            self.log_var_flow = nn.Parameter(torch.zeros(1))
            self.log_var_phase = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        # 预测
        rgb_pred: torch.Tensor,
        flow_pred: torch.Tensor,
        phase_logits: Optional[torch.Tensor] = None,
        quant_loss: Optional[torch.Tensor] = None,
        # GT
        rgb_gt: torch.Tensor = None,
        flow_gt: torch.Tensor = None,
        phase_label: Optional[torch.Tensor] = None,
        # 控制
        training_phase: int = 3,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            rgb_pred: RGB重建 (B, 3, H, W)
            flow_pred: 光流预测 (B, 2, H, W)
            phase_logits: 阶段分类logits (B, num_phases)
            quant_loss: 量化损失 (已计算)
            rgb_gt: RGB GT (B, 3, H, W)
            flow_gt: 光流GT (B, 2, H, W)
            phase_label: 阶段标签 (B,)
            training_phase: 课程学习阶段 (1/2/3)
        
        Returns:
            total_loss: 总损失
            log_dict: 各项损失的字典
        """
        log_dict = {}
        total_loss = 0.0
        
        # ========== 1. RGB重建损失 (始终计算) ==========
        loss_rgb = F.l1_loss(rgb_pred, rgb_gt)
        log_dict['loss/rgb'] = loss_rgb.detach()
        
        if self.use_uncertainty_weighting:
            precision_rgb = torch.exp(-self.log_var_rgb)
            weighted_rgb = precision_rgb * loss_rgb + self.log_var_rgb
            log_dict['sigma/rgb'] = torch.exp(0.5 * self.log_var_rgb).detach()
        else:
            weighted_rgb = self.rgb_weight * loss_rgb
        
        total_loss = total_loss + weighted_rgb
        
        # ========== 2. 量化损失 (始终计算) ==========
        if quant_loss is not None:
            total_loss = total_loss + self.quant_weight * quant_loss
            log_dict['loss/quant'] = quant_loss.detach()
        
        # ========== 3. 光流损失 (Phase 2+) ==========
        if training_phase >= 2 and flow_gt is not None:
            loss_flow = self.flow_loss(flow_pred, flow_gt)
            log_dict['loss/flow'] = loss_flow.detach()
            
            if self.use_uncertainty_weighting:
                precision_flow = torch.exp(-self.log_var_flow)
                weighted_flow = precision_flow * loss_flow + self.log_var_flow
                log_dict['sigma/flow'] = torch.exp(0.5 * self.log_var_flow).detach()
            else:
                weighted_flow = self.flow_weight * loss_flow
            
            total_loss = total_loss + weighted_flow
        
        # ========== 4. 阶段分类 (Phase 3) ==========
        if training_phase >= 3 and phase_logits is not None and phase_label is not None:
            loss_phase, acc = self.phase_loss(phase_logits, phase_label)
            log_dict['loss/phase'] = loss_phase.detach()
            log_dict['acc/phase'] = acc.detach()
            
            if self.use_uncertainty_weighting:
                precision_phase = torch.exp(-self.log_var_phase)
                weighted_phase = precision_phase * loss_phase + self.log_var_phase
                log_dict['sigma/phase'] = torch.exp(0.5 * self.log_var_phase).detach()
            else:
                weighted_phase = self.phase_weight * loss_phase
            
            total_loss = total_loss + weighted_phase
        
        log_dict['loss/total'] = total_loss.detach()
        
        return total_loss, log_dict


class EPE(nn.Module):
    """
    End-Point Error (光流评估指标)
    
    EPE = sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self, 
        flow_pred: torch.Tensor, 
        flow_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            flow_pred: (B, 2, H, W)
            flow_gt: (B, 2, H, W)
            mask: (B, 1, H, W), optional
        
        Returns:
            epe: 平均端点误差
        """
        epe = torch.sqrt(torch.sum((flow_pred - flow_gt) ** 2, dim=1))
        
        if mask is not None:
            epe = epe * mask.squeeze(1)
            return epe.sum() / (mask.sum() + 1e-8)
        else:
            return epe.mean()


def compute_codebook_usage(indices: torch.Tensor, n_embed: int) -> torch.Tensor:
    """
    计算Codebook使用率
    
    Args:
        indices: 量化索引 (B, H, W) or (B*H*W,)
        n_embed: Codebook大小
    
    Returns:
        usage: 使用率 [0, 1]
    """
    indices = indices.flatten()
    unique_indices = torch.unique(indices)
    usage = len(unique_indices) / n_embed
    return torch.tensor(usage, device=indices.device)
