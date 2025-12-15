"""
=============================================================================
向量量化(Vector Quantization)模块实现
=============================================================================
这是VQ-VAE的核心组件,实现了将连续latent向量映射到离散codebook的算法

核心算法:
1. 距离计算: 计算encoder输出与codebook中所有向量的欧氏距离
2. 最近邻搜索: 为每个latent向量找到最近的codebook向量
3. Straight-Through Estimator: 量化后保留梯度传播能力

关键参数:
- n_e: Codebook大小(如1024个向量)
- e_dim: 每个code的维度(如3)
- beta: Commitment loss权重(让encoder输出接近codebook)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange


class VectorQuantizer(nn.Module):
    """
    基础版本的向量量化器
    
    警告: 此类包含beta项的bug,推荐使用VectorQuantizer2
    
    算法流程:
    1. 计算encoder输出z与所有codebook向量的距离
    2. 选择最近的code作为量化结果z_q
    3. 使用Straight-Through Estimator传递梯度
    
    参数:
        n_e: Codebook向量数量
        e_dim: Code维度
        beta: Commitment loss权重
    """
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e  # Codebook大小,如1024
        self.e_dim = e_dim  # Code维度,如3
        self.beta = beta  # Commitment loss权重

        # Codebook: (n_e, e_dim)的可学习embedding表
        # 例如: (1024, 3) - 1024个3维向量
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        # 初始化为[-1/n_e, 1/n_e]的均匀分布
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        """
        前向传播: 连续向量 → 离散向量
        
        输入:
            z: Encoder输出 (B, C, H, W)
               例如: (32, 3, 8, 10) - 32张触觉图的latent
        
        输出:
            z_q: 量化后的latent (B, C, H, W)
            loss: 量化损失(重建loss + commitment loss)
            (perplexity, min_encodings, min_encoding_indices)
        
        量化流程:
        1. 展平: (B, C, H, W) → (B*H*W, C)
        2. 距离计算: ||z - e||² = ||z||² + ||e||² - 2z·e
        3. 最近邻: argmin distance
        4. 查表: 用最近code替换z
        5. STE: 保留梯度
        """
        # 步骤1: 调整维度 BCHW → BHWC
        # (32, 3, 8, 10) → (32, 8, 10, 3)
        z = z.permute(0, 2, 3, 1).contiguous()
        
        # 步骤2: 展平为二维 (B*H*W, C)
        # (32, 8, 10, 3) → (2560, 3)
        z_flattened = z.view(-1, self.e_dim)
        
        # 步骤3: 计算欧氏距离 ||z - e||²
        # 使用公式: ||z-e||² = ||z||² + ||e||² - 2z·e
        # d.shape: (B*H*W, n_e) = (2560, 1024)
        # 每行是一个latent到所有code的距离
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # 步骤4: 找到最近的code索引
        # min_encoding_indices.shape: (2560, 1)
        # 每个latent对应一个code索引 [0-1023]
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)

        # 步骤5: 转换为one-hot编码
        # min_encodings.shape: (2560, 1024)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # 步骤6: 查表获得量化向量
        # z_q = min_encodings @ embedding.weight
        # (2560, 1024) @ (1024, 3) → (2560, 3)
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # 步骤7: 计算量化损失
        # L = ||sg[z] - e||² + β||z - sg[e]||²
        # 第一项: 让codebook靠近encoder输出(detach z)
        # 第二项: commitment loss,让encoder输出靠近codebook(detach z_q)
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # 步骤8: Straight-Through Estimator (STE)
        # 前向: 使用量化值z_q
        # 反向: 梯度直接传给z(绕过argmin)
        # 相当于: z_q = z + (z_q - z).detach()
        z_q = z + (z_q - z).detach()

        # 步骤9: 计算codebook使用率(perplexity)
        # 衡量code的使用是否均匀
        # perplexity越高越好,说明code都被用到了
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # 步骤10: 恢复原始形状 BHWC → BCHW
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        根据code索引查表获得向量
        
        用途: 可视化每个code代表的模式
        """
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)
        
        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q


class VectorQuantizer2(nn.Module):
    """
    ★★★ 推荐使用的向量量化器 ★★★
    改进版本,修复了VectorQuantizer的bug,效率更高
    
    主要改进:
    1. 修复beta项的bug(legacy=False时生效)
    2. 避免昂贵的矩阵乘法
    3. 支持codebook remapping(只使用部分code)
    4. 支持多种索引形状
    
    在UniT中的使用:
    - VQModel使用此类进行触觉latent的量化
    - 训练时学习1024个3维code
    - 推理时提取连续特征(不量化)
    
    参数:
        n_e: Codebook大小(如1024)
        e_dim: Code维度(如3)
        beta: Commitment loss权重(如0.25)
        remap: 是否重映射code(可选)
        unknown_index: 未知索引处理方式
        sane_index_shape: 是否使用(B,H,W)索引形状
        legacy: 是否使用旧版beta(默认True保持兼容)
    """
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e  # Codebook大小
        self.e_dim = e_dim  # Code维度
        self.beta = beta  # Commitment loss权重
        self.legacy = legacy  # 是否使用旧版beta计算

        # Codebook: 可学习的embedding表
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Codebook remapping: 只使用部分code
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        """将索引重映射到实际使用的code"""
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        """将重映射的索引恢复到原始空间"""
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]:
            inds[inds>=self.used.shape[0]] = 0
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        """
        ★★★ 核心量化算法 ★★★
        将连续latent量化到离散codebook
        
        输入:
            z: Encoder输出 (B, C, H, W)
               例如: (32, 3, 8, 10)
        
        输出:
            z_q: 量化后的latent (32, 3, 8, 10)
            loss: 量化损失标量
            (perplexity, min_encodings, min_encoding_indices)
        
        算法详解:
        ==========================================
        步骤1: 维度调整
        ------------------------------------------
        z: (B, C, H, W) → (B, H, W, C)
        目的: 将通道维放到最后,方便后续处理
        
        步骤2: 展平为二维
        ------------------------------------------
        z: (B, H, W, C) → (B*H*W, C)
        例如: (32, 8, 10, 3) → (2560, 3)
        目的: 将每个空间位置的向量独立处理
        
        步骤3: 距离计算
        ------------------------------------------
        公式: d = ||z - e||² = ||z||² + ||e||² - 2z·e
        
        ||z||²: (2560, 1) - 每个latent的模长平方
        ||e||²: (1024,) - 每个code的模长平方
        2z·e: (2560, 3) @ (3, 1024) = (2560, 1024)
        
        结果: d.shape = (2560, 1024)
        每行是一个latent到所有code的距离
        
        步骤4: 最近邻搜索
        ------------------------------------------
        min_encoding_indices = argmin(d, dim=1)
        shape: (2560,) 每个latent的最近code索引
        
        步骤5: 查表
        ------------------------------------------
        z_q = embedding(min_encoding_indices)
        用最近的code向量替换原latent
        
        步骤6: 量化损失
        ------------------------------------------
        如果legacy=False(推荐):
            loss = ||z - sg[e]||² + β||sg[z] - e||²
            第一项: 让encoder靠近codebook
            第二项: 让codebook靠近encoder
        
        如果legacy=True(兼容旧版):
            loss = β||z - sg[e]||² + ||sg[z] - e||²
            (beta项位置错误,但为了兼容性保留)
        
        步骤7: Straight-Through Estimator
        ------------------------------------------
        z_q = z + (z_q - z).detach()
        
        为什么需要STE?
        - argmin操作不可导
        - 需要绕过离散化传递梯度
        - STE让梯度"假装"量化是恒等映射
        
        前向传播: 使用量化值z_q(离散)
        反向传播: 梯度直接传给z(连续)
        
        步骤8: 恢复形状
        ------------------------------------------
        z_q: (2560, 3) → (32, 8, 10, 3) → (32, 3, 8, 10)
        ==========================================
        """
        # 步骤1: BCHW → BHWC
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        
        # 步骤2: 计算距离矩阵
        # ||z - e||² = ||z||² + ||e||² - 2z·e
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        # 步骤3: 最近邻搜索
        min_encoding_indices = torch.argmin(d, dim=1)
        
        # 步骤4: 查表获得量化向量
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # 步骤5: Codebook使用率统计
        perplexity = None
        min_encodings = None

        # 步骤6: 计算量化损失
        if not self.legacy:
            # 推荐的损失计算(修复beta bug)
            # L = ||z - sg[e]||² + β||sg[z] - e||²
            loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            # 旧版损失计算(保持兼容性)
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # 步骤7: Straight-Through Estimator
        # 这是VQ-VAE的核心trick!
        # 前向: 用z_q(离散)
        # 反向: 梯度传给z(连续)
        z_q = z + (z_q - z).detach()

        # 步骤8: 恢复原始形状 BHWC → BCHW
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        # 步骤9: 处理索引形状
        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_codebook_entry(self, indices, shape):
        """
        根据code索引查表获得向量
        
        用途:
        1. 可视化: 看每个code代表什么触觉模式
        2. 生成: 从离散索引直接解码
        
        输入:
            indices: Code索引 (N,)
            shape: 目标形状 (B, H, W, C)
        
        输出:
            z_q: 量化向量 (B, C, H, W)
        """
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        # 从embedding表查找
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class GumbelQuantize(nn.Module):
    """
    Gumbel-Softmax量化器
    
    优点:
    - 可微分的离散化
    - 训练更稳定
    - 使用温度退火控制离散程度
    
    算法:
    1. 计算logits: z → Conv → logits (B, n_embed, H, W)
    2. Gumbel-Softmax采样: logits → soft_one_hot
    3. 加权求和: soft_one_hot @ embedding → z_q
    4. KL散度正则化: 让分布更均匀
    
    参数:
        num_hiddens: Encoder输出通道数
        embedding_dim: Code维度
        n_embed: Codebook大小
        kl_weight: KL散度权重
        temp_init: 初始温度
    """
    def __init__(self, num_hiddens, embedding_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init  # 温度参数
        self.kl_weight = kl_weight

        # 将latent映射到logits
        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        # Codebook
        self.embed = nn.Embedding(n_embed, embedding_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def forward(self, z, temp=None, return_logits=False):
        """
        Gumbel-Softmax量化
        
        流程:
        1. z → proj → logits (B, n_embed, H, W)
        2. Gumbel-Softmax采样
        3. soft_one_hot @ embedding → z_q
        4. 计算KL散度损失
        
        温度控制:
        - 高温: 软分布,接近连续
        - 低温: 硬分布,接近离散
        - 训练中逐渐降温(annealing)
        """
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        # 步骤1: 计算logits
        logits = self.proj(z)
        if self.remap is not None:
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        # 步骤2: Gumbel-Softmax采样
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        
        # 步骤3: 加权求和得到量化向量
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # 步骤4: KL散度正则化
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        if self.use_vqinterface:
            if return_logits:
                return z_q, diff, (None, None, ind), logits
            return z_q, diff, (None, None, ind)
        return z_q, diff, ind

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class EMAVectorQuantizer(nn.Module):
    """
    使用EMA(指数移动平均)更新codebook的量化器
    
    优点:
    1. Codebook通过EMA更新,无需梯度
    2. 训练更稳定,避免codebook collapse
    3. 适合大规模codebook
    
    EMA更新公式:
    N_i = γ·N_i + (1-γ)·n_i  (code使用次数)
    m_i = γ·m_i + (1-γ)·Σz_i  (code向量和)
    e_i = m_i / N_i  (更新后的code)
    
    参数:
        n_embed: Codebook大小
        embedding_dim: Code维度
        commitment_cost: Commitment loss权重
        decay: EMA衰减率(如0.99)
        epsilon: 数值稳定项
    """
    def __init__(self, n_embed, embedding_dim, commitment_cost=0.25, decay=0.99,
                 epsilon=1e-5):
        super(EMAVectorQuantizer, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook embedding
        embed = torch.randn(n_embed, embedding_dim)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

        self.n_embed = n_embed
        self.embedding_dim = embedding_dim

    def forward(self, z):
        """
        EMA量化
        
        训练时:
        1. 量化(最近邻搜索)
        2. EMA更新codebook
        3. 只计算commitment loss
        
        推理时:
        1. 量化(最近邻搜索)
        2. 无需更新codebook
        """
        # 维度调整
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.embedding_dim)

        # 距离计算
        d = (z_flattened.pow(2).sum(1, keepdim=True)
             - 2 * z_flattened @ self.embed.t()
             + self.embed.pow(2).sum(1, keepdim=True).t())

        # 最近邻搜索
        encoding_indices = torch.argmin(d, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embed).type(z_flattened.dtype)
        encoding_indices = encoding_indices.view(*z.shape[:-1])

        # 查表量化
        quantized = F.embedding(encoding_indices, self.embed).view(z.shape)

        # EMA更新codebook(仅训练时)
        if self.training:
            # 统计code使用次数
            self.cluster_size.data.mul_(self.decay).add_(
                encodings.sum(0), alpha=1 - self.decay
            )

            # 累加code向量和
            embed_sum = z_flattened.t() @ encodings
            self.embed_avg.data.mul_(self.decay).add_(embed_sum.t(), alpha=1 - self.decay)

            # 更新codebook
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.epsilon) / (n + self.n_embed * self.epsilon) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            self.embed.data.copy_(embed_normalized)

        # Commitment loss(只约束encoder)
        loss = self.commitment_cost * F.mse_loss(quantized.detach(), z)

        # Straight-Through Estimator
        quantized = z + (quantized - z).detach()

        # 恢复形状
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        return quantized, loss, (None, None, encoding_indices)


"""
=============================================================================
总结: 三种量化器对比
=============================================================================

1. VectorQuantizer2 (推荐)
   - 标准VQ-VAE算法
   - 通过梯度优化codebook
   - 适合中小规模codebook
   - UniT使用此版本

2. GumbelQuantize
   - 可微分离散化
   - 训练更稳定
   - 需要温度退火
   - 适合需要端到端微分的场景

3. EMAVectorQuantizer
   - 通过EMA更新codebook
   - 训练最稳定
   - 适合大规模codebook
   - 避免codebook collapse

在UniT中的使用:
- VQModel默认使用VectorQuantizer2
- beta=0.25, n_e=1024, e_dim=3
- 训练触觉VQ-VAE预训练模型
- 推理时使用to_latent()提取连续特征(不量化)
=============================================================================
"""
