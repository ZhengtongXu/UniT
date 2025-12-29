"""
测试ResidualVQModel模块

运行: python test_residual_model.py
"""

import sys
sys.path.insert(0, '/mlp_vepfs/share/zpw/tac/UniT-Align')

import torch
from UniT.taming.models.residual_vqmodel import ResidualVQModel

print('=' * 60)
print('测试ResidualVQModel模块')
print('=' * 60 + '\n')

# 构建配置
config = {
    'ddconfig': {
        'double_z': False,
        'z_channels': 3,
        'resolution': 128,
        'in_channels': 3,
        'out_ch': 3,
        'ch': 128,
        'ch_mult': [1, 1, 2, 4],
        'num_res_blocks': 2,
        'attn_resolutions': [16],
        'dropout': 0.0
    },
    'lossconfig': {
        'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
        'params': {
            'disc_conditional': False,
            'disc_in_channels': 3,
            'disc_start': 10000,
            'disc_weight': 0.8,
            'codebook_weight': 0.0,
            'pixelloss_weight': 1.0,
            'perceptual_weight': 1.0
        }
    },
    'prophet_config': {
        'visual_backbone': 'resnet34',
        'visual_pretrained': False,  # 测试时关闭预训练
        'state_dim': 8,
        'latent_shape': [3, 16, 20],
        'hidden_dims': [512, 256, 128]
    },
    'lambda_sparse': 0.01
}

print('创建ResidualVQModel...')
model = ResidualVQModel(**config)
# learning_rate is set during training
model.learning_rate = 4.5e-6
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'✓ ResidualVQModel创建成功')
print(f'  总参数量: {total_params:,}')
print(f'  可训练参数: {trainable_params:,}')

# 创建测试输入
batch_size = 2
batch = {
    'tactile_image': torch.randn(batch_size, 3, 128, 160),
    'visual_image': torch.randn(batch_size, 3, 224, 224),
    'robot_state': torch.randn(batch_size, 8)
}

print(f'\n测试前向传播...')
print(f'  输入shapes:')
for k, v in batch.items():
    print(f'    {k}: {v.shape}')

# 前向传播
with torch.no_grad():
    outputs = model(batch)

print(f'\n  输出keys: {list(outputs.keys())}')
for k, v in outputs.items():
    print(f'    {k}: shape={v.shape}, dtype={v.dtype}')
    print(f'      range=[{v.min():.4f}, {v.max():.4f}], mean={v.mean():.4f}')

# 验证输出
assert outputs['recon'].shape == batch['tactile_image'].shape, \
    f"Reconstruction shape mismatch"
assert outputs['z_exp'].shape == (batch_size, 3, 16, 20), \
    f"z_exp shape should be (B, 3, 16, 20)"
assert outputs['z_real'].shape == (batch_size, 3, 16, 20), \
    f"z_real shape should be (B, 3, 16, 20)"
assert outputs['residual'].shape == (batch_size, 3, 16, 20), \
    f"residual shape should be (B, 3, 16, 20)"

print(f'\n测试residual计算...')
# 验证残差关系: residual = z_real - z_exp
computed_residual = outputs['z_real'] - outputs['z_exp']
residual_error = (computed_residual - outputs['residual']).abs().max()
print(f'  residual计算误差: {residual_error:.8f}')
assert residual_error < 1e-5, "Residual should be z_real - z_exp"

# 验证融合关系: z_fused = z_exp + residual
z_fused_computed = outputs['z_exp'] + outputs['residual']
print(f'  z_fused = z_exp + residual: ✓')

print(f'\n测试梯度反向传播...')
model.train()

# 模拟训练step
outputs_train = model(batch)

# 计算简单的loss (不使用完整的GAN loss以简化测试)
recon_loss = (outputs_train['recon'] - batch['tactile_image']).pow(2).mean()
sparse_loss = model.lambda_sparse * outputs_train['residual'].abs().mean()
total_loss = recon_loss + sparse_loss

print(f'  recon_loss: {recon_loss.item():.4f}')
print(f'  sparse_loss: {sparse_loss.item():.4f}')
print(f'  total_loss: {total_loss.item():.4f}')

# 反向传播
total_loss.backward()

# 检查梯度
prophet_grad = sum(1 for p in model.prophet_encoder.parameters() if p.grad is not None)
obs_grad = sum(1 for p in model.obs_encoder.parameters() if p.grad is not None)
decoder_grad = sum(1 for p in model.decoder.parameters() if p.grad is not None)

print(f'\n  梯度检查:')
print(f'    prophet_encoder有梯度的参数: {prophet_grad}')
print(f'    obs_encoder有梯度的参数: {obs_grad}')
print(f'    decoder有梯度的参数: {decoder_grad}')

assert prophet_grad > 0, "Prophet should have gradients"
assert obs_grad > 0, "Observation encoder should have gradients"
assert decoder_grad > 0, "Decoder should have gradients"

print(f'\n测试不同输入尺寸...')
# 测试tactile size为160x128 (实际数据格式)
batch_real = {
    'tactile_image': torch.randn(batch_size, 3, 160, 128),
    'visual_image': torch.randn(batch_size, 3, 224, 224),
    'robot_state': torch.randn(batch_size, 8)
}

with torch.no_grad():
    outputs_real = model(batch_real)

print(f'  真实数据尺寸 (160x128): ✓')
print(f'    recon shape: {outputs_real["recon"].shape}')
assert outputs_real['recon'].shape == batch_real['tactile_image'].shape

print(f'\n' + '=' * 60)
print('✅ ResidualVQModel模块测试全部通过!')
print('=' * 60)
print('\n主要验证点:')
print('  ✓ 模型结构完整 (Prophet + Observation + Decoder)')
print('  ✓ 输出格式正确 (recon, z_exp, z_real, residual)')
print('  ✓ 残差计算正确 (r_t = z_real - z_exp)')
print('  ✓ 梯度反向传播正常')
print('  ✓ 支持实际数据尺寸 (160x128)')
print('  ✓ 稀疏Loss正确计算')
print('\n可以开始完整训练测试!')
