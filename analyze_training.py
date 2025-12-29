"""
分析Residual VQ-VAE训练日志

从WandB binary文件中提取训练曲线并可视化
"""

import struct
import json
import pandas as pd
import numpy as np
from pathlib import Path

def parse_wandb_binary(wandb_path):
    """解析wandb二进制文件 (简化版)"""
    # 读取summary作为最终结果
    summary_path = Path(wandb_path).parent / 'files' / 'wandb-summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    return summary

def extract_from_output_log(log_path):
    """从output.log提取训练曲线"""
    data = {
        'epoch': [],
        'train_loss': [],
        'train_recon_loss': [],
        'train_aux_weight': [],
        'train_sparse_loss': []
    }

    with open(log_path, 'r') as f:
        for line in f:
            # 解析进度条中的epoch和loss
            if 'Epoch' in line and 'loss=' in line:
                try:
                    # 提取epoch
                    epoch_str = line.split('Epoch')[1].split(':')[0].strip()
                    if epoch_str.isdigit():
                        epoch = int(epoch_str)
                    else:
                        continue

                    # 提取loss
                    loss_str = line.split('loss=')[1].split(',')[0].strip()
                    loss = float(loss_str)

                    data['epoch'].append(epoch)
                    data['train_loss'].append(loss)

                    # 尝试提取其他指标
                    if 'train/recon_loss=' in line:
                        recon_str = line.split('train/recon_loss=')[1].split(',')[0].strip()
                        data['train_recon_loss'].append(float(recon_str))
                    else:
                        data['train_recon_loss'].append(None)

                    if 'train/aux_weight=' in line:
                        aux_str = line.split('train/aux_weight=')[1].split(',')[0].strip()
                        data['train_aux_weight'].append(float(aux_str))
                    else:
                        data['train_aux_weight'].append(None)

                    if 'train/sparse_loss=' in line:
                        sparse_str = line.split('train/sparse_loss=')[1].split(',')[0].strip()
                        data['train_sparse_loss'].append(float(sparse_str))
                    else:
                        data['train_sparse_loss'].append(None)

                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(data)

def analyze_training(run_dir):
    """分析训练运行"""
    run_path = Path(run_dir)

    print("=" * 80)
    print(f"分析训练运行: {run_path.name}")
    print("=" * 80)

    # 1. 读取最终summary
    summary_path = run_path / 'files' / 'wandb-summary.json'
    with open(summary_path, 'r') as f:
        summary = json.load(f)

    print("\n【关键发现】训练最终状态 (Epoch 100):")
    print("-" * 80)

    # VQ Loss分析
    train_vq = summary.get('train/vq_loss', 0)
    val_vq = summary.get('val/vq_loss', 0)
    print(f"VQ Loss (训练): {train_vq:.2f}")
    print(f"VQ Loss (验证): {val_vq:.2f}")
    print(f"→ 训练/验证比: {train_vq/val_vq:.3f} (应该接近1.0, 当前过拟合)")

    # 残差范数分析
    train_res_raw = summary.get('train/raw_residual_norm', 0)
    train_res_vq = summary.get('train/residual_norm', 0)
    val_res_raw = summary.get('val/raw_residual_norm', 0)
    print(f"\n残差范数 (原始, 训练): {train_res_raw:.2f}")
    print(f"残差范数 (VQ后, 训练): {train_res_vq:.2f}")
    print(f"残差范数 (原始, 验证): {val_res_raw:.2f}")
    print(f"→ 量化压缩率: {train_res_vq/train_res_raw:.3f} (VQ将残差压缩了{100*(1-train_res_vq/train_res_raw):.1f}%)")

    # 重建Loss分析
    train_recon = summary.get('train/recon_loss', 0)
    train_recon_aux = summary.get('train/recon_loss_aux', 0)
    val_recon = summary.get('val/rec_loss', 0)
    val_recon_aux = summary.get('val/recon_loss_aux', 0)
    print(f"\n重建Loss (Hybrid, 训练): {train_recon:.4f}")
    print(f"重建Loss (Aux, 训练): {train_recon_aux:.4f}")
    print(f"重建Loss (Hybrid, 验证): {val_recon:.4f}")
    print(f"重建Loss (Aux, 验证): {val_recon_aux:.4f}")
    print(f"→ Aux/Hybrid比 (训练): {train_recon_aux/train_recon:.2f} (Aux更低 = obs_encoder质量好)")
    print(f"→ Aux/Hybrid比 (验证): {val_recon_aux/val_recon:.2f}")

    # Sparse Loss分析
    sparse_loss = summary.get('train/sparse_loss', 0)
    lambda_sparse = 0.001
    avg_raw_residual = sparse_loss / lambda_sparse
    print(f"\nSparse Loss: {sparse_loss:.4f}")
    print(f"→ 平均残差L1范数: {avg_raw_residual:.2f} (={sparse_loss:.4f}/{lambda_sparse})")

    # 判别器状态
    train_disc = summary.get('train/disc_loss', 0)
    train_g_loss = summary.get('train/g_loss', 0)
    print(f"\n判别器Loss (训练): {train_disc:.4f}")
    print(f"生成器对抗Loss: {train_g_loss:.4f}")
    print(f"→ 判别器状态: {'正常工作' if 0.5 < train_disc < 1.5 else '可能崩溃/饱和'}")

    # 2. 从output.log提取曲线
    log_path = run_path / 'files' / 'output.log'
    if log_path.exists():
        print(f"\n提取训练曲线...")
        df = extract_from_output_log(log_path)

        # 检测异常
        print(f"\n【异常检测】")
        print("-" * 80)

        # Loss爆炸检测
        max_loss = df['train_loss'].max()
        min_loss = df['train_loss'].min()
        final_loss = df['train_loss'].iloc[-1] if len(df) > 0 else 0

        print(f"Loss范围: {min_loss:.2f} → {max_loss:.2f}")
        print(f"最终Loss: {final_loss:.2f}")

        if max_loss > 1000:
            print(f"⚠️  WARNING: Loss爆炸到 {max_loss:.1f}!")

            # 找到爆炸开始的epoch
            explosion_epoch = df[df['train_loss'] > 100]['epoch'].min()
            print(f"   → Loss爆炸开始于 Epoch {explosion_epoch}")

            # 检查是否与课程学习切换相关
            if 28 <= explosion_epoch <= 32:
                print(f"   → 🔥 关键发现: 爆炸发生在Epoch 30课程切换时 (aux_weight: 1.0→0.1)")
                print(f"   → 这表明Decoder无法适应VQ量化噪声!")

        # 保存数据
        output_csv = Path(run_dir).parent / 'training_curve.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n训练曲线已保存到: {output_csv}")

        return df, summary

    return None, summary

if __name__ == "__main__":
    run_dir = "/root/code/zpw/tac/UniT-Align/representation_models/2025-12-25T11-48-42residual_vqvae_v3_anti_overfit/wandb/run-20251225_114843-3ijlxpfh"
    df, summary = analyze_training(run_dir)

    if df is not None:
        print(f"\n提取到 {len(df)} 个训练点")
        print("\n前10个epoch:")
        print(df.head(10))
        print("\nEpoch 25-35 (课程切换前后):")
        print(df[(df['epoch'] >= 25) & (df['epoch'] <= 35)])
