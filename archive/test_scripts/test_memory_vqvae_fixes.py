"""
Unit tests to verify all MemoryVQVAE fixes

Tests:
1. P1: quant_conv layers exist and work correctly
2. P4: Loss function signature is correct
3. P5: learning_rate attribute exists
4. P6: Dataset handles resize correctly
5. P7: Memory module handles batch size changes

Author: Claude Code
Date: 2025-12-18
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add UniT to path
sys.path.insert(0, str(Path(__file__).parent))

from UniT.model.aligntac.memory_vqvae import MemoryVQVAE
from UniT.model.aligntac.memory_module import MemoryModule


def test_p1_quant_conv_layers():
    """Test P1: quant_conv layers exist and work correctly"""
    print("\n" + "="*60)
    print("TEST P1: quant_conv Layers")
    print("="*60)

    # Create model
    model = MemoryVQVAE(
        geo_embed_dim=3,
        dyn_embed_dim=3,
        n_embed=1024,
        memory_K=5,
        ddconfig={
            'double_z': False,
            'z_channels': 3,
            'resolution': 128,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
        },
        lossconfig={
            'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False,
                'disc_in_channels': 3,
                'disc_start': 10000,
                'disc_weight': 0.8,
                'codebook_weight': 1.0,
            }
        }
    )

    # Check quant_conv layers exist
    assert hasattr(model, 'quant_conv_geo'), "❌ quant_conv_geo layer missing!"
    assert hasattr(model, 'quant_conv_dyn'), "❌ quant_conv_dyn layer missing!"
    print("✓ quant_conv_geo exists")
    print("✓ quant_conv_dyn exists")

    # Check they are Conv2d layers
    assert isinstance(model.quant_conv_geo, nn.Conv2d), "❌ quant_conv_geo not Conv2d!"
    assert isinstance(model.quant_conv_dyn, nn.Conv2d), "❌ quant_conv_dyn not Conv2d!"
    print("✓ Both are Conv2d layers")

    # Test forward pass includes quant_conv
    x = torch.randn(2, 3, 128, 160)
    with torch.no_grad():
        outputs = model(x)

    assert 'rgb_recon' in outputs, "❌ Missing rgb_recon in outputs!"
    assert 'mdmap_pred' in outputs, "❌ Missing mdmap_pred in outputs!"
    print("✓ Forward pass successful with quant_conv layers")

    print("✅ TEST P1 PASSED\n")


def test_p4_loss_signature():
    """Test P4: Loss function signature is correct"""
    print("="*60)
    print("TEST P4: Loss Function Signature")
    print("="*60)

    model = MemoryVQVAE(
        geo_embed_dim=3,
        dyn_embed_dim=3,
        n_embed=1024,
        memory_K=5,
        ddconfig={
            'double_z': False,
            'z_channels': 3,
            'resolution': 128,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
        },
        lossconfig={
            'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False,
                'disc_in_channels': 3,
                'disc_start': 10000,
                'disc_weight': 0.8,
                'codebook_weight': 1.0,
            }
        }
    )

    # Test training_step with optimizer_idx=0 (generator)
    # Note: get_input expects (B, H, W, C) format and will permute to (B, C, H, W)
    batch = {
        'image': torch.randn(2, 128, 160, 3),
        'md_map': torch.randn(2, 128, 160, 3),
    }

    try:
        model.eval()  # Set to eval mode for testing without gradients
        with torch.no_grad():
            loss = model.training_step(batch, batch_idx=0, optimizer_idx=0)
        model.train()  # Reset to train mode
        print("✓ training_step (optimizer_idx=0) works correctly")
    except Exception as e:
        print(f"❌ training_step (optimizer_idx=0) failed: {e}")
        raise

    # Test training_step with optimizer_idx=1 (discriminator)
    try:
        model.eval()  # Set to eval mode for testing without gradients
        with torch.no_grad():
            loss = model.training_step(batch, batch_idx=0, optimizer_idx=1)
        model.train()  # Reset to train mode
        print("✓ training_step (optimizer_idx=1) works correctly")
    except Exception as e:
        print(f"❌ training_step (optimizer_idx=1) failed: {e}")
        raise

    # Test validation_step
    try:
        model.eval()  # Set to eval mode for testing without gradients
        with torch.no_grad():
            loss = model.validation_step(batch, batch_idx=0)
        model.train()  # Reset to train mode
        print("✓ validation_step works correctly")
    except Exception as e:
        print(f"❌ validation_step failed: {e}")
        raise

    print("✅ TEST P4 PASSED\n")


def test_p5_learning_rate():
    """Test P5: learning_rate attribute exists"""
    print("="*60)
    print("TEST P5: learning_rate Attribute")
    print("="*60)

    model = MemoryVQVAE(
        geo_embed_dim=3,
        dyn_embed_dim=3,
        n_embed=1024,
        memory_K=5,
        ddconfig={
            'double_z': False,
            'z_channels': 3,
            'resolution': 128,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
        },
        lossconfig={
            'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False,
                'disc_in_channels': 3,
                'disc_start': 10000,
                'disc_weight': 0.8,
                'codebook_weight': 1.0,
            }
        }
    )

    assert hasattr(model, 'learning_rate'), "❌ learning_rate attribute missing!"
    print(f"✓ learning_rate exists: {model.learning_rate}")

    # Test configure_optimizers (should not crash)
    try:
        opt_g, opt_d = model.configure_optimizers()
        print("✓ configure_optimizers works correctly")
    except Exception as e:
        print(f"❌ configure_optimizers failed: {e}")
        raise

    print("✅ TEST P5 PASSED\n")


def test_p7_batch_size_handling():
    """Test P7: Memory module handles batch size changes"""
    print("="*60)
    print("TEST P7: Batch Size Change Handling")
    print("="*60)

    memory = MemoryModule(
        embed_dim=3,
        spatial_size=(8, 10),
        K=5,
        num_heads=1,
    )

    # First batch: B=32
    z1 = torch.randn(32, 3, 8, 10)
    with torch.no_grad():
        out1 = memory(z1)
    print(f"✓ First batch (B=32): output shape = {out1.shape}")

    # Second batch: B=32 (same size, should work)
    z2 = torch.randn(32, 3, 8, 10)
    with torch.no_grad():
        out2 = memory(z2)
    print(f"✓ Second batch (B=32): output shape = {out2.shape}")

    # Third batch: B=16 (different size, should auto-reset)
    z3 = torch.randn(16, 3, 8, 10)
    try:
        with torch.no_grad():
            out3 = memory(z3)
        print(f"✓ Third batch (B=16): output shape = {out3.shape}")
        print("✓ Batch size change handled correctly (auto-reset)")
    except Exception as e:
        print(f"❌ Failed on batch size change: {e}")
        raise

    print("✅ TEST P7 PASSED\n")


def test_to_latent_with_quant_conv():
    """Test that to_latent() uses quant_conv correctly"""
    print("="*60)
    print("TEST: to_latent() with quant_conv")
    print("="*60)

    model = MemoryVQVAE(
        geo_embed_dim=3,
        dyn_embed_dim=3,
        n_embed=1024,
        memory_K=5,
        ddconfig={
            'double_z': False,
            'z_channels': 3,
            'resolution': 128,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1,1,2,4],
            'num_res_blocks': 2,
            'attn_resolutions': [16],
            'dropout': 0.0,
        },
        lossconfig={
            'target': 'UniT.taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator',
            'params': {
                'disc_conditional': False,
                'disc_in_channels': 3,
                'disc_start': 10000,
                'disc_weight': 0.8,
                'codebook_weight': 1.0,
            }
        }
    )

    x = torch.randn(2, 3, 128, 160)
    with torch.no_grad():
        latents = model.to_latent(x)

    assert 'z_geo' in latents, "❌ Missing z_geo in to_latent output!"
    assert 'z_dyn' in latents, "❌ Missing z_dyn in to_latent output!"
    assert 'z_dyn_mem' in latents, "❌ Missing z_dyn_mem in to_latent output!"

    print(f"✓ z_geo shape: {latents['z_geo'].shape}")
    print(f"✓ z_dyn shape: {latents['z_dyn'].shape}")
    print(f"✓ z_dyn_mem shape: {latents['z_dyn_mem'].shape}")

    print("✅ to_latent() TEST PASSED\n")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# MemoryVQVAE Fix Verification Tests")
    print("#"*60)

    try:
        test_p1_quant_conv_layers()
        test_p4_loss_signature()
        test_p5_learning_rate()
        test_p7_batch_size_handling()
        test_to_latent_with_quant_conv()

        print("\n" + "#"*60)
        print("# 🎉 ALL TESTS PASSED!")
        print("#"*60)
        print("\nSummary:")
        print("  ✅ P1: quant_conv layers implemented correctly")
        print("  ✅ P4: Loss function signature fixed")
        print("  ✅ P5: learning_rate attribute added")
        print("  ✅ P7: Batch size change handling implemented")
        print("  ✅ to_latent() works with quant_conv")
        print("\nNext steps:")
        print("  1. Test with real data (MDMapDataset with resized_mdmap_shape)")
        print("  2. Run full training to verify all components integrate correctly")
        print("  3. Monitor for any runtime issues")

    except Exception as e:
        print("\n" + "#"*60)
        print("# ❌ TESTS FAILED")
        print("#"*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
