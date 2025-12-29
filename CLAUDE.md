# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**UniT-Align** is a robotics tactile representation learning framework that combines visual and tactile sensing for robot manipulation tasks. The project focuses on learning generalizable tactile representations from simple objects that can be transferred to diverse downstream tasks (perception and policy learning).

The latest development on the `aligntac` branch introduces **AlignTac**, a novel dual-stream VQ-VAE architecture that explicitly disentangles geometry and dynamics features for precision assembly applications. AlignTac addresses a key limitation of the original UniT: while UniT excels at RGB reconstruction, precision assembly requires understanding both static geometry (shape/depth) and dynamic information (shear force/sliding/contact state).

**Key Technologies:** PyTorch, PyTorch Lightning, Hydra, VQ-VAE/VQ-GAN, Diffusion Policy

## Common Development Commands

### Environment Setup

```bash
# Create conda environment
cd UniT-Align
mamba env create -f conda_environment.yaml
bash install_custom_packages.sh

# Note: Edit install_custom_packages.sh to adjust conda path if not using ~/miniforge3
# Activate environment
conda activate unitenv
```

### Training Commands

All training uses Hydra configuration system. Main entry point is `train.py`.

```bash
# Representation learning (original UniT with VQ-VAE)
python train.py --config-dir=./UniT/config --config-name=vqvae_representation_key.yaml

# AlignTac dual-stream VQ-VAE training (NEW on aligntac branch)
python train.py --config-dir=./UniT/config --config-name=aligntac_representation.yaml

# MAE representation learning
python train.py --config-dir=./UniT/config --config-name=mae_representation_key.yaml

# Downstream perception tasks (3D pose estimation)
python train.py --config-dir=./UniT/config --config-name=key_vqvae_perception.yaml hydra.run.dir=data/outputs/your_folder_name

# Downstream perception (6D pose estimation)
python train.py --config-dir=./UniT/config --config-name=key_vqvae_perception_6D.yaml hydra.run.dir=data/outputs/your_folder_name

# Policy learning with tactile+vision
python train.py --config-dir=./UniT/config --config-name=train_legs_unit_policy.yaml hydra.run.dir=data/outputs/your_folder_name

# YCB-Sight classification (train representation first)
python train.py --config-dir=./UniT/config --config-name=vqvae_representation_ycb_0.yaml
python train.py --config-dir=./UniT/config --config-name=vqvae_perception_clf.yaml hydra.run.dir=data/outputs/your_folder_name
```

### Data Visualization

```bash
# Visualize specific episodes using rerun.io
EPISODES=-1 python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml
EPISODES="0,1,2" python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml

# Open visualization (results saved to data/<dataset_name>/debug.rrd)
rerun  # then File > Open RRD file
```

### Weights & Biases

```bash
# Login to wandb for perception/policy training logs
wandb login
```

## High-Level Architecture

### Three-Stage Training Pipeline

The project follows a transfer learning pipeline with three distinct stages:

1. **Stage 1: Representation Learning (Self-Supervised)**
   - Input: Tactile images from a single simple object (e.g., Allen key)
   - Model: VQ-VAE encoder + codebook
   - Purpose: Learn generalizable tactile features in a compact latent space
   - Output: Pre-trained encoder + codebook checkpoint
   - Location: Saved to `/representation_models`

2. **Stage 2: Perception Fine-Tuning (Downstream Tasks)**
   - Input: Frozen pre-trained encoder + task-specific datasets
   - Model: Add task-specific head (pose estimation/classification) on top of frozen encoder
   - Tasks: 3D pose, 6D pose, object classification
   - Output: Task-specific perception model
   - Location: Saved to `/data/outputs/perception_models`

3. **Stage 3: Policy Learning (Robot Manipulation)**
   - Input: Multi-modal observations (RGB cameras + frozen tactile encoder + proprioception)
   - Model: Diffusion-based policy combining vision (TimmObsEncoder) + tactile (frozen VQ-VAE)
   - Output: End-to-end manipulation policy
   - Purpose: Deploy to real robot for tasks like assembly, grasping, insertion

### AlignTac Dual-Stream Architecture (aligntac branch)

AlignTac introduces explicit disentanglement of geometry and dynamics:

```
Input: Dual-frame stack [I_t || I_{t-1}]  (6 channels)
         ↓
    Shared Encoder
         ↓
    ┌────────────────┐
    │                │
[Geometry Stream]  [Dynamics Stream]
    │                │
proj_geo          proj_dyn
    │                │
VQ Quantizer      VQ Quantizer
(geo_codebook)    (dyn_codebook)
    │                │
Decoder_RGB       Decoder_Flow
    │                │
RGB Recon.        Optical Flow Pred.
```

**Key Design Principles:**
- **Cross-modal prediction:** Input is RGB-only, but network must predict both RGB reconstruction AND optical flow
- **Physics internalization:** By forcing the network to predict flow from RGB texture changes alone, it learns to internalize the physics of force and deformation
- **Explicit disentanglement:** Separate codebooks + orthogonality constraint prevent feature collapse
- **Curriculum learning:** 3-phase training (RGB → RGB+Flow+Orth → RGB+Flow+Orth+Phase) prevents gradient conflicts

**Training Phases:**
- Phase 1 (0-30 epochs): RGB reconstruction only (warm-up visual features)
- Phase 2 (30-60 epochs): Add flow prediction + orthogonality constraint (inject dynamics)
- Phase 3 (60-100 epochs): Add phase classification (semantic state grounding)

### Configuration System (Hydra)

All hyperparameters live in YAML files under `UniT/config/`. Hydra enables dynamic class instantiation via `_target_` syntax:

```yaml
model:
  _target_: UniT.model.aligntac.aligntac_vqvae.AlignTacVQVAE
  base_learning_rate: 4.5e-6
  params:
    n_embed: 8192
    embed_dim: 512
```

**Configuration Categories:**
- Representation learning configs: `*_representation*.yaml` (6 files)
- Perception configs: `*_perception*.yaml` (14 files)
- Policy configs: `train_legs_*.yaml` (3 files)
- Visualization configs: `data_vis_*.yaml` (3 files)

### Core Modules

**UniT/model/** - Model architectures
- `vision/`: RGB encoder backbones (timm, transformers, multi-view fusion)
- `tactile/`: Tactile-specific components (`ConvPoolingPolicyHead` extracts features from VQ-VAE latents)
- `aligntac/`: NEW - AlignTac dual-stream VQ-VAE + multi-task losses

**UniT/dataset/** - Data loading (8 files)
- `aloha_dataset.py`: Main loader for manipulation demos (zarr/hdf5 format)
- `aligntac_dataset.py`: Dual-frame tactile dataset for AlignTac
- `tactile_representation_dataset.py`: Tactile-only for representation learning
- `tactile_perception_dataset.py`: Downstream perception tasks
- `representation_wrapper.py`: Wraps datasets with pre-trained encoders

**UniT/perception/** - Downstream perception models (6 files)
- Multiple tactile encoding architectures: VQ-VAE, MAE, BYOL, T3, ResNet, NoVQ
- Task heads: 3D pose (4D quaternion), 6D pose (3D position + 4D quaternion), classification

**UniT/policy/** - Manipulation policies (2 files)
- `diffusion_unit_policy.py`: Main UniT policy (vision + tactile, 303 lines)
- `diffusion_unet_timm_policy.py`: Vision-only baseline

**UniT/workspace/** - Training loops (8 files, PyTorch Lightning)
- `train_aligntac_workspace.py`: AlignTac training (NEW, 263 lines, single-GPU only)
- `train_vqvae_representation_workspace.py`: Standard VQ-VAE training
- `train_vqvae_perception_workspace.py`: Perception downstream tasks
- Policy/MAE/T3/BYOL training workspaces

**UniT/taming/** - VQ-VAE/VQ-GAN implementation (15 files)
- Adapted from taming-transformers library
- Core: `models/vqgan.py` (VQModel class), `modules/vqvae/quantize.py` (VectorQuantizer)
- Losses: `modules/losses/vqperceptual.py`, `lpips.py`

### Data Flow

```
Raw Data (zarr/hdf5)
    ↓
ReplayBuffer → SequenceSampler → Dataset → DataLoader
    ↓
Model Training (PyTorch Lightning)
    ↓
Checkpoints → Downstream Task Fine-tuning
    ↓
Policy Deployment (Real Robot)
```

**Data Modalities:**
- RGB images: cam_low, cam_high, wrist cameras (3×224×224)
- Tactile images: left/right GelSight Mini (3×128×160, cropped to remove marker-free edges)
- Proprioception: joint positions (qpos: 14D)
- Actions: robot commands (14D)

### DiffusionUnitPolicy Architecture

The policy combines multi-modal observations for action prediction:

```
Observations:
├─ RGB images → TimmObsEncoder → (D_vis,) visual features
├─ Left tactile → VQ-VAE.to_latent() → ConvPoolingHead → (512,) embedding
└─ Right tactile → VQ-VAE.to_latent() → ConvPoolingHead → (512,) embedding

Global Condition = concat(visual, left_tactile, right_tactile)

ConditionalUnet1D (denoises action sequence)
    ↓
Action Horizon (e.g., 16 future action steps)
```

**Key Parameters:**
- `obs_steps`: History observation steps (default: 2)
- `tactile_emb_dim`: Tactile embedding dimension (default: 512)
- `action_horizon`: Number of predicted action steps (default: 16)
- `latent_shape`: VQ-VAE latent spatial dimensions (default: [3, 16, 20])

## Important Implementation Notes

### Tactile Sensor Requirements

The pre-trained UniT model is designed for **GelSight Mini with markers**. If using different hardware:
- Use [gsmini](https://github.com/raghavauppuluri13/gsmini) for data streaming
- Use [tactile_recorder.py](UniT/utils/tactile_recorder.py) for consistent cropping
- Compare raw images with examples on [project website](https://zhengtongxu.github.io/unit-website/) to verify cropping consistency
- If cropping differs significantly, retrain representation from scratch with your data

### Working with Configurations

Each training job needs a unique output directory:
```bash
# Hydra will create the directory automatically
python train.py --config-dir=./UniT/config --config-name=<config> hydra.run.dir=data/outputs/unique_name
```

Always update checkpoint paths in YAML files when using pre-trained models:
```yaml
model:
  params:
    ckpt_path: /path/to/pretrained/model.ckpt
```

### AlignTac-Specific (aligntac branch)

AlignTac is currently **single-GPU only** (no distributed training). The training workspace explicitly disables DDP:
```python
# In train_aligntac_workspace.py
trainer = Trainer(
    strategy='auto',  # Will use single GPU, not DDP
    precision='16-mixed',
    max_epochs=100
)
```

**Loss Functions (UniT/model/aligntac/losses.py):**
- `AlignTacLoss`: Multi-task loss with uncertainty weighting (learnable σ parameters)
- `FlowLoss`: Charbonnier loss for optical flow prediction
- `PhaseClassificationLoss`: Cross-entropy for assembly phase classification

**Ground Truth Generation:**
- RGB reconstruction: Direct supervision from input frames
- Optical flow: Self-supervised using traditional algorithms (Farneback) or pre-trained RAFT
- Phase labels: Weak supervision from proprioception + heuristic state machine

**Curriculum Callbacks:** Automatically activate loss terms based on epoch:
```python
CurriculumLearningCallback:
  - Phase 1: Only L_rgb
  - Phase 2: L_rgb + L_flow + L_orth
  - Phase 3: All losses (L_rgb + L_flow + L_orth + L_phase)
```

### Debugging and Visualization

The codebase includes ImageLogCallback to log reconstructions periodically. For AlignTac, this logs:
- Original RGB frames
- Reconstructed RGB
- Predicted optical flow (as heatmaps)
- Ground truth optical flow

Use data_vis.py with rerun.io for interactive episode replay (cameras, tactile, proprioception, actions).

### Code Structure Patterns

**Model Definition Pattern:**
All models inherit from PyTorch Lightning's `LightningModule` and implement:
- `__init__()`: Network architecture definition
- `forward()`: Forward pass
- `training_step()`: Loss computation
- `validation_step()`: Validation metrics
- `configure_optimizers()`: Optimizer and scheduler setup

**Dataset Pattern:**
All datasets return dictionaries with standardized keys:
```python
{
    'obs': {
        'cam_low': (T, 3, H, W),
        'cam_high': (T, 3, H, W),
        'tactile_left': (T, 3, H, W),
        'tactile_right': (T, 3, H, W),
        'qpos': (T, 14)
    },
    'action': (T, 14)
}
```

## Dependencies and External Code

**External Repositories Referenced:**
- `UniT/taming/`: Adapted from [taming-transformers](https://github.com/CompVis/taming-transformers)
- `UniT/policy/`: Adapted from [UMI](https://github.com/real-stanford/universal_manipulation_interface)
- Training structure inspired by [diffusion_policy](https://github.com/real-stanford/diffusion_policy)

**Key Libraries:**
- PyTorch 1.12.1 + PyTorch Lightning 1.0.8
- Hydra 1.2.0 (configuration)
- timm 0.9.7 (vision backbones)
- Diffusers 0.11.1 (diffusion models)
- zarr 2.12.0 (data storage)
- wandb 0.13.3 (experiment tracking)
- rerun-sdk (visualization)

## Research Context

This codebase accompanies the paper:
- **Original UniT:** "UniT: Data Efficient Tactile Representation with Generalization to Unseen Objects" (IEEE RA-L 2025)
- **AlignTac (in development):** "AlignTac: Learning Geometry-Dynamics Aware Tactile Representations for Precise Assembly via Multi-Task Contrastive Pretraining" (proposed for IROS)

The AlignTac extension addresses a key limitation: while UniT excels at visual reconstruction, precision assembly requires understanding both geometry (depth/shape) and dynamics (shear force/sliding/contact state). AlignTac achieves this through explicit disentanglement with dual-stream architecture and physics-aware curriculum learning.

**Key Innovation:** AlignTac uses "single-modal input (RGB) → multi-modal output (RGB + Flow)" design to force the encoder to learn the physics behind texture changes rather than just memorizing patterns. This "physics internalization" is critical for precision assembly where small marker displacements represent crucial shear forces.
