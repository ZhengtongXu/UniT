# UniT: Unified Tactile Representation for Robot Learning 
### Zhengtong Xu, Raghava Uppuluri, Xinwei Zhang, Cael Fitch, Philip Glen Crandall, Wan Shou, Dongyi Wang, [Yu She](https://www.purduemars.com/home)

_Purdue University_ and _University of Arkansas_

[Project](https://zhengtongxu.github.io/unifiedtactile.github.io/) | [arXiv]() | [Summary Video]()

---
UniT is a novel approach to tactile representation learning, using VQVAE to learn a compact latent space and serve as the tactile representation. It uses tactile images obtained from a single simple object to train the representation with transferability and generalizability. This tactile representation can be zero-shot transferred to various downstream tasks, including perception tasks and manipulation policy learning. Our benchmarking on an in-hand 3D pose estimation task shows that UniT outperforms existing visual and tactile representation learning methods. Additionally, UniT's effectiveness in policy learning is demonstrated across three real-world tasks involving diverse manipulated objects and complex robot-object-environment interactions. Through extensive experimentation, UniT is shown to be a simple-to-train, plug-and-play, yet widely effective method for tactile representation learning.
---

## Dataset

We have released the dataset for representation learning, in-hand 3D pose estimation, and manipulation policy learning at this [link](https://drive.google.com/drive/folders/1CkPqgNFCE6B1mr2pxYdNdSR-xAkSnxQc?usp=sharing).

## Installation

For installation, please run

```console
$ cd UniT
$ mamba env create -f conda_environment.yaml && bash install_custom_packages.sh
```

## Representation Training

Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).
```console
$ conda activate unitenv
$ wandb login
```

For example, launch representation training with UniT and Allen key dataset
```console
$ python train.py --config-dir=./UniT/config --config-name=vqvae_representation_key.yaml
```
The pretrained models will be saved in /representation_models

Similarly, launch representation training with MAE and Allen key dataset
```console
$ python train.py --config-dir=./UniT/config --config-name=mae_representation_key.yaml
```

We provide multiple YAML files as examples for representation learning, covering all the methods benchmarked in our paper. Feel free to explore and experiment with them!


## Tactile Perception Training
All logs from tactile perception training will be uploaded to wandb.

For example, deploy the trained UniT representation to the in-hand 3D pose estimation task, just run
```console
$ python train.py --config-dir=./UniT/config --config-name=key_vqvae_perception.yaml hydra.run.dir=data/outputs/your_folder_name
```
Please note that you need to specify the path to the pretrained VQVAE in the YAML config file.

Similary, you can run other config files to launch in-hand 3D pose estimation training with all different methods.

## Policy Training
For example, to train the diffusion policy with UniT for the chicken legs hanging task, run
```console
$ python train.py --config-dir=./UniT/config --config-name=train_legs_unit_policy.yaml hydra.run.dir=data/outputs/your_folder_name
```

## Human Demonstration Data Visualization
We have set up a visualization script using [rerun.io](https://rerun.io/).

Run the visualization for any episode index in the chicken hanging task

```console
# visualize last episode
$ EPISODES=-1 python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml
# visualize first 3 episodes
$ EPISODES="0,1,2" python data_vis.py --config-name data_vis_chickenlegs_hanging.yaml
```

Results will be saved to `data/<dataset_name>/debug.rrd` folder
Open the `debug.rrd` file with rerun (on linux machine)

```console
$ rerun
```
Then
```
File > Open RRD file
```

## Acknowledgement

This repository, during construction, referenced the code structure of [diffusion policy](https://github.com/real-stanford/diffusion_policy). We sincerely thank the authors of diffusion policy for open-sourcing such an elegant codebase!