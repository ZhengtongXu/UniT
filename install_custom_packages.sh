#!/bin/bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate unitenv

cd ..
git clone https://github.com/alanzjl/t3
cd t3
pip install -e .
cd ..
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy
pip install -e .
cd ..
cd UniT
pip install -e .