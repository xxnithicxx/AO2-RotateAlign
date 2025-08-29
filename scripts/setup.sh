#!/bin/bash
#SBATCH --job-name=setup_ao2_fix
#SBATCH --output=out/setup_ao2_fix.out
#SBATCH --error=out/setup_ao2_fix.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:00
#SBATCH --nodelist=gpu01

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Checking GPU status with nvidia-smi:"
nvidia-smi

# Tạo môi trường
echo "Creating conda environment 'ao2_fix' with Python 3.10"
conda create -y -n ao2_fix python=3.10

# Kích hoạt môi trường
echo "Activating environment 'ao2_fix'"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ao2_fix

# Cài đặt PyTorch cho CUDA 12.3
echo "Installing PyTorch and related packages with CUDA 12.3"
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Cài đặt MMCV cho CUDA 12.1 (tương thích ngược với 12.3)
echo "Installing mmcv-full"
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# Các phụ thuộc khác
echo "Installing mmdet"
pip install mmdet

echo "Installing mmrotate build requirements"
pip install -r requirements/build.txt

echo "Installing mmrotate in editable mode"
pip install -v -e .

echo "Job finished at: $(date)"
