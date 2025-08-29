#!/bin/bash
#SBATCH --job-name=train_twostage_cdn_ms_liemchinhkhoahoc
#SBATCH --output=out/train_twostage_cdn_ms_liemchinhkhoahoc.out
#SBATCH --error=out/train_twostage_cdn_ms_liemchinhkhoahoc.err
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:3
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=gpu01

# Print start time and node info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Check CUDA version
echo "CUDA version:"
nvcc --version

# Check GPU status
echo "Checking GPU status with nvidia-smi:"
nvidia-smi

# Activate the conda environment
echo "Activating environment 'ao2_fix'"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ao2_fix

# Check Python version
echo "Python version:"
python --version

# Uncomment the following lines when you're ready to run the training script
echo "Running training script"
python tools/train.py /media/lhbac13/AO2-FIX/configs/deformable_detr/deformable_detr_cdn_r50_16x2_50e_dota_ms_liemchinhkhoahoc.py --work-dir work_dirs/deformable_detr_cdn_r50_16x2_50e_dota_ms_liemchinhkhoahoc

echo "Job ended at: $(date)"