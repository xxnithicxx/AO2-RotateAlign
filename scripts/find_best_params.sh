#!/bin/bash
#SBATCH --job-name=find_best_params
#SBATCH --output=out/find_best_params.out
#SBATCH --error=out/find_best_params.err
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
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
python tools/train.py /media/lhbac13/AO2-FIX/configs/deformable_detr/deformable_detr_cdn_r50_16x2_50e_dota_params.py --work-dir work_dirs/best_params

echo "Job ended at: $(date)"
