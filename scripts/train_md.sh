#!/bin/bash
#SBATCH --job-name=train_md
#SBATCH --output=out/train_md.out
#SBATCH --error=out/train_md.err
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
echo "Activating environment 'ao2_base'"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ao2_base

# Check Python version
echo "Python version:"
python --version

# # Check PyTorch and CUDA availability
# echo "Checking PyTorch and CUDA:"
# python << END
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# print(f"CUDA version: {torch.version.cuda}")
# if torch.cuda.is_available():
#     print(f"GPU device name: {torch.cuda.get_device_name(0)}")
# END

# Uncomment the following lines when you're ready to run the training script
echo "Running training script"
python tools/train.py /media/lhbac13/AO2/configs/deformable_detr/matching_degree_loss.py --work-dir work_dirs/matching_degree_loss

echo "Job ended at: $(date)"
