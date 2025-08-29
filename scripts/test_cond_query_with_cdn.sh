#!/bin/bash
#SBATCH --job-name=test_cond_query_with_cdn
#SBATCH --output=out/test_cond_query_with_cdn.out
#SBATCH --error=out/test_cond_query_with_cdn.err
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-gpu=2
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gpu04

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

# Run the test script
echo "Running test script"
python tools/test.py \
    /media/lhbac13/AO2-FIX/configs/deformable_detr/deformable_detr_twostage_cdn_r50_16x2_50e_dota.py \
    /media/lhbac13/AO2-FIX/work_dirs/deformable_detr_twostage_cdn_r50_16x2_50e_dota/latest.pth \
    --format-only \
    --eval-options submission_dir=work_dirs/Task1_results

# Print end time
echo "Job ended at: $(date)"
