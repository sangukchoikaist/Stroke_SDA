#!/bin/bash
# ==============================================================
# SLURM JOB SCRIPT for SDA Grid Search
# Adapted from Server Template (server1)
# ==============================================================

#SBATCH -J SDA_Grid               # Job name
#SBATCH -N 1                      # Number of nodes
#SBATCH -n 1                      # Number of tasks
#SBATCH --gres=gpu:1              # GPU allocation
#SBATCH --cpus-per-task=8         # CPU allocation (8 cores for parallel workers)
#SBATCH --mem=32G                 # Memory allocation (32GB)
#SBATCH -t 48:00:00               # Max runtime
#SBATCH -o logs/%x_%j.out         # Output log
#SBATCH -e logs/%x_%j.err         # Error log

# ================= ENVIRONMENT SETUP =================
# Server-specific conda path from template
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate environment (Check if name 'torch_gpu' exists on server!)
conda activate torch_gpu

echo "Job started on $(hostname) at $(date)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L

export PYTHONUNBUFFERED=1 # Unbuffered output

# ================= DATA PATH REMINDER =================
# ERROR PREVENTION: 
# Please ensure 'sda_training.py' has correct server paths for .h5 files!

# ================= RUN SCRIPT =================
# Create logs directory if missing
mkdir -p logs
mkdir -p results_grid_search

python run_grid_search.py

echo "Job Finished at $(date)"
