#!/bin/bash
# ==============================================================
# SLURM JOB SCRIPT TEMPLATE  (single GPU example)
# --------------------------------------------------------------
# [Server Information]
#   NodeName    : server1
#   Total CPUs  : 64 cores
#   Total Memory: 62000 MB  (~62 GB usable)
#   Total GPUs  : 4 × NVIDIA RTX A5000
# --------------------------------------------------------------
# [Resource limits reminder]
#   - You must request resources within the server limits:
#       --cpus-per-task <= 64
#       --mem <= 62000M (or 62G)
#       --gres=gpu: <= 4
#   - Exceeding these values will cause "Requested node configuration is not available" error.
# ==============================================================
#SBATCH -J SDA_GridSearch
#SBATCH -N 1                              # Number of nodes (always 1 in this server)
#SBATCH -n 1                              # Number of tasks (processes)
#SBATCH --gres=gpu:4                      # GPU allocation
#SBATCH --cpus-per-task=32                # CPU allocation: choose ≤ 64
#SBATCH --mem=58G                         # Memory allocation: choose ≤ 62000 MB (~62G)
#SBATCH -t 48:00:00                       # Max runtime (HH:MM:SS)
#SBATCH -o logs/%x_%j.out                 # Log files   (%x = job name, %j = job ID)
#SBATCH -e logs/%x_%j.err                 # Error files (%x = job name, %j = job ID)

# Initialize Conda
source /opt/miniconda3/etc/profile.d/conda.sh

# Activate the torch_gpu environment
conda activate torch_gpu

# Print environment info for debugging
echo "Python: $(which python)"
echo "Conda Env: $CONDA_DEFAULT_ENV"
nvidia-smi

# Run the grid search script
python run_grid_search_server.py
