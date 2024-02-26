#!/bin/bash
#SBATCH --job-name=myjob
#SBATCH --output=job.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
export PYTHONUNBUFFERED=1
export STDOUT_LINE_BUFFERED=1
export HF_HOME=/local/scratch/username/.cache/
export TRANSFORMERS_CACHE=/local/scratch/username/.cache/huggingface/transformers/
export XDG_CACHE_HOME=/local/scratch/username/.cache/
/local/scratch/username/miniconda3/envs/myproject/bin/python myscript.py

