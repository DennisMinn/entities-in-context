#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --mem=8192
#SBATCH --cpus-per-task=8 -N 1
#SBATCH -p gypsum-m40
#SBATCH -o slurm-%j.out
#SBATCH --time=24:00:00

$1
