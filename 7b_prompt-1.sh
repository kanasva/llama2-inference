#!/bin/bash

### Change here ###
#SBATCH --job-name=7b_prompt-1
#SBATCH --output=7b_prompt-1.txt

#SBATCH --partition=milan-gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --tmp=20GB
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL,TIME_LIMIT_90 
#SBATCH --mail-user=<your-email>

cp /fred/oz176/kan_asvasena/llama2_2.sif $JOBFS/llama2_2.sif

module load apptainer

### Change here ###
apptainer run -B /fred/oz176/kan_asvasena/llama2_storage --nv $JOBFS/llama2_2.sif python llama2.py \
    "prompt-1.csv" \
    "7b" \
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."