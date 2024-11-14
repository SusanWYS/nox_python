#!/bin/bash
#SBATCH -t 20:00:59
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH -n 15
#SBATCH --gres=gpu:1
#SBATCH --mem=600G
#SBATCH --exclude=node093,node101

source /weka/scratch/weka/quest/susanw26/anaconda/etc/profile.d/conda.sh
conda activate nox
echo "Run Started"


python3 model.py 

conda deactivate
echo "Run Ended"
