#!/bin/bash
#SBATCH -t 29:00:59
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --gres=gpu:1
#SBATCH -n 4
#SBATCH --mem=600G
#SBATCH --exclude=node093,node101

source /weka/scratch/weka/quest/susanw26/anaconda/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/weka/scratch/weka/quest/susanw26/anaconda/lib
conda activate nox
echo "Run Started"


python3 simulation.py 

conda deactivate
echo "Run Ended"
