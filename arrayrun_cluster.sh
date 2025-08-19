#!/bin/bash

#SBATCH --mem=50G
#SBATCH --gpus=1
#SBATCH --partition=gpu-h100-80g,gpu-a100-80g,gpu-v100-32g
#SBATCH --exclude=gpu[45,47,48]
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=12
#SBATCH --job-name=tr_mico                     # Change here
#SBATCH --error=./outfiles/tr_mico_%A/e.err    # Change here
#SBATCH --output=./outfiles/tr_mico_%A/o.out   # Change here
#SBATCH --array=0-6


LOGDIR=tr_mico-$SLURM_JOB_ID-log               # Change here
mkdir -p logs/$LOGDIR

module load mamba
source activate env/

BETAS=(
  0.01
  0.001
  0.0001
  0.00001
  0.000001
  0.0000001
  0.00000001
)


BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

# Change here
python3 -m dm_control_local.train \
  --base_dir=logs/$LOGDIR \
  --gin_files=dm_control_local/configs/mico.gin \
  --gin_bindings="deepmind_control_lib.create_deepmind_control_environment.domain_name='cheetah'" \
  --gin_bindings="deepmind_control_lib.create_deepmind_control_environment.task_name='run'" \
  --gin_bindings="SACAgent.seed=1" \
  --gin_bindings="MetricSACAgent.trust_beta=0.00001"