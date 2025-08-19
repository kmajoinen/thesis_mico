#!/bin/bash


RUN_DIR=$1

#export WANDB_NAME="$RUN_DIR" #_"cheetah_run"  #"$RUN_DIR-cheetah_run"

module load mamba
source activate ../tr_deep_bisim4control/env/


#echo $RUN_DIR
#echo ${WANDB_NAME}

wandb sync --project "thesis_baselines" logs/$RUN_DIR/custom/

rm -rf logs/$RUN_DIR