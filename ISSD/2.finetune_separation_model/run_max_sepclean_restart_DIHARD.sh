#!/bin/bash
set -e  # Exit on error

export LD_LIBRARY_PATH=/home/sre/leisun8/miniconda3/lib/:/usr/lib64:$LD_LIBRARY_PATH


MODEL_NAME=$1

DIR_SUFFIX=$2


# If you haven't generated LibriMix start from stage 0

# Main storage directory. You'll need disk space to store LibriSpeech, WHAM and LibriMix this is about 500 Gb
storage_dir=


# After running the recipe a first time, you can run it from stage 3 directly to train new models.

# Path to the python you'll use for the experiment. Defaults to the current python
# You can run ./utils/prepare_python_env.sh to create a suitable python environment, paste the output here.
#python_path=${storage_dir}/asteroid_conda/miniconda3/bin/python
python_path=/home/intern/stniu/anaconda3/envs/asteroid/bin/python

# All the parameters
# General
stage=1  # Controls from which stage to start
# You can ask for several GPUs using id (passed to CUDA_VISIBLE_DEVICES)
id=0
out_dir=librimix_test # Controls the directory name associated to the evaluation results inside the experiment directory

# Network config
n_blocks=8
n_repeats=3
mask_act=relu
# Training config
epochs=1
batch_size=4
num_workers=6
half_lr=yes
early_stop=yes
# Optim config
optimizer=adam
lr=0.001
weight_decay=0.
 
train_dir=./training_data_withoutSNR_noNorm_webrtcvad/${MODEL_NAME}/train/
valid_dir=./training_data_withoutSNR_noNorm_webrtcvad/${MODEL_NAME}/dev/
test_dir=Libri2Mix_8k/max/test/
sample_rate=8000


n_src=2
segment=1
task=sep_clean
filter=filter

tag=ft_${MODEL_NAME} # Controls the directory name associated to the experiment


. utils/parse_options.sh

exp_dir=exp_${DIR_SUFFIX}_16.22_fuxian_test_${segment}_${filter}/train_${tag}_${task}
 
echo "Results from the following experiment will be stored in $exp_dir"

if [[ $stage -le 1 ]]; then
  echo "Stage 1: Training"
  mkdir -p logs
  CUDA_VISIBLE_DEVICES=$id $python_path train_recap_from_pretrained.py --exp_dir $exp_dir \
  --pretrained_model /work/sre/leisun8/tools/source_separation/asteroid/egs/librimix/ConvTasNet_8k/exp/train_2mix_max_sep_clean_restoreFromMod34_fromMod92_sep_clean/_ckpt_epoch_75.ckpt   \
  --n_blocks $n_blocks \
  --n_repeats $n_repeats \
  --mask_act $mask_act \
  --epochs $epochs \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --half_lr $half_lr \
  --early_stop $early_stop \
  --optimizer $optimizer \
  --lr $lr \
  --weight_decay $weight_decay \
  --train_dir $train_dir \
  --valid_dir $valid_dir \
  --sample_rate $sample_rate \
  --n_src $n_src \
  --segment $segment \
  --task $task
fi


# --pretrained_model  /work/sre/leisun8/tools/source_separation/asteroid/egs/librimix/ConvTasNet_8k/exp/train_2mix_max_sep_clean_restoreFromMod34_fromMod92_sep_clean/_ckpt_epoch_75.ckpt  
# --pretrained_model /yrfs1/intern/stniu/asteroid-master/egs/ConvTasNet/exp/train_convtasnet_librimix_8k_max_pretrain/_ckpt_epoch_29_-19.3.ckpt \
