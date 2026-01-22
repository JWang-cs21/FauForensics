#!/bin/bash

#############################################################
## general setting:
debug_phase='debug' ## choice: ['debug', 'run']

## database:     FakeAV or LAV-DF or LipSyncTimit or PolyGlot
Database="FakeAV"

## dataloader choice:   rf or mm
data_type='mm'

## model setting:       
net_type='FauForensics'
num_cls=2

## optimizer:
epochs=50
batch_size=32
opt_type='adamp'
learningrate=1e-4
is_adjustLR=1
LR_decay=0.001
is_multi_lr=0
m_LR_decay=0.001
is_frozen=1

## phase choice:
is_test=0           # [ 0: train       1: val         2: test ]
is_step2=1          # [ 0: pre-train   1: fine-tune    ]

## config for dataloader
dataset_cfg='cfg/dataset.yaml'
PATH_root="/home/ubuntu/data/DFD/AV_stream_224"               ## local
num_workers=8

## other setting:
PATH_save="${net_type}_${data_type}_${Database}_${num_cls}cls_"   
CHECKPOINTSPATH='checkpoints/first.pth'                                     ## first training
# CHECKPOINTSPATH="checkpoints/${PATH_save}/net_end.pth"                      ## training from saved weights
# CHECKPOINTSPATH="checkpoints/${PATH_save}/net_best_test_acc.pth"            ## training from saved weights

if [ "$Database" = "LAV-DF" ]; then
    is_random_clip=0
    iters_accumulate=2
else
    is_random_clip=1
    iters_accumulate=1
fi

train_file='train.csv'
val_file='test.csv'
test_file='test.csv'

## debug mode:
if [ "$debug_phase" = "debug" ]; then
    TQDM=1
elif [ "$debug_phase" = "run" ];then
    TQDM=0
else
    echo "##check debug mode"
fi
##
#############################################################

seed=2024              ## random seeds

CUDA_VISIBLE_DEVICES=0 python -u main.py --PATH_root $PATH_root --PATH_save $PATH_save --checkpointspath $CHECKPOINTSPATH --is_tqdm $TQDM \
                  --database $Database --data_type $data_type --train_file $train_file --test_file $test_file --val_file $val_file \
                  --net_type $net_type --opt_type $opt_type  --num_workers $num_workers --iters_accumulate $iters_accumulate --is_amp 1  \
                  --LR $learningrate --max_epochs $epochs --batch_size $batch_size  --is_test $is_test --is_step2 $is_step2 \
                  --m_LR_decay $m_LR_decay  --LR_decay $LR_decay --is_adjustLR $is_adjustLR --is_multi_lr $is_multi_lr \
                  --LR_policy 'poly' --dataset_cfg $dataset_cfg --is_random_clip $is_random_clip --num_cls $num_cls \
                  --is_checkdata 0 --is_label_dict 1  --is_mul_loss 1 --is_lr_scheduler 0 --is_get_feat 0 --is_eval_val 1   --seed $seed  \
                  

echo "END.."