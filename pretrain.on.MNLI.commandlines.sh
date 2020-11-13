# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause


export BATCHSIZE=16 #32 is also ok
export EPOCHSIZE=20
export LEARNINGRATE=1e-6



CUDA_VISIBLE_DEVICES=0 python -u pretrain.on.MNLI.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 > log.mnli.pretrain.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u pretrain.on.MNLI.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 > log.mnli.pretrain.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u pretrain.on.MNLI.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 > log.mnli.pretrain.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u pretrain.on.MNLI.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 64 > log.mnli.pretrain.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u pretrain.on.MNLI.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 > log.mnli.pretrain.seed.128.txt 2>&1 &
