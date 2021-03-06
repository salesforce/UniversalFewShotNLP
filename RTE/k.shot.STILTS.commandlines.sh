# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

export SHOT=100000 #1, 3, 5, 10, 100000
export BATCHSIZE=5 #2, 3, 5, 2, 5
export EPOCHSIZE=10 #only need max 10 epochs
export LEARNINGRATE=1e-6



CUDA_VISIBLE_DEVICES=0 python -u k.shot.STILTS.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 \
    --kshot $SHOT > log.RTE.STILTS.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u k.shot.STILTS.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 \
    --kshot $SHOT > log.RTE.STILTS.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u k.shot.STILTS.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 \
    --kshot $SHOT > log.RTE.STILTS.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u k.shot.STILTS.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 64 \
    --kshot $SHOT > log.RTE.STILTS.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u k.shot.STILTS.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 \
    --kshot $SHOT > log.RTE.STILTS.$SHOT.shot.seed.128.txt 2>&1 &
