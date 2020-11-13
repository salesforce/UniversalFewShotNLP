export SHOT=3 #1, 3, 5, 10, 0
export BATCHSIZE=2 #2, 2, 4, 4, 8
export EPOCHSIZE=10
export LEARNINGRATE=1e-6



CUDA_VISIBLE_DEVICES=0 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 250 \
    --seed 42 \
    --kshot $SHOT > log.Coref.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 250 \
    --seed 16 \
    --kshot $SHOT > log.Coref.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 250 \
    --seed 32 \
    --kshot $SHOT > log.Coref.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 250 \
    --seed 64 \
    --kshot $SHOT > log.Coref.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 32 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 250 \
    --seed 128 \
    --kshot $SHOT > log.Coref.$SHOT.shot.seed.128.txt 2>&1 &
