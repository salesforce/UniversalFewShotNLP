export SHOT=10 #1, 3, 5, 10, 0
export BATCHSIZE=5 #2, 4, 4, 5, 8
export EPOCHSIZE=20
export LEARNINGRATE=1e-6
export MAXLEN=250


CUDA_VISIBLE_DEVICES=0 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 42 \
    --kshot $SHOT > log.QA.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 16 \
    --kshot $SHOT > log.QA.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 32 \
    --kshot $SHOT > log.QA.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 64 \
    --kshot $SHOT > log.QA.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u full.shot.train.on.target.data.py \
    --task_name rte \
    --do_train \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 128 \
    --kshot $SHOT > log.QA.$SHOT.shot.seed.128.txt 2>&1 &
