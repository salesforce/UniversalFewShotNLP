export SHOT=10000 #1, 3, 5, 10, 100000
export BATCHSIZE=32 #2, 3, 5, 2, 5
export TARGETBATCHSIZE=32
export EPOCHSIZE=1 #only need max 5 epochs
export LEARNINGRATE=1e-7
export DROPOUT=0.1


CUDA_VISIBLE_DEVICES=0 python -u k.shot.GFS.Entail.v2.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 \
    --kshot $SHOT > log.SciTail.GFS.Entail.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u k.shot.GFS.Entail.v2.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 \
    --kshot $SHOT > log.SciTail.GFS.Entail.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u k.shot.GFS.Entail.v2.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 \
    --kshot $SHOT > log.SciTail.GFS.Entail.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u k.shot.GFS.Entail.v2.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 64 \
    --kshot $SHOT > log.SciTail.GFS.Entail.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u k.shot.GFS.Entail.v2.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 \
    --kshot $SHOT > log.SciTail.GFS.Entail.$SHOT.shot.seed.128.txt 2>&1 &
