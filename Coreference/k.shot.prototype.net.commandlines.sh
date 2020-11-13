export SHOT=1 #1, 3, 5, 10
export BATCHSIZE=32
export EPOCHSIZE=5 #only need max 5 epochs
export LEARNINGRATE=1e-6
export MAXLEN=250


# CUDA_VISIBLE_DEVICES=7 python -u k.shot.prototype.net.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 42 \
#     --kshot $SHOT > log.Coref.PrototypeNet.$SHOT.shot.seed.42.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=7 python -u k.shot.prototype.net.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 16 \
#     --kshot $SHOT > log.Coref.PrototypeNet.$SHOT.shot.seed.16.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=6 python -u k.shot.prototype.net.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 32 \
#     --kshot $SHOT > log.Coref.PrototypeNet.$SHOT.shot.seed.32.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=5 python -u k.shot.prototype.net.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 64 \
#     --kshot $SHOT > log.Coref.PrototypeNet.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=7 python -u k.shot.prototype.net.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 256 \
    --kshot $SHOT > log.Coref.PrototypeNet.$SHOT.shot.seed.128.txt 2>&1 &
