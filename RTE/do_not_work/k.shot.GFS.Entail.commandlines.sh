export SHOT=10 #1, 3, 5, 10, 100000
export BATCHSIZE=32 #2, 3, 5, 2, 5
export TARGETBATCHSIZE=2
export EPOCHSIZE=1 #only need max 5 epochs
export LEARNINGRATE=1e-6
export DROPOUT=0.1
export PRETAINEPOCHS=3


CUDA_VISIBLE_DEVICES=0 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 42 \
    --pretrain_epochs $PRETAINEPOCHS \
    --kshot $SHOT > log.RTE.GFS.Entail.source.pretrain.epoch.$PRETAINEPOCHS.drop.$DROPOUT.$TARGETBATCHSIZE.targetBatch.$SHOT.shot.seed.42.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 16 \
    --pretrain_epochs $PRETAINEPOCHS \
    --kshot $SHOT > log.RTE.GFS.Entail.source.pretrain.epoch.$PRETAINEPOCHS.drop.$DROPOUT.$TARGETBATCHSIZE.targetBatch.$SHOT.shot.seed.16.txt 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 32 \
    --pretrain_epochs $PRETAINEPOCHS \
    --kshot $SHOT > log.RTE.GFS.Entail.source.pretrain.epoch.$PRETAINEPOCHS.drop.$DROPOUT.$TARGETBATCHSIZE.targetBatch.$SHOT.shot.seed.32.txt 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 64 \
    --pretrain_epochs $PRETAINEPOCHS \
    --kshot $SHOT > log.RTE.GFS.Entail.source.pretrain.epoch.$PRETAINEPOCHS.drop.$DROPOUT.$TARGETBATCHSIZE.targetBatch.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length 128 \
    --seed 128 \
    --pretrain_epochs $PRETAINEPOCHS \
    --kshot $SHOT > log.RTE.GFS.Entail.source.pretrain.epoch.$PRETAINEPOCHS.drop.$DROPOUT.$TARGETBATCHSIZE.targetBatch.$SHOT.shot.seed.128.txt 2>&1 &
