export SHOT=1 #1, 3, 5, 10
export BATCHSIZE=10
export TARGETBATCHSIZE=1  #1, 2, 3, 6
export EPOCHSIZE=1 #only need max 5 epochs
export LEARNINGRATE=1e-4
export DROPOUT=0.1
export UPDATEBERTLAYERS=5
export MAXLEN=250

# CUDA_VISIBLE_DEVICES=0 python -u k.shot.GFS.Entail.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --target_train_batch_size $TARGETBATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 42 \
#     --update_BERT_top_layers $UPDATEBERTLAYERS \
#     --kshot $SHOT > log.Coref.GFS.Entail.$SHOT.shot.seed.42.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=1 python -u k.shot.GFS.Entail.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --target_train_batch_size $TARGETBATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 16 \
#     --update_BERT_top_layers $UPDATEBERTLAYERS \
#     --kshot $SHOT > log.Coref.GFS.Entail.$SHOT.shot.seed.16.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=2 python -u k.shot.GFS.Entail.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --target_train_batch_size $TARGETBATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 32 \
#     --update_BERT_top_layers $UPDATEBERTLAYERS \
#     --kshot $SHOT > log.Coref.GFS.Entail.$SHOT.shot.seed.32.txt 2>&1 &
#
# CUDA_VISIBLE_DEVICES=3 python -u k.shot.GFS.Entail.py \
#     --do_lower_case \
#     --num_train_epochs $EPOCHSIZE \
#     --train_batch_size $BATCHSIZE \
#     --target_train_batch_size $TARGETBATCHSIZE \
#     --eval_batch_size 64 \
#     --learning_rate $LEARNINGRATE \
#     --max_seq_length $MAXLEN \
#     --seed 64 \
#     --update_BERT_top_layers $UPDATEBERTLAYERS \
#     --kshot $SHOT > log.Coref.GFS.Entail.$SHOT.shot.seed.64.txt 2>&1 &


CUDA_VISIBLE_DEVICES=4 python -u k.shot.GFS.Entail.py \
    --do_lower_case \
    --num_train_epochs $EPOCHSIZE \
    --train_batch_size $BATCHSIZE \
    --target_train_batch_size $TARGETBATCHSIZE \
    --eval_batch_size 64 \
    --learning_rate $LEARNINGRATE \
    --max_seq_length $MAXLEN \
    --seed 128 \
    --update_BERT_top_layers $UPDATEBERTLAYERS \
    --kshot $SHOT > log.Coref.GFS.Entail.$SHOT.shot.seed.128.txt 2>&1 &
