# NCCL_DEBUG=info
NUM_LAYERS=${1:-12}
HIDDEN_SIZE=${2:-768}
NUM_HEADS=${3:-12}
SEQ_LEN=${4:-1024}

export HSPMD_PARALLEL_CHANGE_TEST=PRECISION
export HSPMD_INTERNAL_LOG_LEVEL=INFO
mpirun --allow-run-as-root -np 8 python lhy_symbolic_train.py \
--num_micro_batches 2 \
--dp 2 \
--vocab_size 50257 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--lr 0.01 \
--dropout_prob 0.1