# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-512}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-8}
NUM_MICRO_BATCHES=${6:-2}

export NCCL_DEBUG=VERSION
export HSPMD_SWITCH_ALGORITHM=NEW_GREEDY
export HSPMD_SWITCH_PROFILE=TRACE
export HSPMD_INTERNAL_LOG_LEVEL=INFO
mpirun --allow-run-as-root -np 8 \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python3 lhy_hetero.py \
--num_strategy=2 \
--ds_parallel_config ds_parallel_config/dp2_tp2_pp2.json,ds_parallel_config/hetero/dp2_tp2_pp2.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--num_micro_batches $NUM_MICRO_BATCHES \
--dataset wikicorpus_en \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 20 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \