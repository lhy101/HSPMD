# NCCL_DEBUG=info
NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-256}
GLOBAL_BATCH_SIZE=${5:-16}
MICRO_BATCH_SIZE=${6:-2}

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

PATH="/home/pkuhspmd/envs/miniconda3/envs/hspmd-py/bin:${PATH}"
HSPMD_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../../../" && pwd )"
LD_LIBRARY_PATH="${HSPMD_HOME}/build/lib:${LD_LIBRARY_PATH}"
PYTHONPATH="${HSPMD_HOME}/python:${HSPMD_HOME}/build/lib:${PYTHONPATH}"

export NCCL_DEBUG=VERSION
export HSPMD_INTERNAL_LOG_LEVEL=WARN
mpirun --allow-run-as-root -np 16 \
-H job-4e4cb411-1139-4f15-b221-5a30f1760a2b-master-0:8,job-4e4cb411-1139-4f15-b221-5a30f1760a2b-worker-0:8 \
-x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python train_hspmd_gpt_ds_parallel.py \
--ds_parallel_config ds_parallel_config/two_node/dp4_tp2_pp2.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--seq_length $SEQ_LEN \
--epochs 20 \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--use_multi_node \
