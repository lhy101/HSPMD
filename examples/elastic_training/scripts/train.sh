NUM_LAYERS=${1:-60}
HIDDEN_SIZE=${2:-6656}
FFN_HIDDEN_SIZE=${3:-17920}
NUM_HEADS=${4:-64}
GLOBAL_TOKEN_NUM=${5:-10000}
MAX_SEQ_LEN=${6:-32769}
SERVER_ADDR=${7:-"${IP_1}"} # master-0
SERVER_PORT=${8:-"23333"}
HOST_FILE_PATH=${9:-"${ENV_PATH}/host_32GPUs.yaml"}
ENV_FILE_PATH=${10:-"${ENV_PATH}/env_H20.sh"}

# exp in paper
# example: trace from C1-to-C2
NUM_GPUS=40
python generate_strategy_ds.py -p "./home_cluster_trace" -m "llama"

LOG_FOLDER=logs/switch
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=/jizhicfs/pinxuezhao/lhy/data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

CMD="python3 -u train.py \
--num_strategy=2 \
--ds_parallel_config ds_parallel_config/homo_multi_strategy/strategy_1.json,ds_parallel_config/homo_multi_strategy/strategy_2.json \
--strategy_config homo_cluster_trace/strategy_1.json,homo_cluster_trace/strategy_2.json \
--global_batch_size $GLOBAL_BATCH_SIZE \
--micro_batch_size $MICRO_BATCH_SIZE \
--global_seq_len $SEQ_LEN \
--json_file $JSON_FILE \
--json_key $JSON_KEY \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 30592 \
--hidden_size $HIDDEN_SIZE \
--ffn_hidden_size $FFN_HIDDEN_SIZE \
--num_hidden_layers $NUM_LAYERS \
--num_attention_heads $NUM_HEADS \
--epochs 4 \
--steps 40 \
--lr 1e-4 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
--server_addr ${SERVER_ADDR} \
--server_port ${SERVER_PORT} \
--ngpus ${NUM_GPUS}"

source ${ENV_FILE_PATH}
python3 -m hspmd.rpc.pssh_start \
	--hosts ${HOST_FILE_PATH} \
	--command "$CMD" \
	--server_port ${SERVER_PORT} \
	--ngpus ${NUM_GPUS} \
	--envs ${ENV_FILE_PATH} \
	--log_path ${LOG_FOLDER}
