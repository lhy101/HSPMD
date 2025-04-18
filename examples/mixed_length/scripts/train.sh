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
STRATEGY_POOL_PATH=${11:-"./strategy/strategy_pool_32b.json"}

ITER_PER_RANK=1
WARM_UP=0
COMPUTE_ONLY=0

# exp in paper
# example: 32 H20, 32B model, 32K context length, CommonCrawl
NUM_GPUS=32
# Strategy A and B
MULTI_TP_PP_LIST="[[(16, 1), (4, 1), (4, 1), (4, 1), (4, 1)], [(8, 1), (4, 2), (4, 2), (4, 2)]]"

echo num_gpus=${NUM_GPUS}, global_token_num = ${GLOBAL_TOKEN_NUM}, max_seq_len = ${MAX_SEQ_LEN}
LOG_FOLDER=logs/gpus${NUM_GPUS}_gtn${GLOBAL_TOKEN_NUM}_msl${MAX_SEQ_LEN}
mkdir -p ${LOG_FOLDER}
echo logs will save to ${LOG_FOLDER}...

ROOT_FOLDER=data
DATA_CACHE_PATH=${ROOT_FOLDER}/web
DATA_PATH=${ROOT_FOLDER}/web/web_content_document
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

if [ ! -d "ds_parallel_config" ]; then
  mkdir "ds_parallel_config"
fi

# compute-sanitizer can be added in front of python3 to check illegal mem access bug
CMD="python3 -u train.py \
--iter_per_rank $ITER_PER_RANK \
--warm_up $WARM_UP \
--compute_only $COMPUTE_ONLY \
--multi_tp_pp_list \"${MULTI_TP_PP_LIST}\" \
--global_batch_size $GLOBAL_BATCH_SIZE \
--global_token_num $GLOBAL_TOKEN_NUM \
--max_seq_len $MAX_SEQ_LEN \
--strategy_pool $STRATEGY_POOL_PATH \
--data_path $DATA_PATH \
--data_cache_path $DATA_CACHE_PATH \
--tokenizer_type "GPT2BPETokenizer" \
--split "98,1,1" \
--vocab_file $VOCAB_FILE \
--merge_file $MERGE_FILE \
--vocab_size 50304 \
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
