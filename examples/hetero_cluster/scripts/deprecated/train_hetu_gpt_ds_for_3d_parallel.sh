NUM_LAYERS=${1:-32}
HIDDEN_SIZE=${2:-4096}
NUM_HEADS=${3:-32}
SEQ_LEN=${4:-128}
GLOBAL_BATCH_SIZE=${5:-8}
MICRO_BATCH_SIZE=${6:-2}
DP=${7:-2}
TP=${8:-2}
PP=${9:-2}
NUM_GPUS=$(( $DP * $TP *$PP ))

echo dp=${DP}, tp=${TP}, pp=${PP}, num_gpus=${NUM_GPUS} 

if [[ ${NUM_LAYERS} -eq 32 && ${HIDDEN_SIZE} -eq 4096 && ${NUM_HEADS} -eq 32 ]]; then
	MODEL_SIZE=7b
	echo use gpt 7b model...
elif [[ ${NUM_LAYERS} -eq 40 && ${HIDDEN_SIZE} -eq 5120 && ${NUM_HEADS} -eq 40 ]]; then
	MODEL_SIZE=13b
	echo use gpt 13b model...
else
	echo the model should be 7b or 13b for test.
	exit 0
fi

if [ ${SEQ_LEN} -lt 1024 ]; then
	SEQ=$SEQ_LEN
else
	SEQ=$(( ${SEQ_LEN} / 1024 ))k
fi
echo use seq_len = ${SEQ}

DS_PARALLEL_CONFIG=ds_parallel_config/gpus${NUM_GPUS}/${MODEL_SIZE}/dp${DP}_tp${TP}_pp${PP}.json
if [ ! -f ${DS_PARALLEL_CONFIG} ]; then
	python3 ds_parallel_config/generate_gpt_3d_config.py --model_size ${MODEL_SIZE} --dp ${DP} --tp ${TP} --pp ${PP} --zero
	echo generate ${DS_PARALLEL_CONFIG}...
else
	echo use ${DS_PARALLEL_CONFIG}...
fi

LOG_FOLDER=logs/gpus${NUM_GPUS}_${MODEL_SIZE}_${SEQ}
mkdir -p ${LOG_FOLDER}
LOG_FILE=${LOG_FOLDER}/gbs${GLOBAL_BATCH_SIZE}_mbs${MICRO_BATCH_SIZE}_dp${DP}_tp${TP}_pp${PP}.log
echo log will save to ${LOG_FILE}...

ROOT_FOLDER=data
JSON_FILE=${ROOT_FOLDER}/web/refinedweb0.json
JSON_KEY=content
VOCAB_FILE=${ROOT_FOLDER}/vocab.json
MERGE_FILE=${ROOT_FOLDER}/merges.txt

export LD_LIBRARY_PATH=/jizhicfs/hymiezhao/hspmd-gh/HSPMD-dev/build/hspmd/third_party/cutlass/install:$LD_LIBRARY_PATH
source /jizhicfs/hymiezhao/hspmd-gh/HSPMD-dev/init2.sh
source /jizhicfs/hymiezhao/hspmd-gh/HSPMD-dev/hspmd.exp

# ds_parallel_config can use ds_parallel_config/generate_gpt_3d_config.py for auto-generate
export NCCL_DEBUG=VERSION
export HSPMD_INTERNAL_LOG_LEVEL=WARN
mpirun --allow-run-as-root -np 8 \
--output-filename logs/ds_parallel --merge-stderr-to-stdout \
python3 train_hspmd_gpt_ds_parallel.py \
--ds_parallel_config $DS_PARALLEL_CONFIG \
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
--epochs 1 \
--steps 1000 \
--lr 1e-6 \
--adam_weight_decay 0.01 \
--hidden_act relu \
--dropout_prob 0.1 \
--bf16 \
--use_flash_attn \
2>&1 | tee ${LOG_FILE}
