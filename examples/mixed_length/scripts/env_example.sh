conda activate hspmd
source ../../hspmd.exp

export HSPMD_SWITCH_ALGORITHM=NEW_GREEDY
export HSPMD_SWITCH_PROFILE=TIME
export HSPMD_INTERNAL_LOG_LEVEL=INFO
export HSPMD_STRAGGLER=ANALYSIS

export HSPMD_MEMORY_PROFILE=WARN
export HSPMD_MAX_SPLIT_SIZE_MB=10240
export HSPMD_MAX_INTERNAL_FRAGMENT_SIZE_MB=0
export HSPMD_PRE_ALLOCATE_SIZE_MB=20000

export HSPMD_PARALLEL_ATTN_SPLIT_PATTERN=NORMAL

export NCCL_DEBUG=VERSION

echo "env done"