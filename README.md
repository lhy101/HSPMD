# HSPMD

This repository contains the implementation of HSPMD, a novel framework for efficient deep learning that addresses spatial and temporal workload heterogeneity. It includes our system prototype and scripts to reproduce all results from our paper entitled *HSPMD: Hierarchical and Heterogeneous Single Program Multiple Data for General and Scalable Deep Learning Training*.

## Key Features

- Hierarchical and heterogeneous sharding annotations and communication resolution.
- Progressive graph specialization and dynamic graph switching techniques.
- Optimized solutions for three scenarios: 1. heterogeneous clusters, 2. elastic scenarios, and 3. mixed-length data.

## 1. Build & Compile Our System

To facilitate training using HSPMD sharding annotations and hierarchical communication resolution, we developed a prototype graph-based DL framework. It supports the functionality of progressive graph specialization and dynamic graph switching. We use `cmake >= 3.24` to compile it. Essential third-party packages such as `flash-attn`, `onednn`, and `cutlass` have been prepared and will be compiled automatically.

For GPU communication, we utilize `NCCL` as the backend library, included as a submodule. To download the latest version of `NCCL`, execute `git submodule update --init --recursive`. If you prefer to use a different version of `NCCL`, you may set the environment variable `NCCL_ROOT` to your desired path: `export NCCL_ROOT=/your/path/`. Furthermore, we utilize `grpc` for efficient CPU communication and elastic training. Please install it in advance and use `export GRPC_CMAKE_PREFIX_PATH=/your/path` to let HSPMD locate your `grpc` installation.

The compilation commands are as follows:

```bash
mkdir -p build && cd build
cmake ..
make -j 32
cd ..
source hspmd.exp
```

For specific scenarios (e.g., mixed-length data), cost models are required to determine the optimal training strategy. These models rely on `PuLP` for solving ILP problems. Install it via:

```bash
pip install pulp
```

## 2. HSPMD Annotations

Users can manually define all annotations from scratch to build a model. For convenience and efficiency, we also offer a quick annotation generation solution. For instance, with a Llama-2 32B model configuration, we can describe a strategy as follows:

```bash
# 24*H20+15*H800
NUM_GPUS=44
DP=3
CP_LIST="[1,1,1]"
TP=4 # Max TP Degree
LAYERS_NUM_LIST="[[7,7,23,23],[11,11,38],[12,11,24,13]]"
MICRO_BATCH_NUM_LIST="[29,18,17]"
UNUSED_RANK="[38,39,41,42,43]"
RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:24,9:25,10:26,11:27,12:28,13:29,14:30,15:31,16:8,17:9,18:10,19:11,20:12,21:13,22:14,23:15,24:32,25:33,26:34,27:35,28:16,29:17,30:18,31:19,32:20,33:21,34:22,35:23,36:36,37:37,38:39,39:40,40:38,41:41,42:42,43:43}"
SEQ_LEN_LIST="[4096, 4096, 4096]"
```

By running `python -m hspmd.models.llama.generate_llama_hetero_4d_config`, this packed strategy expression will be converted into actual HSPMD sharding annotations, generating a JSON file containing all weight annotations for the llama model. For instance, in the above scenario, the parameter `llama.blocks.block59.attn.dense` will have:


```json
{
    "split": {"0": [4, 1]},
    "dup": [2, 2],
    "device_group_union": [[28, 29, 30, 31], [38]],
    "type": "variable"
}
```

Here, the `split` and `dup` lists describe HSPMD's *Distributed States Union (DS Union)*, while `device_group_union` represents HSPMD's *Device Group Union (DG Union)*. In this case, both lists contain two elements, indicating an *HSize* of 2. When ZeRO is disabled, the `type` being `variable` indicates an *HDim* of −1, meaning no further parameter partitioning is required (if ZeRO is enabled, *HDim* should be set to 0).

This JSON file encapsulates all annotations for the current parallel strategy and will be passed to `train.py`.

## 3. Communication Operators

As described in our paper, we explicitly declare *CommOps* in the *User-defined Graph* to enable transformations between annotations. For example, in `python/hspmd/models/llama/llama_model.py`, we provide a llama model definition that includes *CommOp* declarations, such as:

```python
...
if next_block.rmsnorm_1.sequence_parallel:
    hidden_states = hspmd.comm(hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
else:
    hidden_states = hspmd.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
...
```

These `hspmd.comm` calls define *CommOps*, enabling transformations between arbitrary HSPMD annotations. Eventually, these *CommOps* will be replaced with concrete communication operators (e.g., send-receive) for actual execution.

## 4. Graph Specialization and Switching

The core logic for these features is efficiently implemented in the C++ backend. The specialization logic can be found in `hspmd/graph/define_and_run_graph.cc(h)` and `hspmd/graph/executable_graph.cc(h)`, while the switching logic is fully implemented in `hspmd/graph/switch_exec_graph.cc(h)`. Both implementations strictly follow the steps described in the paper.

## 5. Three Heterogeneous Scenarios

We provide one-click scripts in the `examples` folder to reproduce all experiments from our paper:

```bash
cd examples
cd hetero_cluster && bash scripts/train.sh
cd elastic_training && bash scripts/train.sh
cd mixed_length && bash scripts/train.sh
```

These scripts will train llama models using HSPMD in three scenarios (heterogeneous clusters, elastic training, and mixed-length data) respectively. Before running `train.sh`, please prepare the training dataset and modify the dataset path in `train.sh`. In the paper, we used the `CommonCrawl` (available at [Hugging Face](https://huggingface.co/datasets/tiiuae/falcon-refinedweb)) and `GitHub` (available at [Hugging Face](https://huggingface.co/datasets/codeparrot/github-code)) datasets. You may also use any preferred dataset. 

For the `CommonCrawl` dataset, you can download it via the provided link or use our script:

```bash
cd hetero_cluster/data_utils
bash data/create_web_dataset.sh
```

Below we provide detailed explanations for each scenario.

### 5.1. Heterogeneous Clusters

In `examples/hetero_cluster/train.sh`, we provide all parallel strategies used for training on heterogeneous clusters. You can run on specific clusters by setting the `CASE` number in the bash script. We also support homogeneous strategies (4D Parallel). Users can define custom parallel strategies in the following format:

```bash
if [[ ${CASE} -eq 0 ]]; then
# test
    HETERO=false
    NUM_GPUS=16
    TP=2
    PP=2
    DP=2
    CP=2
elif [[ ${CASE} -eq 1 ]]; then
# 32*H20+16*H800
    HETERO=true
    NUM_GPUS=48
    DP=4
    CP_LIST="[1,1,1,1]"
    TP=4
    LAYERS_NUM_LIST="[[11,11,38],[11,11,38],[11,11,38],[11,11,38]]"
    MICRO_BATCH_NUM_LIST="[16,16,16,16]"
    UNUSED_RANK="[]"
    RANK_TO_DEVICE_MAPPING="{0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:32,9:33,10:34,11:35,12:8,13:9,14:10,15:11,16:12,17:13,18:14,19:15,20:36,21:37,22:38,23:39,24:16,25:17,26:18,27:19,28:20,29:21,30:22,31:23,32:40,33:41,34:42,35:43,36:24,37:25,38:26,39:27,40:28,41:29,42:30,43:31,44:44,45:45,46:46,47:47}"
    SEQ_LEN_LIST="null"
```

### 5.2. Elastic Training

We provide the elastic training traces C1-C3 and C4-C7 from our paper in `examples/elastic_training/homo_cluster_trace` and `examples/elastic_training/hetero_cluster_trace`, which include all specific strategies employed by HSPMD.

In the script `examples/elastic_training/train.sh`, we demonstrate a training example that sequentially applies different strategies to the C1-C2 trace, with transitions between these strategies seamlessly handled by the *Dynamic Graph Switching* technique. This script can directly reproduce all results presented in the *Case Study* section of our paper.

To experiment with different traces, users can simply modify the trace path and strategy path in `train.sh`:


```bash
python generate_strategy_ds.py -p "./home_cluster_trace" -m "llama"
```

### 5.3. Mixed-length Data

In `examples/mixed_length/train.sh`, we provide examples to reproduce *Figure 16* (32K context length, CommonCrawl) from our paper, including the specific configurations for *Strategy A* and *Strategy B*.

Note that this scenario involves determining which sequences should be processed by which ranks with specific parallelism, requiring pre-profiling results and runtime cost models for decision-making.

We have already provided pre-generated profiling results in `examples/mixed_length/strategy/strategy_pool_32b.json`.  Users can modify this according to their cluster specifications and update the path in `train.sh`:

```bash
STRATEGY_POOL_PATH=${11:-"./strategy/strategy_pool_32b.json"}
```

The cost model and decision-making code reside in `examples/mixed_length/strategy`. We use multiple ILP algorithms for variable-length sequence dispatching and packing. Note this component is decoupled from HSPMD's design, enabling users to define custom load-balancing strategies for variable-length sequences.