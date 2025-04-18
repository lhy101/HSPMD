import os
import hspmd as ht
from hspmd_gpt_ds_parallel import GPTLMHeadModel
from hspmd.nn.modules.parallel_ds import config2ds
from gpt_config import GPTConfig
from data_utils import GPTJsonDataset, get_mask_and_position_ids, build_pretraining_data_loader
import numpy as np
import time
import argparse
import json
import socket
from queue import Queue
from collections import deque
from hspmd.utils.checkpoint import load_checkpoint, save_checkpoint, load_checkpoint_from_megatron

local_device = None
all_devices = None

def distributed_init(use_multi_node: bool = False):
    if use_multi_node:
        hostname = socket.gethostname()
        os.environ['HSPMD_LOCAL_HOSTNAME'] = hostname
        #if hostname == 'job-e44df83d-4af0-4fbf-b066-b4650867451d-master-0':
        #    os.environ['HSPMD_LOCAL_HOSTNAME'] = 'a100-0'
        #elif hostname == 'job-e44df83d-4af0-4fbf-b066-b4650867451d-worker-0':
        #    os.environ['HSPMD_LOCAL_HOSTNAME'] = 'a100-1'
        #else:
        #    raise ValueError(f"Unknown hostname: {hostname}")

    global local_device, all_devices
    ht.init_comm_group(8)
    local_device = ht.local_device()
    all_devices = ht.global_device_group()
    if local_device.index == 0:
        print(f'local_device: {local_device}, all_devices: {all_devices}')

def read_ds_parallel_config(args):
    # read ds_parallel_config from json file
    ds_parallel_config = json.load(open(args.ds_parallel_config, 'r'))
    # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp2_pp2.json', 'r'))
    # ds_parallel_config = json.load(open('./ds_parallel_config/dp2_tp4.json', 'r'))
    print(f'{local_device}: load ds_parallel_config from: {args.ds_parallel_config}')
    zero = ds_parallel_config['zero']
    # assign zero to all variables
    config_queue = Queue()
    for value in ds_parallel_config.values():
        config_queue.put(value)
    while (not config_queue.empty()):
        config = config_queue.get()
        if type(config) == dict:
            if 'type' in config:
                if config['type'] == 'variable' and 'zero' not in config:
                    config['zero'] = zero
            else:
                for value in config.values():
                    config_queue.put(value)
    # print(f'{local_device}: ds_parallel_config: {ds_parallel_config}')
    return ds_parallel_config

def train_dataset_provider(args):
    """Build train dataset."""
    train_dataset = GPTJsonDataset(
        json_file=args.json_file,
        key=args.json_key,
        max_seq_len=args.seq_length,
        vocab_file=args.vocab_file,
        merge_file=args.merge_file)
    return train_dataset

def train_data_iterator(dataset, consumed_samples, mbs, dp_rank, dp_size):
    # print(f'new dataloader: consumed_samples = {consumed_samples}')
    train_dataloader = build_pretraining_data_loader(dataset, consumed_samples, mbs, dp_rank, dp_size)
    train_data_iterator = iter(train_dataloader)
    return train_data_iterator

def pretrain(args):
    ds_parallel_config = read_ds_parallel_config(args)

    config = GPTConfig(vocab_size=args.vocab_size, 
                       n_positions=args.seq_length,
                       n_ctx=args.seq_length,
                       n_embd=args.hidden_size,
                       n_layer=args.num_hidden_layers, 
                       n_head=args.num_attention_heads, 
                       seq_len=args.seq_length,
                       # n_inner=4*args.hidden_size,
                       resid_pdrop=args.dropout_prob,
                       embd_pdrop=args.dropout_prob,
                       attn_pdrop=args.dropout_prob,
                       activation_function=args.hidden_act,
                       global_batch_size=args.global_batch_size,
                       use_flash_attn=args.use_flash_attn,
                       )

    # simple check for gpt blocks range
    ranges = []
    for _, block_config in ds_parallel_config['gpt']['blocks'].items():
        ranges.append(block_config['range'])
    assert ranges[0][0] == 0 and ranges[-1][1] == config.num_hidden_layers-1, \
        f"gpt blocks range: {ranges} is conflict with num_hidden_layers: {config.num_hidden_layers}!"

    # Hetu model definition
    model = GPTLMHeadModel(config=config, ds_parallel_config=ds_parallel_config)

    
    
    input_ds, input_device_group = config2ds(ds_parallel_config['input'])
    label_ds, label_device_group = config2ds(ds_parallel_config['label'])
    # print(f'input_ds: {input_ds}, label_ds: {label_ds}')
    
    global_batch_size = args.global_batch_size
    micro_batch_size = args.micro_batch_size
    seq_len = args.seq_length

    dp_size = input_ds.get_dim(0)
    mbs_times_dp = micro_batch_size * dp_size    

    input_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='input_ids')
    # token_type_ids = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='token_type_ids')
    attention_mask = ht.parallel_placeholder(ht.float32, global_shape=[mbs_times_dp, config.seq_len], ds=input_ds, device_group=input_device_group, name='attention_mask')
    masked_lm_labels = ht.parallel_placeholder(ht.int64, global_shape=[mbs_times_dp, config.seq_len], ds=label_ds, device_group=label_device_group, name='masked_lm_labels')

    print(f'{local_device}: build model begin...')
    loss = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            # token_type_ids=token_type_ids,
                            labels=masked_lm_labels)
    print(f'{local_device}: build model end...')

    loss_mean = loss

    print(f'{local_device}: optimizer minimize begin...')
    # opt = ht.SGDOptimizer(lr=args.lr, momentum = 0.0)
    opt = ht.AdamOptimizer(lr=args.lr)
    train_op = opt.minimize(loss_mean)
    print(f'{local_device}: optimizer minimize end...')
    load_checkpoint_from_megatron(model, opt, "./model_optim_rng.pt", config, local_device)
    print(f'{local_device}: build dataset begin...')
    train_dataset = train_dataset_provider(args)
    print(f'{local_device}: build dataset end...')

    # return
    # device in same dp_group will read the same batch data, idx=-1 means no need to read data
    dup_group_idx, dup_group_num = -1, -1
    if input_device_group.contains(local_device):
        local_device_idx = input_device_group.get_index(local_device)
        dup_group_idx = input_ds.get_dup_group_index(local_device_idx)
        dup_group_num = input_ds.get_dim(0)
    elif label_device_group.contains(local_device):
        local_device_idx = label_device_group.get_index(local_device)
        dup_group_idx = label_ds.get_dup_group_index(local_device_idx)
        dup_group_num = label_ds.get_dim(0)
    else:
        dup_group_num = input_ds.get_dim(0)

    dp_rank = dup_group_idx
    dp_size = dup_group_num
    gbs_per_dp = global_batch_size // dp_size
    mbs_times_dp = micro_batch_size * dp_size
    assert global_batch_size % mbs_times_dp == 0, \
        f'gbs {global_batch_size} must could be divided by mbs {micro_batch_size} * dp {dp_size}'
    num_micro_batches = global_batch_size // mbs_times_dp
    print(f'{local_device}: dp_rank={dp_rank}, dp_size={dp_size}, gbs={global_batch_size}, mbs={micro_batch_size}, num_micro_batches={num_micro_batches}')

    consumed_samples = 0
    if dp_rank != -1:
        train_iter = train_data_iterator(train_dataset, consumed_samples, micro_batch_size, dp_rank, dp_size) # need cache?
    else:
        train_iter = None

    loss_last10 = deque()
    time_last10 = deque()
    profiler_last10 = deque()
    for ep in range(args.epochs):
        for step in range(args.steps):
            # load data for each dp
            if train_iter:
                micro_batches = []
                for _ in range(num_micro_batches):
                    micro_batch = next(train_iter)
                    micro_batches.append(micro_batch)
                micro_batches = np.concatenate(micro_batches, axis=0) # [num_micro_batches, micro_batch_size, max_seq_len + 1]
                # padding sequence
                micro_batches = micro_batches.reshape(gbs_per_dp, -1) # [gbs_per_dp, seq_len + 1]
                labels = micro_batches[:, 1:] # [gbs_per_dp, seq_len]
                tokens = micro_batches[:, :-1] # [gbs_per_dp, seq_len]
                _attention_mask, _position_ids = get_mask_and_position_ids(tokens, train_dataset.encoder.pad_id())
                # _token_type_ids = np.zeros([gbs_per_dp, seq_len])

                feed_dict = {
                    input_ids: tokens.astype(np.int64),
                    # token_type_ids: _token_type_ids.astype(np.int64),
                    attention_mask: _attention_mask.astype(np.int64),
                    masked_lm_labels: labels.astype(np.int64),
                }
            else: # fake data; feed_dict={} will cause segment fault?
                feed_dict = {
                    input_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    # token_type_ids: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                    attention_mask: np.zeros([gbs_per_dp, seq_len]).astype(np.float32),
                    masked_lm_labels: np.zeros([gbs_per_dp, seq_len]).astype(np.int64),
                }
            # run exec graph
            start_time = time.time()
            with ht.profiler(enabled = args.profiler) as profiler:
                results = train_op.graph.run(loss_mean, 
                                            [loss_mean, train_op], 
                                            feed_dict = feed_dict, 
                                            num_micro_batches = num_micro_batches)
                if label_device_group.contains(local_device):
                    if(len(profiler_last10) >= 10):
                        profiler_last10.popleft()
                    profiler_last10.append(profiler.summary())
            end_time = time.time()
            consumed_samples += global_batch_size
            if label_device_group.contains(local_device):
                loss_out = results[0].numpy(force=True).mean()
                print('%s: [Epoch %d] (Iteration %d, consumed_samples = %d): Loss = %.3f, Time = %.4f'%(local_device, ep, step, consumed_samples, loss_out, end_time-start_time))
                if(len(loss_last10) >= 10):
                    loss_last10.popleft()
                    time_last10.popleft()
                loss_last10.append(loss_out)
                time_last10.append(end_time-start_time)
            if input_device_group.contains(local_device) and step == 40 and dp_rank == 0:
               print(f'{local_device}: cuda mem:')
               os.system('nvidia-smi')
                

    output_str = ""
    if(len(loss_last10) == 10 and len(time_last10) == 10):
        print(loss_last10)
        output_str = "%s:\nThe last ten iterations :\nAvg_Loss = %.3f, Avg_Time = %.4f\n" %(local_device, sum(loss_last10) / 10, sum(time_last10) / 10)
    # print("%s: The last ten iterations :\n Avg_Loss = %.3f, Avg_Time = %.4f" %(local_device, sum(loss_last10) / 10, sum(time_last10) / 10))
    if(args.profiler and len(profiler_last10) == 10) :
        graph_view = dict()
        print(profiler_last10[0]["optype_view"])
        for item in profiler_last10[0]["graph_view"]:
            graph_view[item[0]] = 0
        for record in profiler_last10:
            for item in record["graph_view"]:
                graph_view[item[0]] += item[1]
        output_str += "avg-total-run-time : %.4f\n" %(graph_view["total-run-time"] / 10)
        output_str += "avg-optimizer-update : %.4f\n" %(graph_view["optimizer-update"] / 10)
        output_str += "avg-forward-compute : %.4f\n" %(graph_view["forward-compute"] / 10)
        output_str += "avg-backward-compute: %.4f\n" %(graph_view["backward-compute"] / 10)
        output_str += "avg-forward-backward-compute : %.4f\n" %(graph_view["forward-backward-compute"] / 10)
        output_str += "avg-tp-p2p : %.4f\n" %(graph_view["tp-p2p"] / 10)
        output_str += "avg-grads-reduce : %.4f\n" %(graph_view["grads-reduce"] / 10)
        output_str += "avg-tp-collective : %.4f\n" %(graph_view["tp-collective"] / 10)
        output_str += "avg-blocking : %.4f\n" %(graph_view["blocking"] / 10)
        output_str += "avg-other : %.4f\n" %(graph_view["other"] / 10)
    # print(output_str, end = "")
    # save_checkpoint(model, "./checkpoint/temp", config=config, local_device=local_device)
    # print(f"device = {local_device}, test weight = {model.state_dict()['transformer.h.5.mlp.parallel_mlp.dense_4h_to_h.weight']}")
    # print(f'device = {local_device}, save model sucessfully!')
    return model, opt, config, local_device


if __name__ == '__main__':
    def setup_seed(seed):
        np.random.seed(seed)
    setup_seed(12345)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_multi_node", action="store_true", help="use multi node (like 2x8 gpus) to run script."
    )
    parser.add_argument(
        '--gpu_id', type=int, default=0, help='Id of GPU to run.'
    )
    parser.add_argument(
        "--ds_parallel_config", default="ds_parallel_config/dp2_tp2_pp2.json", type=str, help="ds parallel config json file"
    )
    parser.add_argument(
        "--global_batch_size", type=int, default=64, help="Training batch size global"
    )
    parser.add_argument(
        "--micro_batch_size", type=int, default=2, help="Training batch size each micro batch"
    )
    parser.add_argument(
        "--dataset", type=str, default='wikicorpus_en', help="Dataset used to train."
    )
    parser.add_argument(
        "--json_file", type=str, help='data json format file path'
    )
    parser.add_argument(
        "--json_key", type=str, help='json key for tokens'
    )
    parser.add_argument(
        "--vocab_file", type=str, help='gpt vocab file path'
    )
    parser.add_argument(
        "--merge_file", type=str, help='gpt merge file path'
    )
    parser.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    parser.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    parser.add_argument(
        "-a",
        "--num_attention_heads",
        type=int,
        default=12,
        help="Number of attention heads",
    )
    parser.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=4, help="Number of epochs"
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of steps for each epoch",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5, help="Learning rate of adam"
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    parser.add_argument(
        "--hidden_act", type=str, default='gelu', help="Hidden activation to use."
    )
    parser.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use Flash Attention."
    )    
    parser.add_argument(
        "--bf16", action="store_true", help="Use bfloat16."
    )
    parser.add_argument(
        "--profiler", action="store_true", help="Use profiler."
    )
    args = parser.parse_args()

    distributed_init(args.use_multi_node)
    with ht.graph("define_and_run"):
        if args.bf16:
            precision = "ht.bfloat16"
        else:
            precision = "ht.float32"
        print(f'{local_device}: use precision {precision}')
        with ht.autocast(eval(precision)):            
            model, opt, config, local_device = pretrain(args)
            print(f'{local_device}: train hspmd ds parallel end...')

    save_path = './checkpoint/'
    directory = os.path.dirname(save_path)

    # 递归创建目录
    if not os.path.exists(directory):
        os.makedirs(directory)
    save_checkpoint(model, opt, save_path, config=config, local_device=local_device)
    
