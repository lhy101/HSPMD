import hspmd
import numpy as np

from hspmd.utils.parallel import get_multi_ds_parallel_config

def generate_cos_sin(seqlen, rotary_dim, dtype):
    assert rotary_dim % 2 == 0
    angle = np.random.rand(seqlen * 2, rotary_dim // 2) * 2 * np.pi
    cos = np.cos(angle).astype(dtype)
    sin = np.sin(angle).astype(dtype)
    return cos, sin
  
# self-attn
class LLamaAttention(hspmd.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='attn'):
        super().__init__()

        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.use_flash_attn = config.use_flash_attn
        # self.add_bias = True
        self.add_bias = False

        # max_positions = config.max_position_embeddings
        # self.bias = np.tril(np.ones((max_positions, max_positions), dtype=np.int64).reshape(
        #             1, 1, max_positions, max_positions))
        self.masked_value = -1e4

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_groups = config.num_query_groups
        if self.num_heads % self.num_groups != 0:
            raise ValueError(
                f"num_heads must be divisible by num_groups (got {self.num_heads} vs. {self.num_groups})"
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.query_projection_size = self.embed_dim
        self.kv_projection_size = self.num_groups * self.head_dim
        self.num_heads_per_group = self.num_heads // self.num_groups
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        self.qkv_dense = hspmd.nn.HtMultiColumnParallelLinear(
            self.embed_dim,
            self.query_projection_size + 2 * self.kv_projection_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'qkv', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
        )

        self.dense = hspmd.nn.HtMultiRowParallelLinear(
            self.embed_dim,
            self.embed_dim,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense', layer_idx),
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
        )

        self.attn_dropout = hspmd.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = hspmd.nn.Dropout(config.resid_pdrop)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # column parallel, [micro_batch_size * seq_len, 3 * embed_dim]
        qkv = self.qkv_dense(hidden_states)

        '''
        # apply relative positional encoding (rotary embedding)
        # TODO: 支持动态seq_len
        def apply_rotary_pos_emb(x, _name='q'):
            cos_np, sin_np = generate_cos_sin(self.config.seq_len_symbol.data, int(0.5 * self.head_dim), np.float32)
            device_group_hierarchy = self.qkv_dense.device_group_unions
            ds_hierarchy = self.dense.ds_union_map['dup']
            # 去除zero
            ds_hierarchy = [
                hspmd.DistributedStatesUnion([hspmd.DistributedStates(ds.device_num, {-1: ds.device_num}, [-1]) for ds in ds_union.ds_list], ds_union.hetero_dim)
                    for ds_union in ds_hierarchy
            ]
            sin_global = hspmd.from_numpy_parallel(sin_np, ds_hierarchy, device_group_hierarchy=device_group_hierarchy, requires_grad=False, name=f'sin_{_name}')
            cos_global = hspmd.from_numpy_parallel(cos_np, ds_hierarchy, device_group_hierarchy=device_group_hierarchy, requires_grad=False, name=f'cos_{_name}')
            out = hspmd.rotary(x, cos_global, sin_global, inplace=True)
            return out
        # query = apply_rotary_pos_emb(query, _name='q')
        # key = apply_rotary_pos_emb(key, _name='k')
        '''
        
        assert self.use_flash_attn, "currently only support flash attn"
        # already support packing api
        attn_output = hspmd.parallel_attn(
            qkv,             
            self.head_dim, 
            1, # group_query_ratio = q heads / k(v) heads, 1 means MHA and >1 means GQA
            self.config.multi_seq_lens_symbol, 
            self.config.multi_cp_group_symbol,
            self.config.packing,
            self.config.cu_seqlens_list[self.layer_idx],
            self.config.cu_seqlens_list[self.layer_idx],
            self.config.max_seqlen_symbol,
            self.config.max_seqlen_symbol
        )[0]
        
        '''
        # [mbs, seq_len, num_heads, 3 * head_dim]
        qkv = qkv.reshape([self.config.mbs_times_dp_symbol, self.config.seq_len_symbol, hspmd.IntSymbol(self.num_heads), hspmd.IntSymbol(3 * self.head_dim)])
        # [mbs, seq_len, num_heads, head_dim]
        query, key, value = hspmd.split(qkv, 3, qkv.ndim - 1)
        attn_output = hspmd.attn(query, key, value, 0, -1, True)[0]
        # [mbs * seq_len, num_heads * head_dim]
        attn_output = attn_output.reshape([self.config.mbs_times_dp_symbol * self.config.seq_len_symbol, hspmd.IntSymbol(self.num_heads * self.head_dim)])
        '''
        
        # row parallel, shape = [mbs * seq_len, num_heads * head_dim]
        attn_output = self.dense(attn_output)
        # dropout
        # attn_output = self.resid_dropout(attn_output)
        return attn_output



class ParallelMLP(hspmd.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(ParallelMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        # self.add_bias = True
        self.add_bias = False
        
        self.swiglu = True
        ffn_hidden_size = config.ffn_hidden_size # 2.7h
        if self.swiglu:
            ffn_hidden_size *= 2 # for swiglu: h -> 2 * 2.7h

        self.dense_h_to_4h = hspmd.nn.HtMultiColumnParallelLinear(
            config.hidden_size,
            ffn_hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_h_to_4h', layer_idx),
            bias=self.add_bias,
            gather_output=False,
            name=f'colp_{name}'
            # skip_bias_add=True
        )

        # self.bias_gelu_fusion = bias_gelu_fusion
        self.activation_func = hspmd.nn.NewGeLU(get_multi_ds_parallel_config(ds_parallel_configs, 'activation_func', layer_idx))

        self.dense_4h_to_h = hspmd.nn.HtMultiRowParallelLinear(
            config.ffn_hidden_size,
            config.hidden_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'dense_4h_to_h', layer_idx),
            sequence_parallel=True,
            bias=self.add_bias,
            name=f'rowp_{name}'
            # init_method=output_layer_init_method
        )

        self.dropout = hspmd.nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states):
        # [b * seq_len, h] -> [b * seq_len, 4h]
        intermediate_parallel = self.dense_h_to_4h(hidden_states)
        # intermediate_parallel = self.activation_func(intermediate_parallel)
        with hspmd.recompute(multi_recompute = [
            [False] if ds_parallel_config['recompute_granularity'] is None else
            [
                True if dp_recompute_granularity == 'selective' and self.layer_idx in recompute_layer_idxs else False
                for dp_recompute_granularity, recompute_layer_idxs in zip(ds_parallel_config['recompute_granularity'], ds_parallel_config['recompute_layer_idxs_list'])
            ]
            for ds_parallel_config in self.ds_parallel_configs
        ]):
            intermediate_parallel = hspmd.swiglu(intermediate_parallel)

        # [b * seq_len, 4h] -> [b * seq_len, h]
        output = self.dense_4h_to_h(intermediate_parallel)
        # output = self.dropout(output)
        return output

class LLamaMLP(hspmd.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx, name='mlp'):
        super(LLamaMLP, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.parallel_mlp = ParallelMLP(config, ds_parallel_configs, layer_idx, name)

    def forward(self, hidden_states):
        origin_shape = hidden_states.global_shape # [b * seq_len, hidden_size]
        assert len(origin_shape) == 2, "sp: all is 2 dim matmul"
        hidden_states = self.parallel_mlp(hidden_states)
        return hidden_states

class LLamaBlock(hspmd.nn.Module):
    def __init__(self, config, ds_parallel_configs, layer_idx):
        super().__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size

        # sequence parallel: rmsnorm前做reduce-scatter(这一部分由row prallel的reduce-scatter完成); rmsnorm后做allgather
        self.rmsnorm_1 = hspmd.nn.HtMultiParallelRMSNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm1', layer_idx), sequence_parallel=True, name=f'rmsnorm1_block{layer_idx}')
        self.attn = LLamaAttention(config, get_multi_ds_parallel_config(ds_parallel_configs, "attn", layer_idx), layer_idx=layer_idx, name=f'attn_block{layer_idx}')
        self.rmsnorm_2 = hspmd.nn.HtMultiParallelRMSNorm(hidden_size, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm2', layer_idx), sequence_parallel=True, name=f'rmsnorm2_block{layer_idx}')
        self.mlp = LLamaMLP(config, get_multi_ds_parallel_config(ds_parallel_configs, "mlp", layer_idx), layer_idx=layer_idx, name=f'mlp_block{layer_idx}')

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        residual = hidden_states
        hidden_states = self.rmsnorm_1(hidden_states)
        attn_output = self.attn(
            hidden_states, # [b, seq_len, hidden_size]
            attention_mask=attention_mask # [b, 1, 1, seq_len]
        )
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.rmsnorm_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        return hidden_states


class LLamaModel(hspmd.nn.Module):
    def __init__(self, config, ds_parallel_configs):
        super(LLamaModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.dtype = hspmd.float32

        self.embed_dim = config.hidden_size
        self.wte = hspmd.nn.HtMultiVocabParallelEmbedding(config.vocab_size, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wte'), name='wte')
        # self.wpe = hspmd.nn.HtMultiParallelEmbedding(config.max_position_embeddings, self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'wpe'), name='wpe')

        self.drop = hspmd.nn.Dropout(config.embd_pdrop)
        blocks = []
        for i in range(config.num_hidden_layers):
            blocks.append(LLamaBlock(config, get_multi_ds_parallel_config(ds_parallel_configs, f'blocks{i}'), layer_idx=i))
        self.h = hspmd.nn.ModuleList(blocks)
        self.rmsnorm_f = hspmd.nn.HtMultiParallelRMSNorm(self.embed_dim, get_multi_ds_parallel_config(ds_parallel_configs, 'rmsnorm_final'), sequence_parallel=True, name='rmsnorm_final')

    def forward(
        self,
        input_ids,
        position_ids,
        attention_mask=None,
        token_type_ids=None,
    ):
        # input_ids: [b * seq_len]        
        # token_type_ids: [b * seq_len]
        if token_type_ids is not None:
            assert token_type_ids.global_shape == input_ids.global_shape \
                and token_type_ids.distributed_states.check_equal(input_ids.distributed_states), \
                'token_type_ids global_shape and distributed_states should be equal to input_ids'

        # embeddding: [b * seq_len, embed_dim]
        inputs_embeds = self.wte(input_ids) # [b * seq_len, embed_dim]
        # position_embeds = self.wpe(position_ids) # [b * seq_len, embed_dim]
        # hidden_states = inputs_embeds + position_embeds # [b * seq_len, embed_dim]
        hidden_states = inputs_embeds
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids) # [b * seq_len, embed_dim]
            hidden_states = hidden_states + token_type_embeds
        # dropout
        # hidden_states = self.drop(hidden_states)
        
        # for sequence parallel
        # todo: this is pretty hacky, find a better way
        sp = True
        if sp:
            ds_hierarchy_input = hidden_states.ds_hierarchy
            ds_hierarchy_output = []
            for ds_union_input in ds_hierarchy_input:
                ds_list_split0 = []
                for ds_input in ds_union_input.ds_list:
                    ds_split0 = hspmd.DistributedStates(ds_input.device_num, {0: ds_input.device_num}, [0])
                    assert ds_union_input.hetero_dim == -3 or ds_union_input.hetero_dim == 0, \
                        "Workaround: sp assume input only hetero on split0"
                    assert ds_input.device_num == ds_input.get_dim(0) * ds_input.get_dim(-1), \
                        "Workaround: sp assume input only split in dimension 0 for dp"
                    ds_list_split0.append(ds_split0)
                ds_hierarchy_output.append(hspmd.DistributedStatesUnion(ds_list_split0, 0 if ds_union_input.hetero_dim != -3 else -3))
            # [b * seq_len // tp, embed_dim]
            hidden_states = hspmd.comm(hidden_states, ds_hierarchy_output, name="workaround_sp_scatter")

        # 12 x multihead self-attn
        for i, block in enumerate(self.h):
            hidden_states = block(
                hidden_states, # [b * seq_len, embed_dim]
                attention_mask=attention_mask # [b, 1, 1, seq_len]
            )
            # hetero需要显示地插入通信算子
            if i != len(self.h) - 1:
                next_block = self.h[i + 1]
                if next_block.rmsnorm_1.sequence_parallel:
                    hidden_states = hspmd.comm(hidden_states, next_block.rmsnorm_1.ds_union_map['split0'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
                else:
                    hidden_states = hspmd.comm(hidden_states, next_block.attn.qkv_dense.ds_union_map['split0_dup'], next_block.rmsnorm_1.device_group_unions, name=f"pipeline_layer_{i}_comm")
        # rmsnorm
        hidden_states = self.rmsnorm_f(hidden_states)
        return hidden_states

class LLaMALMHeadModel(hspmd.nn.Module):

    def __init__(self, config, ds_parallel_configs):
        super(LLaMALMHeadModel, self).__init__()
        self.config = config
        self.ds_parallel_configs = ds_parallel_configs
        self.transformer = LLamaModel(config, get_multi_ds_parallel_config(ds_parallel_configs, 'llama'))
        self.lm_head = hspmd.nn.HtMultiColumnParallelLinear(
            config.n_embd,
            config.vocab_size,
            get_multi_ds_parallel_config(ds_parallel_configs, 'lm_head'),
            bias=False,
            gather_output=False,
            name='lm_head'
        )
        # share embedding table
        # we manually add comm op here
        # because we don't know if it is a P2P or a BatchedIsendIrecv in hetero settings
        # self.lm_head.weight = hspmd.comm(self.transformer.wte.embedding_table, self.lm_head.ds_union_map['dup_split0'], self.lm_head.device_group_unions, name="share_weight_comm") 
        self.config = config
    
    def forward(
        self, 
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        loss_mask=None,
        token_type_ids=None,
        labels=None
    ):
        # [b * seq_len, n_embd]
        hidden_states = self.transformer(
            input_ids,
            position_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        '''
        # need allgather here: [b * s // tp, h] -> [b * s, h]
        if not hidden_states.check_ds_hierarchy_equal(self.lm_head.ds_union_map['split0_dup']):
            hidden_states = hspmd.comm(hidden_states, self.lm_head.ds_union_map['split0_dup'])
        '''
        
        # column parallel, [b * seq_len, n_embd] -> [b * seq_len, vocab_size], and splited in vocab dimension
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # print(f"lm_logits shape {lm_logits.shape}, labels shape {labels.shape}")
            loss_unreduced = hspmd.vocab_parallel_cross_entropy(lm_logits, labels, ignored_index = -1, reduction = "none").reshape([-1])
            loss_sum = hspmd.sum(hspmd.mul(loss_unreduced, loss_mask))
            loss_valid_tokens = hspmd.sum(loss_mask)
            loss = hspmd.div(loss_sum, loss_valid_tokens)
        return loss
