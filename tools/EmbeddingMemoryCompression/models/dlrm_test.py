from dlrm import DLRM_Head as hspmd_dlrm
# from dlrm_torch import DLRM_Head as torch_dlrm
from dlrm_s_pytorch import DLRM_Net as torch_s_dlrm
import hspmd as ht
import hspmd.layers as htl
import torch
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_embed', type=int, default=100,
                        help='embedding number for each sparse slot')
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='embedding dimension')
    parser.add_argument('--sparse_slot', type=int, default=26,
                        help='number of sparse slot')
    parser.add_argument('--dense_slot', type=int, default=13,
                        help='number of dense slot')
    parser.add_argument('--batch_size', type=int,
                        default=8, help='batch size')
    parser.add_argument('--ln_bot', type=str, default='512-256-64',
                        help='hidden size of bottom mlp layers')
    parser.add_argument('--ln_top', type=str, default='512-256',
                        help='hidden size of top mlp layers')
    parser.add_argument('--test-num', type=int, default=10,
                        help='number of test iterations')
    args = parser.parse_args()

    args.ln_bot = [int(l) for l in args.ln_bot.split('-')]
    args.ln_top = [int(l) for l in args.ln_top.split('-')]

    # use synthetic data
    denses = [np.random.normal(size=(args.batch_size, args.dense_slot)).astype(
        np.float32) for _ in range(args.test_num)]
    sparses = [np.random.randint(0, args.num_embed, size=(
        args.batch_size, args.sparse_slot)).astype(np.int32) for _ in range(args.test_num)]
    offset = np.arange(0, args.num_embed * args.sparse_slot,
                       args.num_embed, dtype=np.int32)
    for sp in sparses:
        sp += offset
    labels = [np.random.randint(0, 2, size=(args.batch_size, 1)).astype(
        np.int32) for _ in range(args.test_num)]

    # initialize hspmd model
    ctx = ht.gpu(0)
    hspmd_embed = htl.Embedding(
        args.num_embed * args.sparse_slot, args.embed_dim, name='weight', ctx=ctx)
    hspmd_model = hspmd_dlrm(args.embed_dim, args.sparse_slot,
                           args.dense_slot, args.ln_bot, args.ln_top)
    hspmd_dense = ht.placeholder_op(name='dense')
    hspmd_sparse = ht.placeholder_op(name='sparse')
    hspmd_label = ht.placeholder_op(name='label')
    embed_input = hspmd_embed(hspmd_sparse)
    loss, prediction = hspmd_model(embed_input, hspmd_dense, hspmd_label)
    opt = ht.optim.SGDOptimizer(learning_rate=0.01)
    train_op = opt.minimize(loss)
    executor = ht.Executor([loss, prediction, train_op], ctx=ctx)

    # initialize torch model
    # torch_embed = nn.Embedding(
    #     args.num_embed * args.sparse_slot, args.embed_dim).cuda(1)
    top_prev = [args.embed_dim + args.sparse_slot *
                (args.sparse_slot + 1) // 2, ]
    torch_model = torch_s_dlrm(args.embed_dim, [args.num_embed] * args.sparse_slot, [
                               args.dense_slot] + args.ln_bot + [args.embed_dim], top_prev + args.ln_top + [1], 'dot', sigmoid_top=2).cuda(1)
    torch_opt = torch.optim.SGD(torch_model.parameters(), lr=0.01)

    # synchronize parameters
    # with open('torchdict.txt', 'w') as fw:
    #     for param, value in torch_model.state_dict().items():
    #         print(param, value.shape, file=fw, flush=True)
    # exit()
    hspmd_dict = {}
    # for k, v in torch_embed.state_dict().items():
    #     param_value = v.cpu().detach().numpy()
    #     hspmd_dict[k] = param_value
    embeds = []
    for k, v in torch_model.state_dict().items():
        print(k, v.shape)
        if k.startswith('emb'):
            embeds.append(v.cpu().detach().numpy())
        else:
            hspmd_name = k.replace('.', '_')
            param_value = v.cpu().detach().numpy()
            if hspmd_name.endswith('weight'):
                param_value = param_value.transpose()
            if hspmd_name.startswith('top_l'):
                hspmd_name = 'top_mlp' + hspmd_name[5:]
            elif hspmd_name.startswith('bot_l'):
                hspmd_name = 'bot_mlp' + hspmd_name[5:]
            hspmd_dict[hspmd_name] = param_value
    hspmd_dict['weight'] = np.concatenate(embeds, axis=0)
    executor.load_dict(hspmd_dict)

    # testing
    for i in range(args.test_num):
        dense = denses[i]
        sparse = sparses[i]
        label = labels[i]

        hspmd_loss_val, hspmd_predict_y, _ = executor.run(
            feed_dict={hspmd_dense: dense, hspmd_sparse: sparse, hspmd_label: label.astype(np.float32)}, convert_to_numpy_ret_vals=True)
        hspmd_loss_val = hspmd_loss_val[0]

        torch_opt.zero_grad()
        torch_sparse_index = [np.arange(args.batch_size)] * args.sparse_slot
        torch_sparse_label = sparse.transpose()
        for i in range(1, args.sparse_slot):
            torch_sparse_label[i] -= args.num_embed * i
        torch_predict_y = torch_model(torch.from_numpy(dense).cuda(1), torch.LongTensor(
            torch_sparse_index).cuda(1), torch.LongTensor(torch_sparse_label).cuda(1))
        torch_loss_val = torch_model.loss_fn(
            torch_predict_y, torch.FloatTensor(label).cuda(1))
        torch_loss_val.backward()
        torch_opt.step()
        torch_loss_val = torch_loss_val.cpu().detach().numpy()
        torch_predict_y = torch_predict_y.cpu().detach().numpy()

        np.testing.assert_allclose(
            hspmd_predict_y, torch_predict_y, atol=2e-6, rtol=1e-5)
        np.testing.assert_allclose(hspmd_loss_val, torch_loss_val, rtol=1e-5)
        print(hspmd_loss_val, torch_loss_val)
