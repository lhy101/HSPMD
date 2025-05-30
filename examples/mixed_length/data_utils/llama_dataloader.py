import torch
import numpy as np
from typing import List

def truncate_fn(batch: List[np.ndarray], pad_id: int, global_token_num: int):
    valid_token_num = sum([np.sum(seq != pad_id) for seq in batch])
    last_seq_valid_token = np.sum(batch[-1] != pad_id)
    assert valid_token_num >= global_token_num and valid_token_num - global_token_num < last_seq_valid_token, "cannot truncate the last seq"
    batch[-1][last_seq_valid_token - (valid_token_num - global_token_num):] = pad_id
    return batch

def build_data_loader(dataset, tokenizer, consumed_samples, global_batch_size=None, global_token_num=None):
    if dataset is None:
        return None
    assert (global_batch_size is None and global_token_num is not None) \
        or (global_batch_size is not None and global_token_num is None), "should only use one of the args: global_batch_size & global_token_num"
    assert hasattr(tokenizer, 'pad'), "tokenizer does not have 'pad' attribute"
    if global_batch_size != None:
        batch_sampler = LLaMANormalSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            global_batch_size=global_batch_size
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
    if global_token_num != None:
        batch_sampler = LLaMAFixedTokenSampler(
            dataset=dataset,
            tokenizer=tokenizer, 
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            global_token_num=global_token_num
        )
        return torch.utils.data.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=lambda batch: truncate_fn(batch, tokenizer.pad, global_token_num)
        )

# directly return the whole global batch, will be split into chunks later
class LLaMANormalSampler:

    def __init__(self, total_samples, consumed_samples, global_batch_size, drop_last=True):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.global_batch_size = global_batch_size
        self.drop_last = drop_last

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        # Last batch will be dropped if drop_last is not set False
        for idx in range(self.consumed_samples, self.total_samples):
            batch.append(idx)
            if len(batch) == self.global_batch_size:
                yield batch
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
            
class LLaMAFixedTokenSampler:

    def __init__(self, dataset, tokenizer, total_samples, consumed_samples, global_token_num, drop_last=True):
        assert hasattr(dataset, '__getitem__'), "dataset does not have '__getitem__' method"
        assert callable(getattr(dataset, '__getitem__')), "dataset '__getitem__' method is not callable"
        assert hasattr(tokenizer, 'pad'), "tokenizer does not have 'pad' attribute"
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.global_token_num = global_token_num
        self.drop_last = drop_last
        self.total_samples = len(dataset)

        # Sanity checks
        assert self.total_samples > 0, \
            'No samples to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'No samples left to consume: {}, {}'.format(self.consumed_samples, self.total_samples)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        batch = []
        current_token_count = 0
        for idx in range(self.consumed_samples, self.total_samples):
            sample = self.dataset.__getitem__(idx)
            sample_token_count = np.sum(np.array(sample) != self.tokenizer.pad) # 获取当前样本的token数
            batch.append(idx)
            # print(idx, sample_token_count, sample != self.tokenizer.pad)
            if current_token_count + sample_token_count >= self.global_token_num:
                yield batch
                batch = []
                current_token_count = 0
            else:
                current_token_count += sample_token_count

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            yield batch
