# Copied from https://github.com/facebookresearch/sscd-copy-detection/blob/95902662f2217a5f4aa45f2a3fc70a01dfd3b66a/sscd/lib/distributed_util.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from classy_vision.generic.distributed_util import (all_reduce_mean,
                                                    all_reduce_sum,
                                                    get_world_size,
                                                    get_rank,
                                                    )
from torch import autograd, distributed



def multi_gather_batch(*tensors):
    world_size = distributed.get_world_size()
    out = []
    handles = []

    for tensor in tensors:
        gathered_shape = (world_size * tensor.shape[0],) + tensor.shape[1:]
        gathered = torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor.contiguous()
        handle = distributed.all_gather(list(torch.chunk(gathered, world_size)), tensor, async_op=True)

        out.append(gathered)
        handles.append(handle)

    for handle in handles:
        handle.wait()

    return out



class GatherAcrossGPU(autograd.Function):
    @staticmethod
    def forward(ctx, embeddings, target, reduce_method):
        ctx.n = embeddings.size(0)
        ctx.reduce_method = reduce_method
        ctx.world_size = get_world_size()
        if target is None:
            if ctx.world_size == 1:
                return embeddings
            else:
                return multi_gather_batch(embeddings)[0]

        assert ctx.n == target.size(0)
        if ctx.world_size == 1:
            ctx.mark_non_differentiable(target)
            return embeddings, target
        all_embeddings, all_target = multi_gather_batch(embeddings, target)
        ctx.mark_non_differentiable(all_target)
        return all_embeddings, all_target

    @staticmethod
    def backward(ctx, all_embeddings_gradient, ignored_target_grad=None):
        if ctx.world_size == 1:
            embeddings_gradient = all_embeddings_gradient
        else:
            # Aggregate gradients across nodes.
            if ctx.reduce_method == 'mean':
                all_reduce_mean(all_embeddings_gradient)
            elif ctx.reduce_method == 'sum':
                all_reduce_sum(all_embeddings_gradient)
            else:
                # Do not accumulate.
                raise ValueError('Reduce method not found')
            rank = get_rank()
            start = ctx.n * rank
            end = start + ctx.n
            # Slice gradient for embeddings that belong to this node.
            embeddings_gradient = all_embeddings_gradient[start:end]
        return (embeddings_gradient, None, None)
    
    
    
def gather_across_gpu(embeddings, targets, reduce_method='sum'):
    return GatherAcrossGPU.apply(embeddings, targets, reduce_method)