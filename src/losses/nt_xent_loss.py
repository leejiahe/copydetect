#Inspired from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

import torch

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        ctx.batch_size = tensor.shape[0]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        
        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op = torch.distributed.ReduceOp.SUM, async_op = False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class NTXentLoss:
    def __init__(self, temperature = 0.9, eps: float = 1e-5):
        self.temperature = temperature
        self.eps = eps
    
    def __call__(self, img_r, img_q, id_r, id_q) -> torch.Tensor:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            img_r_gathered = SyncFunction.apply(img_r)
            img_q_gathered = SyncFunction.apply(img_q)
            id_q_gathered = SyncFunction.apply(id_q)
            
        else:
            img_r_gathered = img_r
            img_q_gathered = img_q
            id_q_gathered = id_q

        img_rq = torch.cat([img_r, img_q], dim = 0) # [2 x batch_size, dim]
        img_rq_gathered = torch.cat([img_r_gathered, img_q_gathered], dim = 0) # [2 x world_size x batch_size, dim]
        cov = torch.divide(torch.mm(img_rq, img_rq_gathered.t().contiguous()), self.temperature) # [2 x batch_size, 2 x world_size x batch_size]
        
        # for numerical stability
        logits_max, _ = torch.max(img_rq, dim = 1, keepdim = True) # [2 x batch_size, 1]
        logits = cov - logits_max.detach() # [2 x batch_size, 2 x world_size x batch_size]
        
        # mask: [batch_size, world_size x batch_size]
        mask = torch.eq(id_r.unsqueeze(dim = 1), id_q_gathered.unsqueeze(dim = 1).t()).float()
        # Tile mask to same shape as logits
        mask = mask.repeat(2, 2) # From [batch_size, world_size x batch_size] to [2 x batch_size, 2 x world_size x batch_size]
        # Zero img_r and img_q to its own transposed term by making diagonal to 0 
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask
        # Remove entries without corresponding img_q pair
        paired = mask.sum(dim = 1) > 0 
        mask = mask[paired]
        logits = logits[paired]
        logits_mask = logits_mask[paired]
        
        # Log probability
        exp_logits = torch.exp(logits) * logits_mask # [2 x batch_size, 2 x world_size x batch_size]
        log_prob = logits - torch.log(exp_logits.sum(dim = 1, keepdim = True)) # [2 x batch_size, 2 x world_size x batch_size]
        mask_log_prob = - (mask * log_prob).sum(dim = 1) / mask.sum(dim = 1) # [2 x batch_size, 1]
        
        return mask_log_prob.mean()

	