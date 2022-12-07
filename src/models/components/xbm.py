from typing import List, Tuple

import torch



class XBM:
    def __init__(self,
                 embedding_size: int,
                 memory_size: int = 1024,
                ):
        self.memory_size = memory_size
        self.feats = torch.zeros(memory_size, embedding_size)
        self.targets = torch.zeros(memory_size, dtype = torch.long)
        self.ptr = 0

    @property
    def is_full(self) -> bool:
        return self.targets[-1].item() != 0

    def get(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if self.is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue(self,
                feats: List[torch.Tensor],
                targets: List[torch.Tensor]
               ) -> None:
        q_size = len(targets)
        
        if self.ptr + q_size > self.memory_size:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size