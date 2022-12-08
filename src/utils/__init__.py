import logging
import warnings
import functools
import math
import os
from typing import List

import cv2

import torch
from torchvision.datasets.folder import is_image_file
import torch.nn.functional as F

from einops import rearrange, repeat

from src.utils.metrics import GroundTruthMatch




@functools.lru_cache()
def get_image_paths(path: str) -> List:
    logging.info(f"Resolving files in: {path}")
    filenames = [f"{file}" for file in os.listdir(path)]
    return path, sorted([fn for fn in filenames if is_image_file(fn)])



def read_ground_truth(filename: str) -> List[GroundTruthMatch]:
    """
    Read groundtruth csv file.
    Must contain query_image_id,db_image_id on each line.
    handles the no header version and DD's version with header
    """
    gt_pairs = []
    with open(filename, "r") as cfile:
        for line in cfile:
            line = line.strip()
            if line == "query_id,reference_id":
                continue
            q, db = line.split(",")
            if db == "":
                continue
            gt_pairs.append(GroundTruthMatch(q, db))
    return gt_pairs



def readimage(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image



def gem_pool(x, p = 3, eps = 1e-6):
    return F.avg_pool2d(x.clamp(min = eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)



def get_region_vector(embeddings: List[torch.Tensor],
                      attn: List[torch.Tensor],
                      dim,
                      h,
                      ) -> List[torch.Tensor]:
    img_tokens = embeddings[:, 1:, :] # B x N x d
    
    cls_attn = attn[:, :, 0, 1:] # B x nh x 1 x N from B x nh x N+1 x N+1
    cls_attn = cls_attn.mean(dim = 1, keepdim = True) # Average across all heads. B x 1 x N
    #cls_attn = cls_attn.max(dim = 1, keepdim = True) # Average across all heads. B x 1 x N
    
    cls_attn = rearrange(cls_attn, 'B h N -> B N h') # B x N x 1
    cls_attn = repeat(cls_attn, 'B N h -> B N (repeat h)', repeat = dim) # B x N x d
    
    region = img_tokens * cls_attn # Elementwise multiplication with class attention
    region = rearrange(region, 'B (h w) d -> B d h w', h = h)
    region = F.normalize(region) # Normalize region vector
    
    gem_pooled_region = gem_pool(region) # Gem pooled along spatial dimensions (h w)
    #gem_pooled_region = F.max_pool2d(region, kernel_size = (region.shape[-2],region.shape[-1]))
    
    gem_pooled_region = rearrange(gem_pooled_region, 'B d h w -> B (d h w)') # Output is B x d
    
    return region



def get_local_vector(embeddings: List[torch.Tensor], h):
        feats = embeddings[:, 1:]
        feats = rearrange(feats, 'B (h w) d ->  B d h w', h = h)
        feats = gem_pool(feats)
        feats = rearrange(feats, 'B d h w -> B (d h w)')
        return feats