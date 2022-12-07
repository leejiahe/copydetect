from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_

from einops import rearrange, repeat



class FFN(nn.Module):
    def __init__(self,
                 dim:int = 768,
                 mlp_ratio:int = 4,
                 dropout = 0., 
                ):
        super().__init__()
        
        hidden_dim = dim * mlp_ratio
        self.ffn = nn.Sequential(nn.Linear(dim, hidden_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim),
                                 nn.Dropout(dropout),
                                )
    #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.ffn(x)
    
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self,
                n_head:int = 12,
                d_head:int = 64,
                d_model:int = 768,
                qkv_bias = False,
                attn_drop:float = 0.,
                proj_drop:float = 0.,
                ):
        super().__init__()
        
        self.n_head = n_head
        self.w_q = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        self.w_k = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        self.w_v = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        
        self.proj = nn.Linear(n_head * d_head, d_model)
        
        self.attn_drop = nn.Dropout(p = attn_drop)
        self.proj_drop = nn.Dropout(p = proj_drop)
        
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(self,
               q: List[torch.Tensor],
               k: List[torch.Tensor],
               v: List[torch.Tensor],
               ):
        
        q = rearrange(self.w_q(q), 'b n (h d) -> b h n d', h = self.n_head)
        k = rearrange(self.w_k(k), 'b n (h d) -> b h n d', h = self.n_head)
        v = rearrange(self.w_v(v), 'b n (h d) -> b h n d', h = self.n_head)
        
        attn  = torch.einsum('b h i d, b h j d -> b h i j', q, k) / np.sqrt(q.shape[-1])
        attn = torch.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)

        return out, attn
    
    

# Inspired by https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
class MultiHeadCosSimAttention(nn.Module):
    def __init__(self,
                n_head:int = 12,
                d_head:int = 64,
                d_model:int = 768,
                qkv_bias = False,
                attn_drop:float = 0.,
                proj_drop:float = 0.,
                ):
        super().__init__()
        
        self.n_head = n_head
        
        self.qkv = nn.Linear(d_model, n_head * d_head * 3, bias = qkv_bias)
        
        self.w_q = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        self.w_k = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        self.w_v = nn.Linear(d_model, n_head * d_head, bias = qkv_bias)
        
        self.proj = nn.Linear(n_head * d_head, d_model)
        
        self.attn_drop = nn.Dropout(p = attn_drop)
        self.proj_drop = nn.Dropout(p = proj_drop)
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((n_head, 1, 1))), requires_grad = True)
        
        #self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            
    def forward(self,
                q: List[torch.Tensor],
                k: List[torch.Tensor],
                v: List[torch.Tensor],
                ):
        
        
        q = rearrange(self.w_q(q), 'b n (h d) -> b h n d', h = self.n_head)
        q = F.normalize(q, dim = -1)
        
        k = rearrange(self.w_k(k), 'b n (h d) -> b h n d', h = self.n_head)
        k = F.normalize(k, dim = -1)
        
        v = rearrange(self.w_v(v), 'b n (h d) -> b h n d', h = self.n_head)
        
        attn  = torch.einsum('b h i d, b h j d -> b h i j', q, k)
        
        logit_scale = torch.clamp(self.logit_scale, max = torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale
        attn = torch.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)
        
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = self.proj_drop(out)
    
        return out
    
    
    
class CrossAttentionBlock(nn.Module):
    def __init__(self,
                 n_head:int = 12,
                 d_head:int = 64,
                 d_model:int = 768,
                 mlp_ratio:int = 4,
                 attn_drop:float = 0,
                 proj_drop:float = 0,
                 drop_path:float = 0.,
                 qkv_bias:bool = False
                ):
        super().__init__()
        
        self.attn = MultiHeadAttention(n_head = n_head,
                                       d_head = d_head,
                                       d_model = d_model,
                                       qkv_bias = qkv_bias,
                                       attn_drop = attn_drop,
                                       proj_drop = proj_drop)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = FFN(dim = d_model,
                       mlp_ratio = mlp_ratio,
                       dropout = proj_drop)
        
        self.norm2 = nn.LayerNorm(d_model)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        #self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, qry, ref):
        x, _ = self.attn(q = qry, k = ref, v = ref)
        x = self.norm1(x + self.drop_path(x)) # Qry is query embeddings and key, value is ref embeddings
        x = self.norm2(x + self.drop_path(self.ffn(x))) # Add (residual) & Norm
        
        return x
    
    
    
class SelfAttentionBlock(nn.Module):
    def __init__(self,
                 n_head:int = 12,
                 d_head:int = 64,
                 d_model:int = 768,
                 mlp_ratio:int = 4,
                 attn_drop:float = 0,
                 proj_drop:float = 0,
                 drop_path:float = 0.,
                 qkv_bias:bool = False,
                ):
        super().__init__()
        
        self.attn = MultiHeadAttention(n_head = n_head,
                                       d_head = d_head,
                                       d_model = d_model,
                                       qkv_bias = qkv_bias,
                                       attn_drop = attn_drop,
                                       proj_drop = proj_drop)
        
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = FFN(dim = d_model,
                       mlp_ratio = mlp_ratio,
                       dropout = proj_drop)
        
        self.norm2 = nn.LayerNorm(d_model)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        #self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_attn = False):
        x, attn = self.attn(q = x, k = x, v = x)
        if return_attn:
            return attn
        x = self.norm1(x + self.drop_path(x)) # Add (residual) & Norm
        x = self.norm2(x + self.drop_path(self.ffn(x))) # Add (residual) & Norm
        
        return x
    
    
    
class CopyHead(nn.Module):
    def __init__(self,
                 n_head:int = 12,
                 d_head:int = 64,
                 d_model:int = 768,
                 mlp_ratio:int = 4,
                 attn_drop:float = 0,
                 proj_drop:float = 0,
                 drop_path:float = 0.,
                 depth:int = 4,
                 bottleneck_dim:int = 256,
                 qkv_bias:bool = False):
        
        super().__init__()
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.sep_token = nn.Parameter(torch.zeros(1, 1, d_model))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth + 1)]  # stochastic depth decay rule
        self.co_blocks = CrossAttentionBlock(n_head = n_head,d_head = d_head,d_model = d_model,mlp_ratio = mlp_ratio,attn_drop = attn_drop,proj_drop = proj_drop,drop_path = drop_path)
        self.sa_blocks = nn.ModuleList([SelfAttentionBlock(n_head = n_head,d_head = d_head,d_model = d_model,mlp_ratio = mlp_ratio,attn_drop = attn_drop,proj_drop = proj_drop,drop_path = drop_path)
                                     for i in range(depth)])
        
        """
        self.blocks = nn.ModuleList([nn.Sequential(CrossAttentionBlock(n_head = n_head,
                                                                       d_head = d_head,
                                                                       d_model = d_model,
                                                                       mlp_ratio = mlp_ratio,
                                                                       attn_drop = attn_drop,
                                                                       proj_drop = proj_drop,
                                                                       drop_path = drop_path),
                                                   SelfAttentionBlock(n_head = n_head,
                                                                      d_head = d_head,
                                                                      d_model = d_model,
                                                                      mlp_ratio = mlp_ratio,
                                                                      attn_drop = attn_drop,
                                                                      proj_drop = proj_drop,
                                                                      drop_path = drop_path))
                                     for i in range(depth)])
        
        self.sa_blocks = nn.Sequential(*[SelfAttentionBlock(n_head = n_head,
                                                            d_head = d_head,
                                                            d_model = d_model,
                                                            mlp_ratio = mlp_ratio,
                                                            attn_drop = attn_drop,
                                                            proj_drop = proj_drop,
                                                            drop_path = drop_path)
                                         for i in range(2)])
        self.head = nn.Sequential(nn.Linear(d_model, bottleneck_dim),
                                  nn.GELU(),
                                  nn.Linear(bottleneck_dim, 1),
                                  nn.Sigmoid())
        """
        
        """
        self.ffn = FFN(dim = d_model,
                       mlp_ratio = mlp_ratio,
                       dropout = attn_drop)
        
        # Classifier head
        self.head = nn.Sequential(nn.LayerNorm(d_model),
                                  nn.Linear(d_model, 1),
                                  nn.Sigmoid())
        
        """
        
        # Classifier head
        self.head = nn.Sequential(nn.Linear(d_model, 1),
                                  nn.Sigmoid())

        
        trunc_normal_(self.sep_token, std = .02)
        trunc_normal_(self.cls_token, std = .02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, query_emb, refer_emb):
        batch_size = query_emb.shape[0]
        
        # Append [CLS] token to the query
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        query_emb = torch.cat((cls_tokens, query_emb), dim = 1)

        # Append [SEP] token to the reference
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)
        refer_emb = torch.cat((sep_tokens, refer_emb), dim = 1)

        return query_emb, refer_emb

    def forward_multiple_ca(self, query_emb, refer_emb):
        query_emb, refer_emb = self.prepare_tokens(query_emb, refer_emb)
        
        for ca_blk, sa_blk in self.blocks:
            x = ca_blk(query_emb, refer_emb)
            query_emb = sa_blk(x)
            
        x = self.sa_blocks(query_emb)
        logits = self.head(x[:, 0])
        
        return logits
    
    def forward_gap(self, query_emb, refer_emb):
        query_emb, refer_emb = self.prepare_tokens(query_emb, refer_emb)
        
        x = self.co_blocks(qry = query_emb, ref = refer_emb)
        
        for blk in self.sa_blocks:
            x = blk(x)
        
        x = self.ffn(x)
        
        # Global average pooling
        x = torch.mean(x, dim = 1)
        
        logits = self.head(x)
        
        return logits
    
    def forward(self, query_emb, refer_emb):
        query_emb, refer_emb = self.prepare_tokens(query_emb, refer_emb)
        
        x = self.co_blocks(qry = query_emb, ref = refer_emb)
        
        for blk in self.sa_blocks:
            x = blk(x)
        
        logits = self.head(x[:, 0])
        
        return logits
    
    def get_last_selfattention(self, query_emb, refer_emb):
        query_emb, refer_emb = self.prepare_tokens(query_emb, refer_emb)
        
        x = self.co_blocks(qry = query_emb, ref = refer_emb)
        
        for i, blk in enumerate(self.sa_blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attn = True)        