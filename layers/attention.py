import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from train.tool import TriangularCausalMask
from layers.PAPE import PAPE
import math

class Fullattention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1):
        super(Fullattention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, embed = False, seasonal_inf=None, type='seasonal'):
        B, L, H, E = queries.shape #(32*96*8*64)
        _, S, _, D = values.shape
        if embed is True:   
            attention_embedding = PAPE(seasonal_inf, D, H)
            queries_1, keys_1 = attention_embedding(queries, keys, type=type)
            
            queries = queries_1 + queries
            keys = keys_1 + keys

        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return (V.contiguous(), A)
        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super(AttentionLayer, self).__init__()

        d_keys = d_model // n_heads
        d_values = d_model // n_heads

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)  
        self.value_projection = nn.Linear(d_model, d_keys * n_heads)   
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, embed = False, seasonal_inf=None, type='seasonal'): 
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            embed,
            seasonal_inf=seasonal_inf,
            type=type
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn