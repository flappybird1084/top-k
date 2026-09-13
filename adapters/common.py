import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, causal=False):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3*dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(approximate='tanh'), nn.Linear(dim*4, dim))
        self.heads, self.causal = heads, causal

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        attn = F.scaled_dot_product_attention(q,k,v,is_causal=self.causal)
        x = x+self.proj(attn.transpose(1,2).reshape(b,t,d))
        return x+self.mlp(self.norm2(x))
