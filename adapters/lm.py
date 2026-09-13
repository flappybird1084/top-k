"""Language-model demo adapter (spec §3.2): decoder-only, ~27M params, seq 512.

Data is one synthetic fixed-seed token shard (no downloads, exactly reproducible
order). Swap `get_dataloader` for a real shard freely, keeping fixed order.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelevo import ops

VOCAB = 8192
DIM = 512
DEPTH = 6
HEADS = 8
SEQ = 512
BATCH = 8
DATA_SEED = 9000
STEPS_PER_EPOCH = 200


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.ln1_w = nn.Parameter(torch.ones(dim))
        self.ln1_b = nn.Parameter(torch.zeros(dim))
        self.ln2_w = nn.Parameter(torch.ones(dim))
        self.ln2_b = nn.Parameter(torch.zeros(dim))
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        B, T, D = x.shape
        h = ops.layer_norm(x, self.ln1_w, self.ln1_b)
        qkv = self.qkv(h).reshape(B, T, 3, self.heads, D // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = ops.layer_norm(x, self.ln2_w, self.ln2_b)
        h = ops.gelu_mlp(h, self.fc1.weight, self.fc1.bias)
        return x + self.fc2(h)


class LM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Parameter(torch.randn(1, SEQ, DIM) * 0.02)
        self.blocks = nn.ModuleList(Block(DIM, HEADS) for _ in range(DEPTH))
        self.ln_w = nn.Parameter(torch.ones(DIM))
        self.ln_b = nn.Parameter(torch.zeros(DIM))
        self.head = nn.Linear(DIM, VOCAB, bias=False)
        nn.init.normal_(self.tok.weight, std=0.02)
        self.head.weight = self.tok.weight

    def forward(self, x):
        h = self.tok(x) + self.pos[:, : x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        h = ops.layer_norm(h, self.ln_w, self.ln_b)
        return self.head(h)


def build_model():
    return LM()


def get_dataloader(split: str):
    seed = DATA_SEED if split == "train" else DATA_SEED + 1

    def gen():
        g = torch.Generator().manual_seed(seed)
        for _ in range(STEPS_PER_EPOCH):
            toks = torch.randint(0, VOCAB, (BATCH, SEQ + 1), generator=g)
            yield toks[:, :-1], toks[:, 1:]

    return gen()


def loss_fn(model, batch):
    dev = next(model.parameters()).device
    x, y = (t.to(dev, non_blocking=True) for t in batch)
    logits = model(x)
    return F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1))
