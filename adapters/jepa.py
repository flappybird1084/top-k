"""JEPA demo adapter (spec §3.2): ViT-based at 64px, context encoder + EMA target
encoder + small predictor, block masking with fixed mask sizes so shapes are constant.

Data is synthetic (fixed-seed random images) so runs need no downloads and data
order is exactly reproducible — swap `get_dataloader` for a real dataset freely,
keeping the fixed order.

Contract: build_model() / get_dataloader(split) / loss_fn(model, batch).
Optional hook the harness calls after optimizer.step(): model.post_optimizer_step().
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernelevo import ops

IMG = 64
PATCH = 8
GRID = IMG // PATCH            # 8
N_TOKENS = GRID * GRID         # 64
DIM = 256
DEPTH = 6
HEADS = 8
PRED_DIM = 128
PRED_DEPTH = 2
BATCH = 64
K_TGT = 16                     # 4 target blocks of 2x2
K_CTX = 40
EMA_MOMENTUM = 0.996
DATA_SEED = 7000
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
        a = F.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        h = ops.layer_norm(x, self.ln2_w, self.ln2_b)
        h = ops.gelu_mlp(h, self.fc1.weight, self.fc1.bias)
        return x + self.fc2(h)


class Encoder(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)
        self.pos = nn.Parameter(torch.randn(1, N_TOKENS, dim) * 0.02)
        self.blocks = nn.ModuleList(Block(dim, heads) for _ in range(depth))
        self.ln_w = nn.Parameter(torch.ones(dim))
        self.ln_b = nn.Parameter(torch.zeros(dim))

    def tokens(self, imgs):
        return self.patch(imgs).flatten(2).transpose(1, 2)  # (B, N, D)

    def forward_masked(self, imgs, idx):
        x = ops.masked_gather_add(self.tokens(imgs), idx, self.pos)
        for blk in self.blocks:
            x = blk(x)
        return ops.layer_norm(x, self.ln_w, self.ln_b)

    def forward_full(self, imgs):
        x = self.tokens(imgs) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return ops.layer_norm(x, self.ln_w, self.ln_b)


class Predictor(nn.Module):
    def __init__(self, dim, pdim, depth, heads=4):
        super().__init__()
        self.inp = nn.Linear(dim, pdim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, pdim) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, N_TOKENS, pdim) * 0.02)
        self.blocks = nn.ModuleList(Block(pdim, heads) for _ in range(depth))
        self.out = nn.Linear(pdim, dim)

    def forward(self, ctx_repr, ctx_idx, tgt_idx):
        B = ctx_repr.shape[0]
        ctx = self.inp(ctx_repr) + torch.gather(
            self.pos.expand(B, -1, -1), 1,
            ctx_idx.unsqueeze(-1).expand(-1, -1, self.pos.shape[-1]))
        tgt = self.mask_token.expand(B, tgt_idx.shape[1], -1) + torch.gather(
            self.pos.expand(B, -1, -1), 1,
            tgt_idx.unsqueeze(-1).expand(-1, -1, self.pos.shape[-1]))
        x = torch.cat([ctx, tgt], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.out(x[:, -tgt_idx.shape[1]:])


class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = Encoder(DIM, DEPTH, HEADS)
        self.target = copy.deepcopy(self.context)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(DIM, PRED_DIM, PRED_DEPTH)

    @torch.no_grad()
    def post_optimizer_step(self):
        for pt, po in zip(self.target.parameters(), self.context.parameters()):
            ops.ema_update(pt.data, po.data, EMA_MOMENTUM)


def build_model():
    return JEPA()


def _block_mask(g):
    """4 non-overlapping 2x2 target blocks on the 8x8 grid; K_CTX visible context tokens."""
    taken = torch.zeros(GRID, GRID, dtype=torch.bool)
    tgt = []
    while len(tgt) < 4:
        r = int(torch.randint(0, GRID - 1, (1,), generator=g))
        c = int(torch.randint(0, GRID - 1, (1,), generator=g))
        if taken[r:r + 2, c:c + 2].any():
            continue
        taken[r:r + 2, c:c + 2] = True
        tgt += [r * GRID + c, r * GRID + c + 1, (r + 1) * GRID + c, (r + 1) * GRID + c + 1]
    tgt = torch.tensor(tgt, dtype=torch.long)
    rest = torch.tensor([i for i in range(N_TOKENS) if not taken.flatten()[i]], dtype=torch.long)
    perm = torch.randperm(rest.numel(), generator=g)
    return rest[perm[:K_CTX]], tgt


def get_dataloader(split: str):
    seed = DATA_SEED if split == "train" else DATA_SEED + 1

    def gen():
        g = torch.Generator().manual_seed(seed)
        for _ in range(STEPS_PER_EPOCH):
            imgs = torch.randn(BATCH, 3, IMG, IMG, generator=g)
            pairs = [_block_mask(g) for _ in range(BATCH)]
            ctx = torch.stack([p[0] for p in pairs])
            tgt = torch.stack([p[1] for p in pairs])
            yield imgs, ctx, tgt

    return gen()


def loss_fn(model, batch):
    dev = next(model.parameters()).device
    imgs, ctx_idx, tgt_idx = (t.to(dev, non_blocking=True) for t in batch)
    ctx_repr = model.context.forward_masked(imgs, ctx_idx)
    with torch.no_grad():
        full = model.target.forward_full(imgs)
        tgt_repr = torch.gather(full, 1, tgt_idx.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
    pred = model.predictor(ctx_repr, ctx_idx, tgt_idx)
    return F.smooth_l1_loss(pred, tgt_repr)
