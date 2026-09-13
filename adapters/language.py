"""~21M-parameter byte-level decoder on a pinned TinyStories subset."""
import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from adapters.common import Block


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(257,384)
        self.position = nn.Parameter(torch.randn(1,512,384)*.02)
        self.blocks = nn.Sequential(*[Block(384,6,causal=True) for _ in range(12)])
        self.norm = nn.LayerNorm(384)
        self.head = nn.Linear(384,257,bias=False)
        self.head.weight = self.embedding.weight
        # The tied head must not inherit Embedding's unit-variance default:
        # it makes the initial logits saturate around the current input token.
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module,(nn.Linear,nn.Embedding)):
            nn.init.normal_(module.weight,mean=0.,std=.02)
        if isinstance(module,nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self,x):
        return self.head(self.norm(self.blocks(self.embedding(x)+self.position[:, :x.shape[1]])))


def build_model():
    return Decoder()


def get_dataloader(split):
    if split not in {'train','val'}:
        raise ValueError(split)
    path = Path(os.environ.get('KE_DATA_DIR','data'))/'tinystories-f54c09f'
    path.mkdir(parents=True, exist_ok=True)
    target = path/f'{split}.npy'
    if not target.exists():
        from datasets import load_dataset
        ds = load_dataset('roneneldan/TinyStories', revision='f54c09f', split='train' if split=='train' else 'validation', streaming=True)
        tokens = []
        for i, row in enumerate(ds):
            tokens.extend(row['text'].encode('utf-8'))
            tokens.append(256)
            if i >= (4095 if split=='train' else 255):
                break
        values = np.asarray(tokens, dtype=np.uint16)
        np.save(target, values)
        target.with_suffix('.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest())
    assert hashlib.sha256(target.read_bytes()).hexdigest() == target.with_suffix('.sha256').read_text()
    values = np.load(target, allow_pickle=False)
    blocks = torch.from_numpy(values[:len(values)//513*513].astype(np.int64).reshape(-1,513))
    return DataLoader(TensorDataset(blocks[:,:512].contiguous(), blocks[:,1:].contiguous()),
                      batch_size=int(os.environ.get('KE_BATCH_SIZE',16)), shuffle=False, drop_last=True)


def loss_fn(model,batch):
    logits = model(batch[0])
    return F.cross_entropy(logits.flatten(0,1).float(), batch[1].flatten())
