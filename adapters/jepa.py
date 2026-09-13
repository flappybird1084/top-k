"""Small JEPA-style workload, not a claim of reproducing pretrained I-JEPA."""
import copy
import os
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from adapters.common import Block
from kernel_evolution.ops import EMARegion, GatherRegion


class JEPA(nn.Module):
    def __init__(self):
        super().__init__()
        d = 192
        self.patch = nn.Conv2d(3, d, 8, stride=8)
        self.position = nn.Parameter(torch.randn(1,64,d)*.02)
        self.encoder = nn.Sequential(*[Block(d,3) for _ in range(6)], nn.LayerNorm(d))
        self.target_patch = copy.deepcopy(self.patch).requires_grad_(False)
        self.target_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.target_position = nn.Parameter(self.position.detach().clone(), requires_grad=False)
        self.predictor = nn.Sequential(Block(d,3), nn.Linear(d,d))
        self.mask_token = nn.Parameter(torch.zeros(1,16,d))
        self.gather = GatherRegion()
        self.ema = EMARegion()

    def forward(self, images):
        b = images.shape[0]
        # A fixed 4x4 target block in an 8x8 patch grid; context is the complement.
        target_idx = torch.tensor([r*8+c for r in range(2,6) for c in range(2,6)], device=images.device)
        context_idx = torch.tensor([i for i in range(64) if not (2<=i//8<6 and 2<=i%8<6)], device=images.device)
        tokens = self.patch(images).flatten(2).transpose(1,2).contiguous()
        context = self.encoder(self.gather(tokens, self.position, context_idx.expand(b,-1)))
        target_position = self.position[:, target_idx]
        predictor_in = torch.cat([context, self.mask_token.expand(b,-1,-1)+target_position],1)
        prediction = self.predictor(predictor_in)[:, -16:]
        with torch.no_grad():
            teacher = self.target_encoder(self.target_patch(images).flatten(2).transpose(1,2)+self.target_position)
            teacher = F.layer_norm(teacher[:, target_idx], (teacher.shape[-1],))
        return prediction, teacher


def build_model():
    return JEPA()


def get_dataloader(split):
    from torchvision import datasets, transforms
    if split not in {'train','val'}:
        raise ValueError(split)
    transform = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor(),
        transforms.Normalize((.4914,.4822,.4465),(.247,.243,.262))])
    ds = datasets.CIFAR10(str(Path(os.environ.get('KE_DATA_DIR','data'))/'cifar10'), train=split=='train', download=True, transform=transform)
    return DataLoader(ds, batch_size=int(os.environ.get('KE_BATCH_SIZE',16)), shuffle=False, drop_last=True, num_workers=0)


def loss_fn(model, batch):
    prediction, teacher = model(batch[0])
    return F.mse_loss(prediction.float(), teacher.float())


@torch.no_grad()
def post_optimizer_step(model):
    pairs = list(zip(model.target_encoder.parameters(), model.encoder.parameters()))
    pairs += list(zip(model.target_patch.parameters(), model.patch.parameters()))
    pairs += [(model.target_position, model.position)]
    for target, source in pairs:
        target.copy_(model.ema(target, source, .996))
