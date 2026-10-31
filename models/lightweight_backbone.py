"""RepViT backbone stages."""

import torch
from torch import nn


class BackboneStage(nn.Module):
    """One native RepViT stage; adapters branch off after each output."""

    input_channels = (3, 24, 48, 96, 192)
    output_channels = (24, 48, 96, 192, 384)
    bn_eps = 1e-5
    bn_momentum = 0.1

    def __init__(self, blocks, stage_index):
        super().__init__()
        self.blocks = blocks
        self.stage_index = stage_index
        if stage_index == 0:
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        if self.stage_index == 0:
            x = (x - self.mean.to(x)) / self.std.to(x)
        return self.blocks(x)


def build_backbone_stages(pretrained=True):
    """Create RepViT-M0.9 once; retain its five stride 2/4/8/16/32 stages."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError('RepViT requires timm>=1.0.30.') from exc
    encoder = timm.create_model('repvit_m0_9', pretrained=pretrained)


    stages = (
        nn.Sequential(encoder.stem.conv1, encoder.stem.act1),
        nn.Sequential(encoder.stem.conv2, encoder.stages[0]),
        encoder.stages[1], encoder.stages[2], encoder.stages[3],
    )
    return [BackboneStage(stage, index) for index, stage in enumerate(stages)]


def restore_backbone_batchnorm(model):
    """Restore RepViT BatchNorm settings after YOLO initialization."""
    for stage in model.modules():
        if isinstance(stage, BackboneStage):
            for module in stage.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eps = BackboneStage.bn_eps
                    module.momentum = BackboneStage.bn_momentum
