import pydoc
import warnings
from os.path import join
from torch import nn
import torch

import dynamic_network_architectures

from difference_weighting.utils import recursive_find_python_class
from difference_weighting.building_blocks.difference_weighting_block import DifferenceWeightingBlock


class LongiUNetDiffWeighting(nn.Module):
    def __init__(self, input_channels, num_classes, backbone_class_name, **architecture_kwargs):
        super().__init__()
        backbone_class = pydoc.locate(backbone_class_name)
        # sometimes things move around, this makes it so that we can at least recover some of that
        if backbone_class is None:
            warnings.warn(f'Network class {backbone_class_name} not found. Attempting to locate it within '
                        f'dynamic_network_architectures.architectures...')
            backbone_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                                backbone_class_name.split(".")[-1],
                                                'dynamic_network_architectures.architectures')
            if backbone_class is not None:
                print(f'FOUND IT: {backbone_class}')
            else:
                raise ImportError('Network class could not be found, please check/correct your plans file')

        # basic channel concatenation
        self.backbone = backbone_class(
            input_channels=input_channels,
            num_classes=num_classes,
            **architecture_kwargs
        )

        self.skip_diff_weighting = DifferenceWeightingBlock(architecture_kwargs['features_per_stage'], architecture_kwargs['conv_op'])

    def forward(self, d_c, d_p=None):
            # allow for concatenation at different points in the code
            if d_p is None:
                d_c, d_p = torch.tensor_split(d_c, 2, dim=1)
            skips_current = self.backbone.encoder(d_c)
            skips_prior = self.backbone.encoder(d_p)
            # CHANGED: Unpack the tuple returned by our modified block
            skips, attention_maps = self.skip_diff_weighting(skips_current, skips_prior)
            x = self.backbone.decoder(skips)
            return x, attention_maps
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if hasattr(self.backbone, name):
                return getattr(self.backbone, name)
            raise

    def __setattr__(self, name, value):
        if name != 'backbone' and hasattr(self, 'backbone') and hasattr(self.backbone, name):
            setattr(self.backbone, name, value)
        else:
            super().__setattr__(name, value)
