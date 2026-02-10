from typing import Union, List, Tuple, Type

from torch import nn
from torch.nn.modules.conv import _ConvNd

from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm


class DifferenceWeightingBlock(nn.Module):
    def __init__(self, features_per_stage: Union[int, List[int], Tuple[int, ...]], conv_op: Type[_ConvNd]):
        super().__init__()

        for norm in range(len(features_per_stage)):
            setattr(self, f"norm_{norm}", get_matching_instancenorm(conv_op)(features_per_stage[norm], affine=True))
    
    def forward(self, skips_current, skips_prior):
        attention_maps = []
        
        for num_skip, (skip_current, skip_prior) in enumerate(zip(skips_current, skips_prior)):
            weighting = skip_current - skip_prior

            norm = getattr(self, f"norm_{num_skip}")
            weighting = norm(weighting)

            attention.maps.append(weighting)
            
            skips_current[num_skip] = skip_current * weighting + skip_current
        return skips_current, attention_maps
