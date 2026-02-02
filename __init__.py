# -*- coding: utf-8 -*-
"""
DAD-Net: Distribution-Aligned Dual-Stream Framework for Cross-Domain Micro-Expression Recognition

This package provides the implementation of DAD-Net, a dual-branch deep learning
framework for micro-expression recognition that synergizes cross-domain knowledge
transfer from macro-expressions with motion-aware learning.

Example:
    >>> import dad_net
    >>> model = dad_net.get_dad_net(num_classes=3)
    >>> # or
    >>> from dad_net.models import MicroFlowNeXt, DADNet

Reference:
    Zhang, L., & Ben, X. (2025). DAD-Net: A Distribution-Aligned Dual-Stream Framework
    for Cross-Domain Micro-Expression Recognition.
"""

__version__ = "1.0.0"
__author__ = "Lan Zhang, Xianye Ben"

# Handle imports for both installed package and direct usage
try:
    from .models import (
        MicroFlowNeXt,
        DADNet,
        get_microflownext,
        get_dad_net,
    )
    from .losses import (
        GaussianSlicedWassersteinLoss,
        gaussian_sliced_wasserstein_distance,
        DADNetLoss,
    )
except ImportError:
    from models import (
        MicroFlowNeXt,
        DADNet,
        get_microflownext,
        get_dad_net,
    )
    from losses import (
        GaussianSlicedWassersteinLoss,
        gaussian_sliced_wasserstein_distance,
        DADNetLoss,
    )

__all__ = [
    # Models
    'MicroFlowNeXt',
    'DADNet',
    'get_microflownext',
    'get_dad_net',
    # Losses
    'GaussianSlicedWassersteinLoss',
    'gaussian_sliced_wasserstein_distance',
    'DADNetLoss',
    # Version
    '__version__',
]
