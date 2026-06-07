"""DAD-Net: a distribution-aligned dual-stream framework for micro-expression
recognition.

The package exposes the MicroFlowNeXt backbone, the dual-stream DAD-Net model,
the GSWD alignment objective, and the data and training infrastructure used in
the paper.
"""

from .builder import build_dadnet, build_student
from .configs import BENCHMARKS, TRAINING, get_benchmark
from .losses import gaussian_sliced_wasserstein_distance, get_kd_loss
from .models import DADNet, MicroFlowNeXt, get_dad_net, get_microflownext

__version__ = "1.0.0"

__all__ = [
    "MicroFlowNeXt",
    "DADNet",
    "get_microflownext",
    "get_dad_net",
    "build_dadnet",
    "build_student",
    "gaussian_sliced_wasserstein_distance",
    "get_kd_loss",
    "BENCHMARKS",
    "TRAINING",
    "get_benchmark",
    "__version__",
]
