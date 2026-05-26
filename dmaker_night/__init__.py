"""D-MAKER: Distributed Evidential Reasoning for Interpretable Collaborative Inference."""

__version__ = "1.0.0"

from .core import DMaker, DMakerFlood, centralized_maker
from .baselines import (
    CentralizedMAKER,
    DempsterConsensus,
    DistributedGradientTracking,
    DistributedBayesianFilter,
    MajorityVoting,
)
from .data import SyntheticDataset, UCIGasDataset, UAVSwarmDataset, Trial
from .airsim_uav import AirSimUAVDataset
from . import metrics
from . import utils

__all__ = [
    "DMaker",
    "DMakerFlood",
    "centralized_maker",
    "CentralizedMAKER",
    "DempsterConsensus",
    "DistributedGradientTracking",
    "DistributedBayesianFilter",
    "MajorityVoting",
    "SyntheticDataset",
    "UCIGasDataset",
    "UAVSwarmDataset",
    "AirSimUAVDataset",
    "Trial",
    "metrics",
    "utils",
]
