"""
Fast SAC is a high-performance implementation of Soft Actor-Critic (SAC)
for reinforcement learning.
"""

# Core model components
from fast_sac.fast_sac import Actor, Critic
from fast_sac.fast_sac_utils import EmpiricalNormalization, SimpleReplayBuffer

__all__ = [
    # Core model components
    "Actor",
    "Critic",
    "EmpiricalNormalization",
    "SimpleReplayBuffer",
]
