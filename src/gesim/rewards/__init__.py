"""Reward clients for scoring generated rollouts.

This build ships only the ``RewardClient`` protocol and ``RewardResult`` type;
no concrete reward model is bundled. Pass any object implementing
``RewardClient`` to ``WorldModelEnv(reward=...)`` to attach your own per-frame
scoring. See docs/adding_rewards.md.
"""

from gesim.rewards.base import RewardClient, RewardResult

__all__ = ["RewardClient", "RewardResult"]
