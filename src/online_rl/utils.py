"""
Utility functions for Online RL training
"""

import torch
from typing import List


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns

    Args:
        rewards: List of rewards
        gamma: Discount factor

    Returns:
        List of discounted returns
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def normalize_returns(returns: torch.Tensor) -> torch.Tensor:
    """
    Normalize returns for training stability

    Args:
        returns: Tensor of returns

    Returns:
        Normalized returns
    """
    if len(returns) > 1:
        return (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def extract_action(text: str, action_space: List[str]) -> str:
    """
    Extract action from generated text

    Args:
        text: Generated text from model
        action_space: List of valid actions

    Returns:
        Extracted action or 'noop' if not found
    """
    text = text.lower().strip()
    for action in action_space:
        if action in text:
            return action
    return 'noop'
