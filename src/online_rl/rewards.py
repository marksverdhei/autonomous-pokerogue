"""
Reward functions for Pokemon RL training
"""

from PIL import Image
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from environment import PokemonBrowserEnv


def spam_start(
    prev_obs: Image.Image,
    next_obs: Image.Image,
    action: str,
    env: 'PokemonBrowserEnv',
    **kwargs
) -> float:
    """
    Reward function that encourages pressing the 'start' button

    Args:
        prev_obs: Screenshot before action (PIL Image)
        next_obs: Screenshot after action (PIL Image)
        action: Action taken
        env: Environment instance
        **kwargs: Additional context (for future extensibility)

    Returns:
        Reward value
    """
    # if action == 'start':
    #     return 1.0
    if action == 'a':
        return 1.0
    return -0.1


def max_contrast(
    prev_obs: Image.Image,
    next_obs: Image.Image,
    action: str,
    env: 'PokemonBrowserEnv',
    **kwargs
) -> float:
    """
    Reward function to change the image the most.

    Args:
        prev_obs: Screenshot before action (PIL Image)
        next_obs: Screenshot after action (PIL Image)
        action: Action taken
        env: Environment instance
        **kwargs: Additional context (for future extensibility)

    Returns:
        Reward value
    """
    size = (128,128)
    prev_obs = prev_obs.resize(size)
    next_obs = next_obs.resize(size)
    # Flatten to 1D float arrays
    v1 = np.asarray(prev_obs, dtype=np.float32).flatten()
    v2 = np.asarray(next_obs, dtype=np.float32).flatten()

    # Normalize
    v1 = v1 / (np.linalg.norm(v1) + 1e-12)
    v2 = v2 / (np.linalg.norm(v2) + 1e-12)

    # Cosine similarity = dot product of normalized vectors
    return 1 - float(np.dot(v1, v2))

