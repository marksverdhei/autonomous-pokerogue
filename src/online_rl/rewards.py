"""
Reward functions for Pokemon RL training
"""

from PIL import Image
from typing import TYPE_CHECKING, Optional, List, Tuple
import numpy as np
from paddleocr import PaddleOCR
import re

if TYPE_CHECKING:
    from environment import PokemonBrowserEnv

# Global OCR instance (initialized lazily)
_ocr_instance = None


def _get_ocr_instance():
    """Get or create the global OCR instance"""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)
    return _ocr_instance


def _run_ocr(img: Image.Image) -> dict:
    """
    Run OCR on an image and return the result

    Args:
        img: PIL Image

    Returns:
        OCR result dictionary
    """
    ocr = _get_ocr_instance()

    # Convert PIL Image to RGB (in case it's RGBA) then to numpy array for PaddleOCR
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img)

    # Run OCR
    result = ocr.predict(input=img_array)

    if not result:
        return {}

    return result[0] if len(result) > 0 else {}


def _extract_top_rightmost_number(ocr_result: dict) -> Optional[float]:
    """
    Extract the top-rightmost number from OCR results

    Args:
        ocr_result: OCR result dictionary

    Returns:
        The number found in the top-rightmost text, or None if not found
    """
    if not ocr_result or 'dt_polys' not in ocr_result or 'rec_texts' not in ocr_result:
        return None

    boxes = ocr_result['dt_polys']
    texts = ocr_result['rec_texts']

    if not boxes or not texts:
        return None

    # Find the top rightmost text box
    top_rightmost = None
    max_x = -float('inf')
    min_y = float('inf')

    for bbox, text in zip(boxes, texts):
        # Get the rightmost x coordinate (top-right corner)
        x_right = bbox[1][0]  # x coordinate of top-right corner
        y_top = bbox[0][1]    # y coordinate of top-left corner

        # Find top-rightmost: prioritize top (min y), then right (max x)
        if y_top < min_y or (y_top == min_y and x_right > max_x):
            min_y = y_top
            max_x = x_right
            top_rightmost = text

    if not top_rightmost:
        return None

    # Try to parse as number directly
    try:
        return float(top_rightmost.replace(',', ''))
    except ValueError:
        # Try to extract numeric part if it contains a number
        numbers = re.findall(r'\b\d+\.?\d*\b', top_rightmost)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None

    return None


def ocr_top_right_has_number(ocr_result: dict) -> float:
    """
    OCR subfunction: Check if top-right corner has a number

    Args:
        ocr_result: OCR result dictionary

    Returns:
        1.0 if number found, 0.0 otherwise
    """
    number = _extract_top_rightmost_number(ocr_result)
    return 1.0 if number is not None else 0.0


def ocr_top_right_number_increments(prev_ocr_result: dict, next_ocr_result: dict) -> float:
    """
    OCR subfunction: Check if top-right number increments from prev to next frame

    Args:
        prev_ocr_result: OCR result from previous frame
        next_ocr_result: OCR result from next frame

    Returns:
        10.0 if number incremented, 0.0 otherwise
    """
    prev_number = _extract_top_rightmost_number(prev_ocr_result)
    next_number = _extract_top_rightmost_number(next_ocr_result)

    print(f"[ocr_increment] prev_number={prev_number}, next_number={next_number}")

    if prev_number is not None and next_number is not None:
        if next_number > prev_number:
            increment = next_number - prev_number
            print(f"[ocr_increment] Number incremented by {increment}! Giving reward.")
            return 10.0

    return 0.0


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


def reward_ocr(
    prev_obs: Image.Image,
    next_obs: Image.Image,
    action: str,
    env: 'PokemonBrowserEnv',
    **kwargs
) -> float:
    """
    Reward function that uses PaddleOCR to detect text.
    Rewards if top-right number increments between frames.

    Args:
        prev_obs: Screenshot before action (PIL Image)
        next_obs: Screenshot after action (PIL Image)
        action: Action taken
        env: Environment instance
        **kwargs: Additional context (for future extensibility)

    Returns:
        Reward value
    """
    # Run OCR on both frames
    prev_ocr_result = _run_ocr(prev_obs)
    next_ocr_result = _run_ocr(next_obs)

    # Check if top-right number increments (large reward)
    reward = ocr_top_right_number_increments(prev_ocr_result, next_ocr_result)

    return reward

