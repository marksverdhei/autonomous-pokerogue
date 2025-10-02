"""
Reward functions for Pokemon RL training
"""

from PIL import Image
from typing import TYPE_CHECKING
import numpy as np
from paddleocr import PaddleOCR

if TYPE_CHECKING:
    from environment import PokemonBrowserEnv

# Global OCR instance (initialized lazily)
_ocr_instance = None


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
    For testing: rewards if the top rightmost text is a number.

    Args:
        prev_obs: Screenshot before action (PIL Image)
        next_obs: Screenshot after action (PIL Image)
        action: Action taken
        env: Environment instance
        **kwargs: Additional context (for future extensibility)

    Returns:
        Reward value
    """
    # Use global OCR instance (lazy initialization)
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)

    ocr = _ocr_instance

    # Convert PIL Image to RGB (in case it's RGBA) then to numpy array for PaddleOCR
    if next_obs.mode != 'RGB':
        next_obs = next_obs.convert('RGB')
    img_array = np.array(next_obs)

    # Run OCR
    result = ocr.predict(input=img_array)
    # print(result)

    # Check if any text was detected
    if not result:
        print("[reward_ocr] No OCR results")
        return 0.0

    print(f"[reward_ocr] Got {len(result)} result objects")

    # Find the top rightmost text box
    # Result contains detection objects with bbox and text attributes
    top_rightmost = None
    max_x = -float('inf')
    min_y = float('inf')

    for i, res in enumerate(result):
        # Access as dictionary
        if 'dt_polys' in res and 'rec_texts' in res:
            boxes = res['dt_polys']
            texts = res['rec_texts']
            scores = res.get('rec_scores', [1.0] * len(texts))

            print(f"[reward_ocr] Found {len(texts)} text detections")
            for j, (bbox, text, score) in enumerate(zip(boxes, texts, scores)):
                print(f"[reward_ocr] Detection {j}: text='{text}', score={score}, bbox={bbox}")
                # bbox is a numpy array of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                # Get the rightmost x coordinate (top-right corner)
                x_right = bbox[1][0]  # x coordinate of top-right corner
                y_top = bbox[0][1]    # y coordinate of top-left corner

                # Find top-rightmost: prioritize top (min y), then right (max x)
                if y_top < min_y or (y_top == min_y and x_right > max_x):
                    min_y = y_top
                    max_x = x_right
                    top_rightmost = (text, score)

    # Check if the top rightmost text is a number or contains a number
    if top_rightmost:
        print("found_text", top_rightmost)
        text, confidence = top_rightmost
        # Check if text is a number (allows decimals)
        try:
            float(text.replace(',', ''))  # Remove commas for numbers like 1,000
            return 1.0  # Reward if it's a number
        except ValueError:
            # Try to extract numeric part if it contains a number
            import re
            # Find any standalone number in the text
            numbers = re.findall(r'\b\d+\.?\d*\b', text)
            if numbers:
                print(f"[reward_ocr] Found number '{numbers[0]}' in text '{text}'")
                return 1.0
            return 0.0  # No reward if not a number

    return 0.0

