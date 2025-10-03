"""
Unit tests for OnlinePokemonRLTrainer
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
from PIL import Image
from trainer import OnlinePokemonRLTrainer
from rewards import reward_ocr


def create_test_image(value):
    """Create a test image filled with a specific value"""
    # 10x10 RGB image with all pixels set to [value, value, value]
    return np.full((10, 10, 3), value, dtype=np.uint8)


def mock_step(action, prev_obs, duration_ms=100):
    """Mock environment step that returns different images"""
    # Extract the unique value from prev_obs to determine next image
    prev_value = prev_obs[0, 0, 0]
    next_value = prev_value + 1
    next_obs = create_test_image(next_value)
    reward = 0.0
    done = False
    info = {'action': action}
    return next_obs, reward, done, info


@pytest.fixture
def mock_trainer():
    """Create a trainer with mocked dependencies"""
    import torch

    with patch('trainer.PokemonBrowserEnv'), \
         patch('trainer.AutoProcessor'), \
         patch('trainer.AutoModelForImageTextToText') as mock_model_class:

        # Mock the model to have fake parameters so optimizer doesn't fail
        mock_model = Mock()
        fake_param = torch.nn.Parameter(torch.zeros(1))
        mock_model.parameters.return_value = [fake_param]
        mock_model.device = torch.device('cpu')
        mock_model_class.from_pretrained.return_value = mock_model

        trainer = OnlinePokemonRLTrainer(
            model_name="test_model",
            max_turns=3,
        )

        # Mock the environment
        trainer.env.reset = Mock(return_value=create_test_image(0))
        trainer.env.step = Mock(side_effect=mock_step)
        trainer.env.action_space = ['up', 'down']

        # Mock the processor
        trainer.processor = Mock()

        # Track the history passed to generate_action_with_logprobs
        trainer._history_log = []

        # Setup action tokens
        trainer.action_token_ids = [1, 2]
        trainer.token_id_to_action = {1: 'up', 2: 'down'}

        yield trainer


def test_history_contains_different_images(mock_trainer):
    """Test that history accumulates different images, not the same image repeated"""

    def mock_generate_action(history, action_history, temperature=1.0):
        # Store history for inspection
        mock_trainer._history_log.append([img.copy() for img in history])

        # Return mock values
        import torch
        log_prob = torch.tensor(0.0, requires_grad=True)
        entropy = torch.tensor(0.0)
        num_tokens = 100
        return 'up', log_prob, entropy, num_tokens

    # Mock generate_action_with_logprobs
    mock_trainer.generate_action_with_logprobs = Mock(side_effect=mock_generate_action)

    # Mock log inputs to avoid file I/O
    mock_trainer._log_episode_inputs = Mock()

    # Run a 3-step episode
    with patch('trainer.compute_returns', return_value=[0.0, 0.0, 0.0]), \
         patch('trainer.normalize_returns', return_value=np.array([0.0, 0.0, 0.0])):
        mock_trainer.train_episode(max_steps=3, episode_num=0)

    # Verify we collected history 3 times (one per step)
    assert len(mock_trainer._history_log) == 3, "Should have 3 history snapshots"

    # Step 1: history should contain 1 image (value=0)
    step1_history = mock_trainer._history_log[0]
    assert len(step1_history) == 1, "Step 1 should have 1 image in history"
    assert step1_history[0][0, 0, 0] == 0, "Step 1 image should have value 0"

    # Step 2: history should contain 2 different images (values=0,1)
    step2_history = mock_trainer._history_log[1]
    assert len(step2_history) == 2, "Step 2 should have 2 images in history"
    assert step2_history[0][0, 0, 0] == 0, "Step 2 first image should have value 0"
    assert step2_history[1][0, 0, 0] == 1, "Step 2 second image should have value 1"

    # Step 3: history should contain 3 different images (values=0,1,2)
    step3_history = mock_trainer._history_log[2]
    assert len(step3_history) == 3, "Step 3 should have 3 images in history"
    assert step3_history[0][0, 0, 0] == 0, "Step 3 first image should have value 0"
    assert step3_history[1][0, 0, 0] == 1, "Step 3 second image should have value 1"
    assert step3_history[2][0, 0, 0] == 2, "Step 3 third image should have value 2"

    # Verify images are different from each other
    for i, history in enumerate(mock_trainer._history_log):
        unique_values = set(img[0, 0, 0] for img in history)
        assert len(unique_values) == len(history), \
            f"Step {i+1}: All images in history should be different"


def test_history_sliding_window(mock_trainer):
    """Test that history respects max_turns sliding window"""

    def mock_generate_action(history, action_history, temperature=1.0):
        mock_trainer._history_log.append([img.copy() for img in history])
        import torch
        log_prob = torch.tensor(0.0, requires_grad=True)
        entropy = torch.tensor(0.0)
        num_tokens = 100
        return 'up', log_prob, entropy, num_tokens

    mock_trainer.generate_action_with_logprobs = Mock(side_effect=mock_generate_action)
    mock_trainer._log_episode_inputs = Mock()

    # Run a 5-step episode with max_turns=3
    with patch('trainer.compute_returns', return_value=[0.0]*5), \
         patch('trainer.normalize_returns', return_value=np.array([0.0]*5)):
        mock_trainer.train_episode(max_steps=5, episode_num=0)

    # Verify sliding window behavior
    assert len(mock_trainer._history_log[0]) == 1, "Step 1: 1 image"
    assert len(mock_trainer._history_log[1]) == 2, "Step 2: 2 images"
    assert len(mock_trainer._history_log[2]) == 3, "Step 3: 3 images (max)"
    assert len(mock_trainer._history_log[3]) == 3, "Step 4: 3 images (sliding window)"
    assert len(mock_trainer._history_log[4]) == 3, "Step 5: 3 images (sliding window)"

    # Verify step 4 has values [1, 2, 3] (oldest dropped)
    step4_values = [img[0, 0, 0] for img in mock_trainer._history_log[3]]
    assert step4_values == [1, 2, 3], "Step 4 should have values [1, 2, 3]"

    # Verify step 5 has values [2, 3, 4] (oldest dropped)
    step5_values = [img[0, 0, 0] for img in mock_trainer._history_log[4]]
    assert step5_values == [2, 3, 4], "Step 5 should have values [2, 3, 4]"


def test_action_history_in_prompt(mock_trainer):
    """Test that action history is properly included in prompts"""

    # Track both image and action history passed to generate_action_with_logprobs
    mock_trainer._action_history_log = []

    def mock_generate_action(history, action_history, temperature=1.0):
        # Store both image and action history for inspection
        mock_trainer._history_log.append([img.copy() for img in history])
        mock_trainer._action_history_log.append(action_history.copy())

        # Alternate between 'up' and 'down' actions
        action = 'up' if len(action_history) % 2 == 0 else 'down'

        import torch
        log_prob = torch.tensor(0.0, requires_grad=True)
        entropy = torch.tensor(0.0)
        num_tokens = 100
        return action, log_prob, entropy, num_tokens

    mock_trainer.generate_action_with_logprobs = Mock(side_effect=mock_generate_action)
    mock_trainer._log_episode_inputs = Mock()

    # Run a 3-step episode
    with patch('trainer.compute_returns', return_value=[0.0]*3), \
         patch('trainer.normalize_returns', return_value=np.array([0.0]*3)):
        mock_trainer.train_episode(max_steps=3, episode_num=0)

    # Verify action history accumulates correctly
    assert len(mock_trainer._action_history_log) == 3, "Should have 3 action history snapshots"

    # Step 1: No previous actions
    step1_actions = mock_trainer._action_history_log[0]
    assert step1_actions == [], "Step 1 should have no action history"

    # Step 2: Should have action from step 1
    step2_actions = mock_trainer._action_history_log[1]
    assert step2_actions == ['up'], "Step 2 should have action history ['up']"

    # Step 3: Should have actions from steps 1 and 2
    step3_actions = mock_trainer._action_history_log[2]
    assert step3_actions == ['up', 'down'], "Step 3 should have action history ['up', 'down']"


def test_action_history_in_prompt_format(mock_trainer):
    """Test that prompts are correctly formatted with action history"""

    # Test the prepare_prompt method directly
    mock_trainer.env.action_space = ['up', 'down', 'left', 'right']

    # Test with no action history (first step)
    prompt1 = mock_trainer.prepare_prompt(num_images=1, action_history=[])
    expected1 = "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction:"
    assert prompt1 == expected1, f"Prompt 1 mismatch:\nGot: {prompt1}\nExpected: {expected1}"

    # Test with one action in history (second step)
    prompt2 = mock_trainer.prepare_prompt(num_images=2, action_history=['up'])
    expected2 = (
        "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction: up\n"
        "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction:"
    )
    assert prompt2 == expected2, f"Prompt 2 mismatch:\nGot: {prompt2}\nExpected: {expected2}"

    # Test with two actions in history (third step)
    prompt3 = mock_trainer.prepare_prompt(num_images=3, action_history=['up', 'down'])
    expected3 = (
        "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction: up\n"
        "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction: down\n"
        "<image>\nSelect one of the following possible actions: ['up', 'down', 'left', 'right']\nAction:"
    )
    assert prompt3 == expected3, f"Prompt 3 mismatch:\nGot: {prompt3}\nExpected: {expected3}"


def test_action_history_sliding_window(mock_trainer):
    """Test that action history respects max_turns sliding window"""

    mock_trainer._action_history_log = []

    def mock_generate_action(history, action_history, temperature=1.0):
        mock_trainer._action_history_log.append(action_history.copy())
        # Return sequential actions
        actions = ['up', 'down', 'left', 'right', 'a']
        action = actions[len(action_history) % len(actions)]
        import torch
        log_prob = torch.tensor(0.0, requires_grad=True)
        entropy = torch.tensor(0.0)
        num_tokens = 100
        return action, log_prob, entropy, num_tokens

    mock_trainer.generate_action_with_logprobs = Mock(side_effect=mock_generate_action)
    mock_trainer._log_episode_inputs = Mock()

    # Run a 5-step episode with max_turns=3
    with patch('trainer.compute_returns', return_value=[0.0]*5), \
         patch('trainer.normalize_returns', return_value=np.array([0.0]*5)):
        mock_trainer.train_episode(max_steps=5, episode_num=0)

    # Verify action history respects sliding window
    # Step 1: no actions (len=0, 0%5=0, action='up')
    assert mock_trainer._action_history_log[0] == []
    # Step 2: 1 action (len=1, 1%5=1, action='down')
    assert mock_trainer._action_history_log[1] == ['up']
    # Step 3: 2 actions (len=2, 2%5=2, action='left')
    assert mock_trainer._action_history_log[2] == ['up', 'down']
    # Step 4: 2 actions (sliding window kicks in, oldest dropped) (len=2, 2%5=2, action='left')
    assert mock_trainer._action_history_log[3] == ['down', 'left']
    # Step 5: 2 actions (sliding window continues) (len=2, 2%5=2, action='left')
    assert mock_trainer._action_history_log[4] == ['left', 'left']


def test_reward_ocr_battle_double():
    """Test that reward_ocr detects number increment in battle_double.png"""
    from rewards import ocr_top_right_has_number, _run_ocr

    # Load the test image
    image_path = '/home/me/Repos/autonomous-pokerogue/assets/gamestates/battle_double.png'
    img = Image.open(image_path)

    # Test that OCR subfunction detects the number
    ocr_result = _run_ocr(img)
    has_number_reward = ocr_top_right_has_number(ocr_result)
    assert has_number_reward > 0.0, f"Expected positive reward for detecting number in top corner, got {has_number_reward}"

    # Test that reward_ocr returns 0 when same frame (no increment)
    mock_env = Mock()
    reward_same = reward_ocr(
        prev_obs=img,
        next_obs=img,
        action='test',
        env=mock_env
    )
    assert reward_same == 0.0, f"Expected 0 reward for same frame (no increment), got {reward_same}"


def test_reward_ocr_increment():
    """Test that reward_ocr gives large reward when top-right number increments"""
    from rewards import _run_ocr, ocr_top_right_number_increments
    from PIL import Image, ImageDraw, ImageFont

    # Create two test images with different numbers in top-right corner
    def create_image_with_number(number):
        img = Image.new('RGB', (100, 100), color='white')
        draw = ImageDraw.Draw(img)
        # Draw number in top-right corner
        draw.text((70, 10), str(number), fill='black')
        return img

    img_4 = create_image_with_number(4)
    img_5 = create_image_with_number(5)

    # Mock environment
    mock_env = Mock()

    # Test increment from 4 to 5
    reward = reward_ocr(
        prev_obs=img_4,
        next_obs=img_5,
        action='test',
        env=mock_env
    )

    # Should give large reward for increment
    assert reward > 5.0, f"Expected large reward (>5.0) for number increment 4->5, got {reward}"
