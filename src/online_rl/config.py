"""
Configuration and constants for Pokemon Online RL
"""

# Action space mapping
ACTION_SPACE = {
    'up': 'ArrowUp',
    'down': 'ArrowDown',
    'left': 'ArrowLeft',
    'right': 'ArrowRight',
    'a': 'z',
    'b': 'x',
    'start': 'Escape',
}

# Environment settings
CONSTRAIN_ACTIONS = True
START_FROM_ACTIVE_SESSION = True

# Training hyperparameters
DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_DEBUG_PORT = 9222
DEFAULT_OUTPUT_DIR = "./pokemon_online_rl"
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_GAMMA = 0.99
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_TEMPERATURE = 1.5
DEFAULT_NUM_EPISODES = 100
DEFAULT_MAX_STEPS_PER_EPISODE = 2
DEFAULT_SAVE_EVERY = 10
