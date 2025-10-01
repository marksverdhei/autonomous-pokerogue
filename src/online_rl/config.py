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
DEFAULT_IMAGE_RESOLUTION = None  # tuple[int, int] | None - (width, height) to resize images
DEFAULT_IMAGE_RESOLUTION = (640, 360)  # tuple[int, int] | None - (width, height) to resize images
# DEFAULT_IMAGE_RESOLUTION = (512, 512)  # tuple[int, int] | None - (width, height) to resize images
# DEFAULT_IMAGE_RESOLUTION = (128, 128)  # tuple[int, int] | None - (width, height) to resize images
# DEFAULT_IMAGE_RESOLUTION = (256, 128)  # tuple[int, int] | None - (width, height) to resize images
DEFAULT_LOG_INPUTS = True  # bool - log images and text inputs during training
# DEFAULT_MAX_TURNS = 2  # int | None - max history turns to pass to model (None = unlimited)
DEFAULT_MAX_TURNS = None  # int | None - max history turns to pass to model (None = unlimited)

# Training hyperparameters
DEFAULT_MODEL_NAME = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
DEFAULT_DEBUG_PORT = 9222
DEFAULT_OUTPUT_DIR = "./pokemon_online_rl"
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_GAMMA = 0.99
DEFAULT_ENTROPY_COEF = 0.01
DEFAULT_TEMPERATURE = 1.5
DEFAULT_NUM_EPISODES = 100
DEFAULT_MAX_STEPS_PER_EPISODE = 3
DEFAULT_SAVE_EVERY = 10
