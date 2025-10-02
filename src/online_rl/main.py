"""
Complete Online RL Training for Pokemon VLM
No pre-collected dataset - generates actions and learns in real-time
"""

from rewards import *
from trainer import OnlinePokemonRLTrainer
from config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_DEBUG_PORT,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_LEARNING_RATE,
    DEFAULT_GAMMA,
    DEFAULT_ENTROPY_COEF,
    DEFAULT_NUM_EPISODES,
    DEFAULT_MAX_STEPS_PER_EPISODE,
    DEFAULT_SAVE_EVERY,
    DEFAULT_TEMPERATURE,
    DEFAULT_IMAGE_RESOLUTION,
    DEFAULT_LOG_INPUTS,
    DEFAULT_MAX_TURNS,
)

def main():
    """
    Train Pokemon VLM with online RL

    Make sure Chrome is running with:
    google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
    """
    trainer = OnlinePokemonRLTrainer(
        model_name=DEFAULT_MODEL_NAME,
        debug_port=DEFAULT_DEBUG_PORT,
        output_dir=DEFAULT_OUTPUT_DIR,
        learning_rate=DEFAULT_LEARNING_RATE,
        gamma=DEFAULT_GAMMA,
        entropy_coef=DEFAULT_ENTROPY_COEF,
        reward_functions=[reward_ocr],
        image_resolution=DEFAULT_IMAGE_RESOLUTION,
        log_inputs=DEFAULT_LOG_INPUTS,
        max_turns=DEFAULT_MAX_TURNS,
    )

    trainer.train(
        num_episodes=DEFAULT_NUM_EPISODES,
        max_steps_per_episode=DEFAULT_MAX_STEPS_PER_EPISODE,
        save_every=DEFAULT_SAVE_EVERY,
        temperature=DEFAULT_TEMPERATURE,
    )


if __name__ == "__main__":
    main()
