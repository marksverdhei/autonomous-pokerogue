"""
Pokemon VLM Browser Interface using Playwright (Synchronous Wrapper)
Compatible with TRL GRPO and standard RL training loops
Integrated with GRPO trainer for VLM training
"""

import asyncio
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from typing import Tuple, Optional, Dict, Any, List
from io import BytesIO
from PIL import Image
import numpy as np
import threading
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import json
import os


class PokemonBrowserEnv:
    """
    Synchronous browser environment for Pokemon VLM training
    Compatible with TRL GRPO and standard Gym-like interfaces
    """
    
    # Configuration constants
    HAS_LOGIN_SCREEN = True  # Set to False to skip login sequence
    CONSTRAIN_ACTIONS = True  # Set to False to allow model to learn invalid actions
    START_FROM_ACTIVE_SESSION = True  # Set to True to use existing browser tab without navigation
    
    def __init__(self, debug_port: int = 9222):
        """
        Initialize the browser environment
        
        Args:
            debug_port: Port number for the browser debugging endpoint
        """
        self.debug_port = debug_port
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright: Optional[Playwright] = None
        
        # Pokemon action mappings
        self.action_to_key = {
            'up': 'ArrowUp',
            'down': 'ArrowDown',
            'left': 'ArrowLeft',
            'right': 'ArrowRight',
            'a': 'z',  # Typically A button in browser Pokemon games
            'b': 'x',  # Typically B button
            'start': 'Enter',
            'select': 'Shift',
            'noop': None  # No operation
        }
        
        # Action space for RL compatibility
        self.action_space = list(self.action_to_key.keys())
        self.action_to_idx = {action: idx for idx, action in enumerate(self.action_space)}
        self.idx_to_action = {idx: action for action, idx in self.action_to_idx.items()}
    
    def connect(self, url: Optional[str] = None, username: str = "cantlogin675", password: str = "cantlogin"):
        """
        Connect to existing browser instance on debug port and login
        
        Args:
            url: Optional URL to navigate to after connecting (ignored if START_FROM_ACTIVE_SESSION is True)
            username: Username for login
            password: Password for login
        """
        import time
        
        self.playwright = sync_playwright().start()
        
        # Connect to existing browser instance
        self.browser = self.playwright.chromium.connect_over_cdp(
            f"http://localhost:{self.debug_port}"
        )
        
        # Get the first page or create new one
        contexts = self.browser.contexts
        if contexts and len(contexts[0].pages) > 0:
            self.page = contexts[0].pages[0]
        else:
            context = self.browser.new_context()
            self.page = context.new_page()
        
        # If starting from active session, skip navigation and login
        if self.START_FROM_ACTIVE_SESSION:
            print("Using active browser session (no navigation or login)")
            self.page.bring_to_front()
            return
        
        # Otherwise, navigate and potentially login
        if url:
            self.page.goto(url)
            self.page.wait_for_load_state('networkidle')
            
            # Login sequence (only if enabled)
            if self.HAS_LOGIN_SCREEN:
                time.sleep(3)  # Wait for page to fully load
                self.page.keyboard.type(username)
                self.page.keyboard.press('Tab')
                self.page.keyboard.type(password)
                self.page.keyboard.press('Enter')
                time.sleep(2)  # Wait for login to complete
        
        # Focus the page to ensure key events are captured
        self.page.bring_to_front()
    
    def reset(self, url: Optional[str] = None) -> np.ndarray:
        """
        Reset the environment (Gym-style interface)
        
        Args:
            url: Optional URL to navigate to
            
        Returns:
            Initial observation (screenshot as numpy array)
        """
        if url and self.page:
            self.page.goto(url)
            self.page.wait_for_load_state('networkidle')
        
        # Return initial observation
        return self.capture_screenshot()
    
    def step(self, action: str | int, duration_ms: int = 100) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment (Gym-style interface)
        
        Args:
            action: Action name (str) or index (int)
            duration_ms: How long to hold the key
            
        Returns:
            observation: Screenshot as numpy array
            reward: Reward signal (computed based on action)
            done: Whether episode is done
            info: Additional information dictionary
        """
        # Convert action index to action name if needed
        if isinstance(action, int):
            if action not in self.idx_to_action:
                raise ValueError(f"Invalid action index: {action}. Valid range: 0-{len(self.action_space)-1}")
            action = self.idx_to_action[action]
        
        # Execute action
        self.send_action(action, duration_ms)
        
        # Get new observation
        observation = self.capture_screenshot()
        
        # Compute reward based on action (PoC: reward for pressing 'start')
        if action == 'start':
            reward = 1.0
        else:
            reward = -0.1
        
        # Penalty for invalid actions (only if not constrained)
        if not self.CONSTRAIN_ACTIONS and not self.validate_action(action):
            reward = -1.0
        
        done = False
        info = {'action': action}
        
        return observation, reward, done, info
    
    def capture_screenshot(self, as_array: bool = True) -> np.ndarray | bytes:
        """
        Capture screenshot of the browser
        
        Args:
            as_array: If True, return numpy array; if False, return bytes
            
        Returns:
            Screenshot as numpy array (H, W, C) or bytes
        """
        if not self.page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        
        # Capture screenshot as bytes
        screenshot_bytes = self.page.screenshot(type='png')
        
        if as_array:
            # Convert to numpy array for VLM processing
            image = Image.open(BytesIO(screenshot_bytes))
            return np.array(image)
        else:
            return screenshot_bytes
    
    def capture_element_screenshot(
        self, 
        selector: str, 
        as_array: bool = True
    ) -> np.ndarray | bytes:
        """
        Capture screenshot of a specific element (useful for game canvas)
        
        Args:
            selector: CSS selector for the element (e.g., 'canvas', '#game')
            as_array: If True, return numpy array; if False, return bytes
            
        Returns:
            Screenshot as numpy array or bytes
        """
        if not self.page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        
        element = self.page.query_selector(selector)
        if not element:
            raise ValueError(f"Element with selector '{selector}' not found")
        
        screenshot_bytes = element.screenshot(type='png')
        
        if as_array:
            image = Image.open(BytesIO(screenshot_bytes))
            return np.array(image)
        else:
            return screenshot_bytes
    
    def send_action(self, action: str, duration_ms: int = 100):
        """
        Send action to the browser (translates to keyboard input)
        
        Args:
            action: Action name from action_to_key mapping
            duration_ms: How long to hold the key (in milliseconds)
        """
        if not self.page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        
        action = action.lower()
        if action not in self.action_to_key:
            raise ValueError(f"Unknown action: {action}. Valid actions: {list(self.action_to_key.keys())}")
        
        key = self.action_to_key[action]
        
        if key is None:  # noop action
            self.page.wait_for_timeout(duration_ms)
            return
        
        # Press and hold key
        self.page.keyboard.down(key)
        self.page.wait_for_timeout(duration_ms)
        self.page.keyboard.up(key)
    
    def send_key_sequence(self, actions: list[str], duration_per_key: int = 100):
        """
        Send a sequence of actions
        
        Args:
            actions: List of action names
            duration_per_key: Duration for each key press in milliseconds
        """
        for action in actions:
            self.send_action(action, duration_per_key)
    
    def get_page_state(self) -> dict:
        """
        Get additional state information from the page
        Useful for extracting game-specific state if available
        
        Returns:
            Dictionary with page state information
        """
        if not self.page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        
        return {
            'url': self.page.url,
            'title': self.page.title(),
        }
    
    def execute_js(self, script: str) -> Any:
        """
        Execute JavaScript in the browser context
        Useful for extracting game state or modifying game behavior
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Result of the JavaScript execution
        """
        if not self.page:
            raise RuntimeError("Browser not connected. Call connect() first.")
        
        return self.page.evaluate(script)
    
    def validate_action(self, action: str) -> bool:
        """
        Validate if an action is legal
        
        Args:
            action: Action name to validate
            
        Returns:
            True if action is valid, False otherwise
        """
        return action.lower() in self.action_to_key
    
    def close(self):
        """Close the browser connection"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


# ============================================================================
# GRPO Training Integration
# ============================================================================

class PokemonGRPOTrainer:
    """
    GRPO Trainer for Pokemon VLM using TRL
    """
    
    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        debug_port: int = 9222,
        game_url: str = "https://pokerogue.net",
        output_dir: str = "./pokemon_vlm_grpo",
        cache_dir: str = "./trajectory_cache",
    ):
        """
        Initialize GRPO trainer
        
        Args:
            model_name: HuggingFace model identifier
            debug_port: Browser debugging port
            game_url: URL of the Pokemon game
            output_dir: Directory to save model checkpoints
            cache_dir: Directory to cache collected trajectories
        """
        self.model_name = model_name
        self.debug_port = debug_port
        self.game_url = game_url
        self.output_dir = output_dir
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize environment
        self.env = PokemonBrowserEnv(debug_port=debug_port)
        
        # Load model and processor
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name,
                                                       # dtype=torch.bfloat16
                                                       )
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            # dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Setup action tokens for constrained decoding
        self.setup_action_tokens()
        
        # Training config
        self.config = GRPOConfig(
            output_dir=output_dir,
            learning_rate=1e-5,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            remove_unused_columns=False,
            # GRPO-specific parameters
            num_generations=4,  # Number of completions to generate per prompt
            max_prompt_length=2048,  # Max length for prompts (image tokens ~800+)
            max_completion_length=10,  # Max length for completions (just action word)
            temperature=0.7,  # Sampling temperature
            beta=0.04,  # KL penalty coefficient
            # bf16=True,
            bf16=False,
            fp16=False,
        )
    
    def setup_action_tokens(self):
        """
        Setup action token mappings for the model
        Tokenize all valid actions and store their token IDs
        """
        self.action_to_text = {
            'up': 'up',
            'down': 'down',
            'left': 'left',
            'right': 'right',
            'a': 'a',
            'b': 'b',
            'start': 'start',
            'select': 'select',
            'noop': 'noop',
        }
        
        # Tokenize actions and get their token IDs
        self.action_token_ids = []
        self.token_id_to_action = {}
        
        for action, text in self.action_to_text.items():
            tokens = self.processor.tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) > 0:
                token_id = tokens[0]  # Use first token
                self.action_token_ids.append(token_id)
                self.token_id_to_action[token_id] = action
        
        print(f"Action tokens setup: {len(self.action_token_ids)} actions")
        print(f"Token ID to action mapping: {self.token_id_to_action}")
    
    def prepare_prompt(self, image: np.ndarray) -> str:
        """
        Prepare the prompt for the VLM
        SmolVLM requires <image> token in the prompt
        
        Args:
            image: Screenshot as numpy array
            
        Returns:
            Prompt string
        """
        # Simple PoC: Train to press start
        # SmolVLM uses <image> token to indicate where the image should be processed
        prompt = (
            "<image>You are playing Pokemon. Press the start button. "
            "Available actions: up, down, left, right, a, b, start, select, noop. "
            "Respond with only one action word."
        )
        return prompt
    
    def get_action_from_model(self, image: np.ndarray) -> str:
        """
        Get action prediction from the VLM
        
        Args:
            image: Screenshot as numpy array
            
        Returns:
            Predicted action string
        """
        print("  [Model] Processing screenshot and generating action...")
        
        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(image)
        
        # Prepare inputs
        prompt = self.prepare_prompt(image)
        inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        print(f"  [Model] Input tokens: {inputs['input_ids'].shape[1]}")
        
        # Generate with constraints if enabled
        gen_kwargs = {
            "max_new_tokens": 10,
            "do_sample": True,
            "temperature": 0.7,
        }
        
        if self.env.CONSTRAIN_ACTIONS:
            print("  [Model] Using constrained generation (valid actions only)")
            # For constrained sampling, we need to use a logits processor
            # instead of force_words_ids which requires beam search
            from transformers import LogitsProcessorList
            
            class ActionConstraintLogitsProcessor:
                def __init__(self, allowed_token_ids, vocab_size):
                    self.allowed_token_ids = set(allowed_token_ids)
                    self.vocab_size = vocab_size
                
                def __call__(self, input_ids, scores):
                    # Set all non-allowed tokens to -inf
                    mask = torch.ones(scores.shape[-1], dtype=torch.bool, device=scores.device)
                    for token_id in self.allowed_token_ids:
                        if token_id < scores.shape[-1]:
                            mask[token_id] = False
                    scores[:, mask] = float('-inf')
                    return scores
            
            # Add logits processor for constrained generation
            vocab_size = self.model.config.vocab_size
            logits_processor = LogitsProcessorList([
                ActionConstraintLogitsProcessor(self.action_token_ids, vocab_size)
            ])
            gen_kwargs["logits_processor"] = logits_processor
        else:
            print("  [Model] Using unconstrained generation")
        
        # Generate
        print("  [Model] Generating...")
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        print(f"  [Model] Generated text: '{generated_text}'")
        
        # Extract action from generated text
        action = self.extract_action(generated_text)
        print(f"  [Model] Extracted action: '{action}'")
        
        return action
    
    def extract_action(self, text: str) -> str:
        """
        Extract action from generated text
        
        Args:
            text: Generated text from model
            
        Returns:
            Action string (defaults to 'noop' if invalid)
        """
        text_lower = text.lower().strip()
        
        # Check if any valid action is in the text
        for action in self.env.action_space:
            if action in text_lower:
                return action
        
        # Default to noop if no valid action found
        return 'noop'
    
    def compute_reward(self, obs: np.ndarray, action: str, info: Dict) -> float:
        """
        Compute reward for the current step
        
        Simple PoC: Reward the model for pressing 'start'
        
        Args:
            obs: Current observation
            action: Action taken
            info: Additional info from environment
            
        Returns:
            Reward value
        """
        # Simple PoC: High reward for pressing start, small penalty for anything else
        if action == 'start':
            reward = 1.0
        else:
            reward = -0.1
        
        # Penalty for invalid actions (only if not constrained)
        if not self.env.CONSTRAIN_ACTIONS and not self.env.validate_action(action):
            reward = -1.0
        
        return reward
    
    def collect_trajectories(self, num_episodes: int = 10, max_steps: int = 100) -> Dataset:
        """
        Collect trajectories for GRPO training
        
        Args:
            num_episodes: Number of episodes to collect
            max_steps: Maximum steps per episode
            
        Returns:
            Dataset containing trajectories
        """
        cache_file = os.path.join(
            self.cache_dir, 
            f"trajectories_ep{num_episodes}_steps{max_steps}.json"
        )
        
        # Check if cached trajectories exist
        if os.path.exists(cache_file):
            print(f"Loading cached trajectories from {cache_file}")
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            # Reconstruct PIL images from base64
            import base64
            from io import BytesIO
            
            dataset_dict = {
                'prompt': cached_data['prompt'],
                'image': [],
                'action': cached_data['action'],
                'reward': cached_data['reward'],
            }
            
            for img_b64 in cached_data['image']:
                img_bytes = base64.b64decode(img_b64)
                pil_image = Image.open(BytesIO(img_bytes))
                dataset_dict['image'].append(pil_image)
            
            dataset = Dataset.from_dict(dataset_dict)
            print(f"Loaded {len(dataset)} cached trajectory steps")
            return dataset
        
        # Otherwise, collect new trajectories
        print("No cached trajectories found, collecting new ones...")
        trajectories = []
        
        for episode in range(num_episodes):
            print(f"\n{'─'*60}")
            print(f"EPISODE {episode + 1}/{num_episodes}")
            print(f"{'─'*60}")
            
            obs = self.env.reset()
            episode_data = {
                'images': [],
                'actions': [],
                'rewards': [],
                'prompts': [],
            }
            
            for step in range(max_steps):
                print(f"\n[Step {step + 1}/{max_steps}]")
                
                # Get action from model
                action = self.get_action_from_model(obs)
                
                # Execute action
                print(f"  [Env] Executing action: '{action}'")
                next_obs, _, done, info = self.env.step(action)
                
                # Compute reward
                reward = self.compute_reward(next_obs, action, info)
                print(f"  [Reward] {reward:.2f}")
                
                # Store trajectory data
                episode_data['images'].append(Image.fromarray(obs))
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
                episode_data['prompts'].append(self.prepare_prompt(obs))
                
                obs = next_obs
                
                if done:
                    print(f"  [Env] Episode terminated early")
                    break
            
            trajectories.append(episode_data)
            episode_reward = sum(episode_data['rewards'])
            print(f"\n{'─'*60}")
            print(f"Episode {episode + 1} Summary:")
            print(f"  Steps: {len(episode_data['actions'])}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Reward: {episode_reward/len(episode_data['actions']):.2f}")
            print(f"{'─'*60}")
        
        # Convert to HuggingFace Dataset format
        dataset_dict = {
            'prompt': [],
            'image': [],
            'action': [],
            'reward': [],
        }
        
        for traj in trajectories:
            for i in range(len(traj['actions'])):
                dataset_dict['prompt'].append(traj['prompts'][i])
                dataset_dict['image'].append(traj['images'][i])
                dataset_dict['action'].append(traj['actions'][i])
                dataset_dict['reward'].append(traj['rewards'][i])
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Save to cache
        print(f"Saving trajectories to cache: {cache_file}")
        import base64
        from io import BytesIO
        
        cache_data = {
            'prompt': dataset_dict['prompt'],
            'image': [],
            'action': dataset_dict['action'],
            'reward': dataset_dict['reward'],
        }
        
        # Convert PIL images to base64 for JSON serialization
        for pil_image in dataset_dict['image']:
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            cache_data['image'].append(img_b64)
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        print(f"Cached {len(dataset)} trajectory steps")
        
        return dataset
    
    def train(self, num_iterations: int = 10, episodes_per_iteration: int = 10, max_steps_per_episode: int = 50):
        """
        Run GRPO training loop with episodic updates
        The model continuously interacts with the browser and improves
        
        Args:
            num_iterations: Number of training iterations (collect → train → repeat)
            episodes_per_iteration: Number of episodes to collect per iteration
            max_steps_per_episode: Maximum steps per episode
        """
        # Connect to browser
        print("Connecting to browser...")
        self.env.connect(url=self.game_url)
        print("Browser connected!")
        
        # Define reward function for GRPO
        def reward_func(samples, prompts, outputs, tokenizer, **kwargs):
            """
            Compute rewards for generated outputs
            For our PoC: reward samples that contain 'start'
            """
            rewards = []
            for output in outputs:
                # Decode the output
                text = tokenizer.decode(output, skip_special_tokens=True).lower()
                # Check if it contains 'start'
                if 'start' in text:
                    rewards.append(1.0)
                else:
                    rewards.append(-0.1)
            return rewards
        
        try:
            for iteration in range(num_iterations):
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{num_iterations}")
                print(f"{'='*60}\n")
                
                # Collect trajectories with current model
                print(f"Collecting {episodes_per_iteration} episodes...")
                dataset = self.collect_trajectories(
                    num_episodes=episodes_per_iteration,
                    max_steps=max_steps_per_episode
                )
                
                print(f"Collected {len(dataset)} trajectory steps")
                
                # Train on collected trajectories
                print(f"\nTraining on collected data...")
                trainer = GRPOTrainer(
                    model=self.model,
                    reward_funcs=[reward_func],  # Required argument
                    args=self.config,
                    train_dataset=dataset,
                    processing_class=self.processor,
                )
                
                trainer.train()
                
                # Save checkpoint after each iteration
                checkpoint_path = f"{self.output_dir}/iteration_{iteration + 1}"
                print(f"\nSaving checkpoint to {checkpoint_path}")
                self.model.save_pretrained(checkpoint_path)
                self.processor.save_pretrained(checkpoint_path)
                
                # Compute and log statistics
                avg_reward = sum(dataset['reward']) / len(dataset['reward'])
                start_count = sum(1 for a in dataset['action'] if a == 'start')
                start_percentage = (start_count / len(dataset['action'])) * 100
                
                print(f"\nIteration {iteration + 1} Stats:")
                print(f"  Average Reward: {avg_reward:.3f}")
                print(f"  'Start' Actions: {start_count}/{len(dataset['action'])} ({start_percentage:.1f}%)")
            
            # Save final model
            print("\nSaving final model...")
            self.model.save_pretrained(f"{self.output_dir}/final_model")
            self.processor.save_pretrained(f"{self.output_dir}/final_model")
            
            print("\n" + "="*60)
            print("Training complete!")
            print("="*60)
            
        finally:
            # Cleanup
            self.env.close()


def train_pokemon_vlm():
    """
    Example: Train a VLM to play Pokemon using GRPO
    Simple PoC: Train the model to press 'start' indefinitely
    Configured for online RL (1 episode per update)
    """
    # Make sure Chrome is running with debugging enabled:
    # google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
    
    trainer = PokemonGRPOTrainer(
        model_name="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        debug_port=9222,
        game_url="https://pokerogue.net",
        output_dir="./pokemon_vlm_checkpoints",
    )
    
    # Start training with online RL configuration
    # The model collects 1 episode, trains, then collects another with updated model
    trainer.train(
        num_iterations=50,          # Many iterations for online RL
        episodes_per_iteration=1,   # 1 episode per update = online RL
        max_steps_per_episode=20    # 20 steps per episode
    )


def main():
    train_pokemon_vlm()

if __name__ == "__main__":
    main()
