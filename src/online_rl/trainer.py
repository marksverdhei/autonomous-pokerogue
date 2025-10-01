"""
Online RL trainer for Pokemon VLM
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict, List, Callable, Optional
from transformers import AutoProcessor, AutoModelForImageTextToText

from environment import PokemonBrowserEnv
from utils import compute_returns, normalize_returns


class OnlinePokemonRLTrainer:
    """
    Online RL trainer - no dataset, learns from live interaction
    Uses policy gradient (REINFORCE-style) updates
    """

    def __init__(
        self,
        model_name: str = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        debug_port: int = 9222,
        output_dir: str = "./pokemon_online_rl",
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        reward_functions: Optional[List[Callable]] = None,
        image_resolution: Optional[Tuple[int, int]] = None,
        log_inputs: bool = False,
        max_turns: Optional[int] = 1,
    ):
        self.model_name = model_name
        self.debug_port = debug_port
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.log_inputs = log_inputs
        self.max_turns = max_turns

        os.makedirs(output_dir, exist_ok=True)
        if self.log_inputs:
            self.logs_dir = os.path.join(output_dir, "input_logs")
            os.makedirs(self.logs_dir, exist_ok=True)

        # Initialize environment
        self.env = PokemonBrowserEnv(
            debug_port=debug_port,
            reward_functions=reward_functions,
            image_resolution=image_resolution,
        )

        # Load model and processor
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Override processor image size if resolution is specified
        if image_resolution is not None and hasattr(self.processor, 'image_processor'):
            width, height = image_resolution
            self.processor.image_processor.size = {"width": width, "height": height}
            print(f"Set processor image size to: {width}x{height}")

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float32,
        )

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate
        )

        # Setup action tokens
        self.setup_action_tokens()

        # Training statistics
        self.episode_rewards = []
        self.episode_actions = []

    def setup_action_tokens(self):
        """Map actions to token IDs"""
        self.action_token_ids = []
        self.token_id_to_action = {}

        for action in self.env.action_space:
            tokens = self.processor.tokenizer.encode(action, add_special_tokens=False)
            if len(tokens) > 0:
                token_id = tokens[0]
                self.action_token_ids.append(token_id)
                self.token_id_to_action[token_id] = action

        print(f"Action tokens: {self.token_id_to_action}")

    def prepare_prompt(self, num_images: int, action_history: List[str]) -> str:
        """
        Create prompt for VLM with multiple images and actions in history

        Args:
            num_images: Number of images in history
            action_history: List of previous actions taken

        Returns:
            Formatted prompt with interleaved images and actions
        """
        # Build conversation history
        prompt_parts = []

        # For each historical turn (image + action pair)
        for i in range(len(action_history)):
            prompt_parts.append(f"<image>\nSelect one of the following possible actions: {self.env.action_space}\nAction: {action_history[i]}")

        # Add current turn (image without action yet)
        prompt_parts.append(f"<image>\nSelect one of the following possible actions: {self.env.action_space}\nAction:")

        return "\n".join(prompt_parts)

    def generate_action_with_logprobs(
        self,
        history: List[np.ndarray],
        action_history: List[str],
        temperature: float = 1.0
    ) -> Tuple[str, torch.Tensor, torch.Tensor, int]:
        """
        Generate action and compute log probability

        Args:
            history: List of observations (images) to pass to model
            action_history: List of previous actions taken
            temperature: Sampling temperature

        Returns:
            action: Generated action string
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            num_tokens: Number of input tokens
        """
        # Convert all images to PIL
        pil_images = [Image.fromarray(img) for img in history]
        prompt_text = self.prepare_prompt(len(pil_images), action_history)

        # Debug: Log image sizes before processor
        if len(pil_images) > 0:
            img_sizes = [img.size for img in pil_images]
            print(f"  [DEBUG] Image sizes before processor: {img_sizes}")

        # Prepare inputs with history
        inputs = self.processor(
            images=pil_images,
            text=prompt_text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Debug: Log pixel values shape after processor
        if 'pixel_values' in inputs:
            print(f"  [DEBUG] pixel_values shape after processor: {inputs['pixel_values'].shape}")

        # Clear any cached key-values to ensure fresh context each time
        self.model.config.use_cache = False

        # Log context size
        input_length = inputs['input_ids'].shape[1]

        # Forward pass to get logits
        with torch.set_grad_enabled(True):
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Verify output sequence length matches input
            assert logits.shape[1] == input_length, \
                f"Unexpected output length! Input: {input_length}, Output: {logits.shape[1]}"

            # Get logits for next token position
            next_token_logits = logits[:, -1, :] / temperature

            # Apply action constraints if enabled
            if self.env.CONSTRAIN_ACTIONS:
                mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                for token_id in self.action_token_ids:
                    if token_id < next_token_logits.shape[-1]:
                        mask[:, token_id] = False
                next_token_logits = next_token_logits.masked_fill(mask, float('-inf'))

            # Sample action token
            probs = torch.softmax(next_token_logits, dim=-1)
            action_token = torch.multinomial(probs, num_samples=1)

            # Compute log probability
            log_prob = torch.log(probs[0, action_token[0, 0]] + 1e-10)

            # Compute entropy for exploration bonus
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

        # Decode action
        action_token_id = action_token[0, 0].item()
        action = self.token_id_to_action.get(action_token_id, 'noop')

        return action, log_prob, entropy, input_length

    def _log_episode_inputs(
        self,
        episode_num: int,
        images: List[np.ndarray],
        actions: List[str],
        token_counts: List[int],
    ):
        """Log images and text inputs from an episode"""
        episode_dir = os.path.join(self.logs_dir, f"episode_{episode_num}")
        os.makedirs(episode_dir, exist_ok=True)

        # Save images
        for step, img in enumerate(images):
            pil_img = Image.fromarray(img)
            pil_img.save(os.path.join(episode_dir, f"step_{step}.png"))

        # Save text prompt (final step with full history)
        if images:
            # For logging, include all actions except the last one (which hasn't been taken yet)
            prompt_text = self.prepare_prompt(len(images), actions[:-1] if len(actions) > 0 else [])
            with open(os.path.join(episode_dir, "final_prompt.txt"), "w") as f:
                f.write(prompt_text)

        # Save metadata
        metadata = {
            "episode": episode_num,
            "num_steps": len(images),
            "actions": actions,
            "token_counts": token_counts,
        }
        with open(os.path.join(episode_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    def train_episode(
        self,
        max_steps: int = 50,
        temperature: float = 1.0,
        episode_num: int = 0,
    ) -> Dict[str, float]:
        """
        Run one episode and update the model

        Returns:
            Dictionary with episode statistics
        """
        # Storage for episode
        log_probs = []
        entropies = []
        rewards = []
        actions_taken = []
        episode_images = []
        token_counts = []
        history = []  # History of observations for multi-turn context
        action_history = []  # History of actions taken

        # Reset environment
        obs = self.env.reset()

        # Collect episode
        print(f"\n{'─'*60}")
        print("COLLECTING EPISODE")
        print(f"{'─'*60}")

        for step in range(max_steps):
            # Add current observation to history
            history.append(obs)

            # Apply sliding window if max_turns is set
            if self.max_turns is not None and len(history) > self.max_turns:
                history = history[-self.max_turns:]
                # Also trim action_history to match (it should be len(history) - 1)
                if len(action_history) >= self.max_turns:
                    action_history = action_history[-(self.max_turns - 1):]

            # Generate action with log probability using history and action_history
            action, log_prob, entropy, num_tokens = self.generate_action_with_logprobs(
                history,
                action_history,
                temperature=temperature
            )

            # Execute action (pass current obs as prev_obs for reward computation)
            next_obs, reward, done, info = self.env.step(action, prev_obs=obs)

            # Store
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            actions_taken.append(action)
            episode_images.append(obs)
            token_counts.append(num_tokens)

            # Add action to history for next step
            action_history.append(action)

            print(f"[Step {step+1}/{max_steps}] action={action}, reward={reward:.2f}, history_len={len(history)}, action_history_len={len(action_history)}, tokens={num_tokens}")

            # Pipeline: next_obs becomes prev_obs for next iteration
            obs = next_obs

            if done:
                break

        # Log inputs if enabled (only final step with full history)
        if self.log_inputs:
            self._log_episode_inputs(episode_num, episode_images, actions_taken, token_counts)

        # Compute returns
        returns = compute_returns(rewards, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.model.device)

        # Normalize returns (helps with training stability)
        returns = normalize_returns(returns)

        # Compute policy gradient loss with gradient accumulation
        print(f"\n{'─'*60}")
        print("UPDATING MODEL")
        print(f"{'─'*60}")

        self.optimizer.zero_grad()

        total_policy_loss = 0.0
        total_entropy_loss = 0.0

        # Accumulate gradients step by step to save memory
        for i, (log_prob, entropy, R) in enumerate(zip(log_probs, entropies, returns)):
            # Policy gradient: -log_prob * return
            policy_loss = -log_prob * R

            # Entropy bonus for exploration
            entropy_loss = -entropy

            # Combined loss for this step
            step_loss = (policy_loss + self.entropy_coef * entropy_loss) / len(log_probs)

            # Backward pass for this step (accumulates gradients)
            step_loss.backward()

            # Track losses for logging (detach to save memory)
            total_policy_loss += policy_loss.detach().item()
            total_entropy_loss += entropy_loss.detach().item()

        # Average losses for logging
        avg_policy_loss = total_policy_loss / len(log_probs)
        avg_entropy_loss = total_entropy_loss / len(log_probs)

        print(f"Policy Loss: {avg_policy_loss:.4f}")
        print(f"Entropy Loss: {avg_entropy_loss:.4f}")
        print(f"Total Loss: {avg_policy_loss + self.entropy_coef * avg_entropy_loss:.4f}")

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Single optimizer step with accumulated gradients
        self.optimizer.step()

        # Compute statistics
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)

        stats = {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': len(rewards),
            'policy_loss': avg_policy_loss,
            'entropy': avg_entropy_loss,
        }

        return stats

    def train(
        self,
        num_episodes: int = 100,
        max_steps_per_episode: int = 50,
        save_every: int = 10,
        temperature: float = 1.0,
    ):
        """
        Main training loop

        Args:
            num_episodes: Number of episodes to train
            max_steps_per_episode: Max steps per episode
            save_every: Save checkpoint every N episodes
            temperature: Sampling temperature (higher = more exploration)
        """
        # Connect to browser
        print("Connecting to browser...")
        self.env.connect()
        print("Browser connected!\n")

        try:
            for episode in range(num_episodes):
                print(f"\n{'='*60}")
                print(f"EPISODE {episode + 1}/{num_episodes}")
                print(f"{'='*60}")

                # Train one episode
                stats = self.train_episode(
                    max_steps=max_steps_per_episode,
                    temperature=temperature,
                    episode_num=episode,
                )

                # Log statistics
                print(f"\n{'─'*60}")
                print("EPISODE SUMMARY")
                print(f"{'─'*60}")
                print(f"Total Reward: {stats['total_reward']:.2f}")
                print(f"Avg Reward: {stats['avg_reward']:.3f}")
                print(f"Steps: {stats['steps']}")
                print(f"Policy Loss: {stats['policy_loss']:.4f}")
                print(f"{'─'*60}\n")

                # Save checkpoint
                if (episode + 1) % save_every == 0:
                    checkpoint_path = f"{self.output_dir}/episode_{episode + 1}"
                    print(f"Saving checkpoint to {checkpoint_path}")
                    self.model.save_pretrained(checkpoint_path)
                    self.processor.save_pretrained(checkpoint_path)

            # Save final model
            final_path = f"{self.output_dir}/final_model"
            print(f"\nSaving final model to {final_path}")
            self.model.save_pretrained(final_path)
            self.processor.save_pretrained(final_path)

            print("\n" + "="*60)
            print("TRAINING COMPLETE!")
            print("="*60)

        finally:
            self.env.close()
