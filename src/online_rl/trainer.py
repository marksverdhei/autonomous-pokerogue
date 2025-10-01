"""
Online RL trainer for Pokemon VLM
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import Tuple, Dict
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
    ):
        self.model_name = model_name
        self.debug_port = debug_port
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        os.makedirs(output_dir, exist_ok=True)

        # Initialize environment
        self.env = PokemonBrowserEnv(debug_port=debug_port)

        # Load model and processor
        print(f"Loading model: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
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

    def prepare_prompt(self, image: np.ndarray) -> str:
        """Create prompt for VLM"""
        return f"<image>\nSelect one of the following possible actions: {self.env.action_space}\nAction:"

    def generate_action_with_logprobs(
        self,
        image: np.ndarray,
        temperature: float = 1.0
    ) -> Tuple[str, torch.Tensor, torch.Tensor]:
        """
        Generate action and compute log probability

        Returns:
            action: Generated action string
            log_prob: Log probability of the action
            entropy: Entropy of the action distribution
        """
        pil_image = Image.fromarray(image)
        prompt_text = self.prepare_prompt(image)

        # Prepare inputs - each call is independent, no history accumulation
        inputs = self.processor(
            images=pil_image,
            text=prompt_text,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Clear any cached key-values to ensure fresh context each time
        self.model.config.use_cache = False

        # Log context size to verify only one image
        input_length = inputs['input_ids'].shape[1]

        # Forward pass to get logits
        with torch.set_grad_enabled(True):
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Verify output sequence length matches input (no accumulation)
            assert logits.shape[1] == input_length, \
                f"Context accumulation detected! Input: {input_length}, Output: {logits.shape[1]}"

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

        return action, log_prob, entropy

    def train_episode(
        self,
        max_steps: int = 50,
        temperature: float = 1.0,
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

        # Reset environment
        obs = self.env.reset()

        # Collect episode
        print(f"\n{'─'*60}")
        print("COLLECTING EPISODE")
        print(f"{'─'*60}")

        for step in range(max_steps):
            # Generate action with log probability
            action, log_prob, entropy = self.generate_action_with_logprobs(
                obs,
                temperature=temperature
            )

            # Execute action
            next_obs, reward, done, info = self.env.step(action)

            # Store
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            actions_taken.append(action)

            print(f"[Step {step+1}/{max_steps}] action={action}, reward={reward:.2f}")

            obs = next_obs

            if done:
                break

        # Compute returns
        returns = compute_returns(rewards, self.gamma)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.model.device)

        # Normalize returns (helps with training stability)
        returns = normalize_returns(returns)

        # Compute policy gradient loss
        print(f"\n{'─'*60}")
        print("UPDATING MODEL")
        print(f"{'─'*60}")

        policy_losses = []
        entropy_losses = []

        for log_prob, entropy, R in zip(log_probs, entropies, returns):
            # Policy gradient: -log_prob * return
            policy_loss = -log_prob * R
            policy_losses.append(policy_loss)

            # Entropy bonus for exploration
            entropy_losses.append(-entropy)

        # Total loss
        policy_loss = torch.stack(policy_losses).mean()
        entropy_loss = torch.stack(entropy_losses).mean()
        total_loss = policy_loss + self.entropy_coef * entropy_loss

        print(f"Policy Loss: {policy_loss.item():.4f}")
        print(f"Entropy Loss: {entropy_loss.item():.4f}")
        print(f"Total Loss: {total_loss.item():.4f}")

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Compute statistics
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)
        start_count = sum(1 for a in actions_taken if a == 'start')
        start_percentage = (start_count / len(actions_taken)) * 100

        stats = {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'steps': len(rewards),
            'start_count': start_count,
            'start_percentage': start_percentage,
            'policy_loss': policy_loss.item(),
            'entropy': entropy_losses[0].item() if entropy_losses else 0,
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
                )

                # Log statistics
                print(f"\n{'─'*60}")
                print("EPISODE SUMMARY")
                print(f"{'─'*60}")
                print(f"Total Reward: {stats['total_reward']:.2f}")
                print(f"Avg Reward: {stats['avg_reward']:.3f}")
                print(f"Steps: {stats['steps']}")
                print(f"'Start' Actions: {stats['start_count']}/{stats['steps']} ({stats['start_percentage']:.1f}%)")
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
