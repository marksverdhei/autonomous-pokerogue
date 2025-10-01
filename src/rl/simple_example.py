
# Example: Integration with TRL GRPO
def example_trl_integration():
    """
    Example showing how to integrate with TRL GRPO
    """
    print("\n=== TRL GRPO Integration Example ===\n")
    
    # Setup
    env = PokemonBrowserEnv(debug_port=9222)
    env.connect(url="<pokemon-game-url>")
    
    # Mock VLM model (replace with your actual model)
    class MockVLM:
        def predict(self, image):
            import random
            return random.choice(env.action_space)
    
    model = MockVLM()
    
    # TRL-style training loop
    num_episodes = 10
    max_steps_per_episode = 100
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = []
        
        for step in range(max_steps_per_episode):
            # VLM predicts action from screenshot
            action = model.predict(obs)
            
            # Execute action in environment
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)
            
            if done:
                break
        
        print(f"Episode {episode + 1}: Total reward = {sum(episode_rewards)}")
    
    env.close()

    # Uncomment to see TRL integration example
    # example_trl_integration()



# Example usage with TRL-compatible training loop
def main():
    """Example usage showing TRL-compatible synchronous interface"""
    
    # Initialize environment
    env = PokemonBrowserEnv(debug_port=9222)
    
    # Connect to browser
    # Start Chrome with: google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/chrome-debug
    env.connect(url="https://pokerogue.net")
    
    print("Connected to browser")
    print(f"Available actions: {env.action_space}")
    
    # Reset environment (Gym-style)
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Simple training loop (TRL-compatible)
    for episode in range(3):
        print(f"\n--- Episode {episode + 1} ---")
        obs = env.reset()
        
        for step in range(5):
            # Example: Take random action (in real training, this would be model.predict(obs))
            import random
            action = random.choice(env.action_space)
            
            print(f"Step {step + 1}: Taking action '{action}'")
            obs, reward, done, info = env.step(action, duration_ms=150)
            
            print(f"  Observation shape: {obs.shape}")
            print(f"  Reward: {reward}")
            print(f"  Done: {done}")
            
            if done:
                break
    
    # Alternative: Direct action interface (non-Gym style)
    print("\n--- Direct action interface ---")
    env.send_action('right', duration_ms=200)
    screenshot = env.capture_screenshot()
    print(f"Screenshot captured: {screenshot.shape}")
    
    # Send action sequence
    env.send_key_sequence(['up', 'up', 'a'], duration_per_key=150)
    
    # Get page state
    state = env.get_page_state()
    print(f"Page state: {state}")
    
    # Example: Execute JavaScript to get game state
    try:
        game_state = env.execute_js("window.gameState || 'No game state available'")
        print(f"Game state: {game_state}")
    except Exception as e:
        print(f"Could not get game state: {e}")
    
    # Close connection
    env.close()
    print("\nBrowser connection closed")
