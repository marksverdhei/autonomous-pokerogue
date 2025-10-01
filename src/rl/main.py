"""
Pokemon VLM Browser Interface using Playwright (Synchronous Wrapper)
Compatible with TRL GRPO and standard RL training loops
"""

import asyncio
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from typing import Tuple, Optional, Dict, Any
from io import BytesIO
from PIL import Image
import numpy as np
import threading


class PokemonBrowserEnv:
    """
    Synchronous browser environment for Pokemon VLM training
    Compatible with TRL GRPO and standard Gym-like interfaces
    """
    
    # Configuration constants
    HAS_LOGIN_SCREEN = True  # Set to False to skip login sequence
    
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
            url: Optional URL to navigate to after connecting
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
            reward: Reward signal (you'll need to implement this based on game state)
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
        
        # Placeholder for reward and done - implement based on your game logic
        reward = 0.0
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


if __name__ == "__main__":
    # Run the example
    main()
    
    # Uncomment to see TRL integration example
    # example_trl_integration()
