"""
Browser environment for Pokemon game
"""
from enum import auto
import time
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict, List, Callable
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
import nest_asyncio
nest_asyncio.apply()

ACTION_SPACE = {
    'up': 'ArrowUp',
    'down': 'ArrowDown',
    'left': 'ArrowLeft',
    'right': 'ArrowRight',
    'a': 'z',
    'b': 'x',
    'start': 'Escape',
}

# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)


class PokemonBrowserEnv:
    """Browser environment for Pokemon game"""
    def __init__(
        self,
        debug_port: int = 9222,
        reward_functions: Optional[List[Callable]] = None,
        image_resolution: Optional[Tuple[int, int]] = None,
    ):
        self.debug_port = debug_port
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright: Optional[Playwright] = None

        self.action_to_key = ACTION_SPACE
        self.action_space = list(self.action_to_key.keys())

        # Reward functions
        self.reward_functions = reward_functions if reward_functions is not None else []

        # Image resolution (width, height)
        self.image_resolution = image_resolution

        self.sleep = 0.1

    def connect(self, url: Optional[str] = None):
        """Connect to existing browser"""
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.connect_over_cdp(
            f"http://localhost:{self.debug_port}"
        )

        contexts = self.browser.contexts
        if contexts and len(contexts[0].pages) > 0:
            self.page = contexts[0].pages[0]
        else:
            context = self.browser.new_context()
            self.page = context.new_page()

        if url:
            self.page.goto(url)
            self.page.wait_for_load_state('networkidle')

        self.page.bring_to_front()

    def capture_screenshot(self, as_array: bool = True) -> np.ndarray:
        """Capture screenshot"""
        if not self.page:
            raise RuntimeError("Browser not connected")

        screenshot_bytes = self.page.screenshot(type='png')

        if as_array:
            image = Image.open(BytesIO(screenshot_bytes))
            # Resize if resolution is specified
            if self.image_resolution is not None:
                image = image.resize(self.image_resolution, Image.LANCZOS)
            return np.array(image)
        return screenshot_bytes

    def send_action(self, action: str, duration_ms: int = 100):
        """Execute action in browser"""
        print("Sendt action:", action)
        if not self.page:
            raise RuntimeError("Browser not connected")

        action = action.lower()
        if action not in self.action_to_key:
            raise ValueError(f"Unknown action: {action}")

        key = self.action_to_key[action]

        if key is None:
            self.page.wait_for_timeout(duration_ms)
            return

        self.page.keyboard.down(key)
        self.page.wait_for_timeout(duration_ms)
        self.page.keyboard.up(key)

        if self.sleep:
            time.sleep(self.sleep)


    def step(self, action: str, prev_obs: np.ndarray, duration_ms: int = 500) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute step (Gym interface)

        Args:
            action: Action to take
            prev_obs: Screenshot before action (for reward computation)
            duration_ms: Duration to hold key

        Returns:
            next_obs: Screenshot after action
            reward: Computed reward
            done: Whether episode is done
            info: Additional info
        """
        self.send_action(action, duration_ms)
        next_obs = self.capture_screenshot()

        # Convert numpy arrays to PIL Images for reward functions
        prev_obs_pil = Image.fromarray(prev_obs)
        next_obs_pil = Image.fromarray(next_obs)

        # Compute reward using reward functions
        reward = 0.0
        for reward_fn in self.reward_functions:
            reward += reward_fn(
                prev_obs=prev_obs_pil,
                next_obs=next_obs_pil,
                action=action,
                env=self
            )

        done = False
        info = {'action': action}

        return next_obs, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment"""

        self.sleep = 0.5
        return self.capture_screenshot()

    def close(self):
        """Close browser connection"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def ocr(self):
        image = self.capture_screenshot()
        # image = Image.fromarray(image)
        prediction, = ocr.predict(image)

        return [s.lower() for s in prediction['rec_texts']]


env = PokemonBrowserEnv(image_resolution=(640, 360))


def navigate_title_screen():
    ocr_data = env.ocr()
    
    if any('continue' in s for s in ocr_data):
        navigate_titlescreen_continue()
    else:
        navigate_titlescreen_newgame()

def navigate_titlescreen_newgame():
    print("New game")
    env.send_action("a")
    env.send_action("a")
    navigate_pc()

def navigate_titlescreen_continue():
    print("Continue")
    env.send_action("a")
    env.send_action("a")


def navigate_pc():
    # PC
    # Random pokemon
    env.send_action("left")

    for i in range(6):
        env.send_action("a")

    # enter game
    env.send_action("up")
    env.send_action("up")

    env.send_action("a")

def autoplay():
    navigate_title_screen()

    while True:
        ocr_data = env.ocr()
        print(ocr_data)
        env.send_action("a")

        # Cover moves - 'It won't have any effect.'
        if any("effect" in s for s in ocr_data):
            env.send_action("down")



def test_ocr():
    while True:
        print(env.ocr())
        input()

# def unstuck():
    # It won't have any effect.

    


def main():
    # env.connect('https://pokerogue.net')
    env.connect()
    try:
        # test_ocr()
        autoplay()
    except KeyboardInterrupt:
        print("Terminating")
        env.close()


if __name__ == "__main__":
    main()
