"""
Browser environment for Pokemon game
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, Dict
from playwright.sync_api import sync_playwright, Page, Browser, Playwright

from config import ACTION_SPACE, CONSTRAIN_ACTIONS, START_FROM_ACTIVE_SESSION


class PokemonBrowserEnv:
    """Browser environment for Pokemon game"""

    CONSTRAIN_ACTIONS = CONSTRAIN_ACTIONS
    START_FROM_ACTIVE_SESSION = START_FROM_ACTIVE_SESSION

    def __init__(self, debug_port: int = 9222):
        self.debug_port = debug_port
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
        self.playwright: Optional[Playwright] = None

        self.action_to_key = ACTION_SPACE
        self.action_space = list(self.action_to_key.keys())

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

        if self.START_FROM_ACTIVE_SESSION:
            print("Using active browser session")
            self.page.bring_to_front()
            return

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
            return np.array(image)
        return screenshot_bytes

    def send_action(self, action: str, duration_ms: int = 100):
        """Execute action in browser"""
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

    def step(self, action: str, duration_ms: int = 100) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute step (Gym interface)"""
        self.send_action(action, duration_ms)
        observation = self.capture_screenshot()

        # Compute reward
        reward = 1.0 if action == 'start' else -0.1
        done = False
        info = {'action': action}

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """Reset environment"""
        return self.capture_screenshot()

    def close(self):
        """Close browser connection"""
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()
