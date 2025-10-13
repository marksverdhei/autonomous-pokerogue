import time
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, Optional, List, Callable
from playwright.sync_api import sync_playwright, Page, Browser, Playwright
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

ACTION_SPACE = {
    'up': 'ArrowUp',
    'down': 'ArrowDown',
    'left': 'ArrowLeft',
    'right': 'ArrowRight',
    'a': 'z',
    'b': 'x',
    'start': 'Escape',
}

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
        self.url = "https://pokerogue.net"

        self.sleep = 0.1

    def connect(self):
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

        if self.page.url != self.url:
            self.page.goto(self.url)
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
