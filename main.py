import asyncio
import os
from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI
import yaml

from PIL import Image
import imagehash

os.environ["ANONYMIZED_TELEMETRY"] = "false"

def load_config():
    with open("config.yaml") as f:
        conf = yaml.safe_load(f)
        print(conf)

    token = os.getenv(conf["api_token_var"], "")
    return conf, token





def identify_cursor():
    """
    Used for textual scaffolding
    """
    pass


def identify_state():
    """
    which menu
    """
    pass


def screenshot_ocr():
    pass


def classify_screen(img_path, examples):
    """
    Classify img_path as one of the example labels.
    
    examples: dict[label -> path_to_example_image]
    Returns: best_label, best_distance
    """
    img = Image.open(img_path).convert("RGB")
    h = imagehash.phash(img)  # 64-bit perceptual hash
    
    best_label, best_dist = None, 999
    for label, ex_path in examples.items():
        ex_hash = imagehash.phash(Image.open(ex_path).convert("RGB"))
        dist = h - ex_hash  # Hamming distance
        if dist < best_dist:
            best_label, best_dist = label, dist
    
    return best_label, best_dist

# --- Example usage ---
# examples = {
#     "main menu": "main_menu.png",
#     "battle": "battle.png",
#     "party menu": "party_menu.png",
#     "computer": "computer.png",
# }


async def main():
    conf, token = load_config()
    llm_conf = conf["llm"]
    llm_conf["api_key"] = token
    agent_conf = conf["agent"]
    llm = ChatOpenAI(**llm_conf)
    brave_path = "/usr/bin/brave-browser"
    browser = Browser(
        headless=False,
        cdp_url="http://localhost:9222",
        executable_path=brave_path,
        args=[
            "--enable-gpu",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
            "--use-angle=gl",
            "--disable-software-rasterizer",
        ],
    )

    agent = Agent(
        **agent_conf,
        browser=browser,
        llm=llm,
    )

    result = await agent.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main()) 
