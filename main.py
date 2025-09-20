import asyncio
import os
from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI

os.environ["ANONYMIZED_TELEMETRY"] = "false"
key = os.environ["OPENROUTER_API_KEY"]

vllm_url = "http://localhost:8000/v1"
or_url = "https://openrouter.ai/api/v1"

AT_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
AT_BASE_URL = "https://api.anthropic.com/v1"

llm = ChatOpenAI(
    model="claude-sonnet-4-20250514",
    base_url=AT_BASE_URL,
    api_key=AT_API_KEY,
    max_completion_tokens=1024,   # keep completions modest to avoid context-limit 400s
)

prompt = """
Your task is to play pokerogue and try to beat the game.
Actions are made by keystrokes. 
'space' - confirm
'backspace' - back
arrow keys ('up', 'down', 'left', 'right') - move up down left right
Mouse is not usable in this game.

You can catch new pokemon to assemble a strong team.
Between fights you can pick and buy rewards.
To proceed through dialogue, press space.
Good luck.
"""

async def main():

    brave_path = "/usr/bin/brave-browser"  # change to your system path

    browser = Browser(
        headless=False,
        cdp_url="http://localhost:9222",
        executable_path=brave_path,  # 👈 use Brave instead of default Chromium
        args=[
            "--enable-gpu",
            "--ignore-gpu-blocklist",
            "--enable-webgl",
            "--use-angle=gl",
            "--disable-software-rasterizer",
        ],
    )

    agent = Agent(
        task=prompt,
        browser=browser,
        llm=llm,
    )

    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 
