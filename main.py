import asyncio
import os
from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI
from dotenv import load_dotenv
import yaml

os.environ["ANONYMIZED_TELEMETRY"] = "false"
data = load_dotenv()

with open("config.yaml") as f:
    conf = yaml.safe_load(f)
    print(conf)

token = os.getenv(conf["api_token_var"], "")

llm_conf = conf["llm"]
llm_conf["api_key"] = token

agent_conf = conf["agent"]

llm = ChatOpenAI(**llm_conf)

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
        **agent_conf,
        browser=browser,
        llm=llm,
    )

    result = await agent.run()
    print(result)

if __name__ == "__main__":
    asyncio.run(main()) 
