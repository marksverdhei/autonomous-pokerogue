import asyncio
import os
from browser_use import Agent, Browser, Tools
from browser_use.llm import ChatOpenAI
import yaml
from openai import OpenAI

from PIL import Image
import imagehash
from tools import SEND_KEYS_ONLY

os.environ["ANONYMIZED_TELEMETRY"] = "false"

def load_config():
    with open("config.yaml") as f:
        conf = yaml.safe_load(f)
        print(conf)

    token = os.getenv(conf["api_token_var"], "")
    return conf, token


def pick_first_model(client: OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available at the endpoint http://localhost:8000/v1")
    return models.data[0].id


async def main():
    conf, token = load_config()
    llm_conf = conf["llm"]
    if llm_conf['model'] == '*':
        llm_conf['model'] = pick_first_model(OpenAI(base_url=llm_conf['base_url'], api_key=token))
        print('Using local model:', llm_conf['model'])
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
        tools=SEND_KEYS_ONLY,
        image_detail='low',
    )

    result = await agent.run()
    print(result)


if __name__ == "__main__":
    asyncio.run(main()) 

