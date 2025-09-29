import asyncio
import os
from browser_use import Agent, Browser
from browser_use.llm import ChatOpenAI
import yaml
from openai import OpenAI

from tools import SEND_KEYS_ONLY
from sys import argv

os.environ["ANONYMIZED_TELEMETRY"] = "false"


def load_config(conf):
    with open(conf) as f:
        conf = yaml.safe_load(f)
        print(conf)

    token = os.getenv(conf["api_token_var"], "")
    return conf, token


def pick_first_model(client: OpenAI) -> str:
    models = client.models.list()
    if not models.data:
        raise RuntimeError("No models available at the endpoint http://localhost:8000/v1")
    return models.data[0].id


async def main(conf):
    conf, token = load_config(conf)
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
    )

    agent = agent(
        **agent_conf,
        browser=browser,
        llm=llm,
        tools=SEND_KEYS_ONLY,
        image_detail='low',
        save_conversation_path="./agent_data/",
        file_system_path="./agent_data/",
        # this path is just available on my fork
        agent_base_path="./agent_data/"
    )

    result = await agent.run(
        on_step_start=hook
    )
    print(result)

async def hook(*args, **kwargs):
    print(args, kwargs)

if __name__ == "__main__":
    conf = argv[1]
    asyncio.run(main(conf)) 

