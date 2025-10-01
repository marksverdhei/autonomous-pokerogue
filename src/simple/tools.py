from typing import Literal
from browser_use import Tools, Agent, ActionResult

DEFAULT_TOOLS = list(Tools().registry.registry.actions.keys())

SEND_KEYS_ONLY = Tools(
    exclude_actions=list(set(DEFAULT_TOOLS) - {'send_keys'})
)

CLASSIC_KEYS = ['a', 'b', 'up', 'down', 'left', 'right', 'start']
CLASSIC_KEY_TYPE = Literal['a', 'b', 'up', 'down', 'left', 'right', 'start']

INPUT_ONLY = Tools(exclude_actions=DEFAULT_TOOLS)

@INPUT_ONLY.action(description=f'Give an input to the game. Input type: {CLASSIC_KEYS}')
def send_input(input_key: CLASSIC_KEY_TYPE) -> ActionResult:
    return 'ok'
