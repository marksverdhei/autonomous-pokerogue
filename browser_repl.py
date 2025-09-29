
import asyncio
from browser_use import Browser, Tools
from browser_use.tools.views import SendKeysAction
import asyncio
from browser_use.browser.session import BrowserSession
from browser_use.browser.events import SendKeysEvent

async def invoke_send_keys(browser_session: BrowserSession, keys_to_send: str):
    print(f"sending keys: {keys_to_send}")
    send_keys_event = SendKeysEvent(keys=keys_to_send)
    event = browser_session.event_bus.dispatch(send_keys_event)
    await event
    await event.event_result(raise_if_any=True, raise_if_none=False)


async def main():
    brave_path = "/usr/bin/brave-browser"
    browser = Browser(
        headless=False,
        cdp_url="http://localhost:9222",
        executable_path=brave_path,
    )
    png = browser.page.screenshot(full_page=False)  # or element_handle.screenshot()
    print(png)
    # send_keys = Tools().registry.registry.actions["send_keys"].function
    # now call the built-in send_keys
    # (Browser manages only one session unless you run multiple agents)

    

    # await browser.start()
    # current = await browser.get_current_page()
    # print(current)
    # while True:
    #     await current.press('Enter')
    #     # await invoke_send_keys(browser, 'Space')
    #     input()

asyncio.run(main())
