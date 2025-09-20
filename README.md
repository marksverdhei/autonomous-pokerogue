# autonomous-pokerogue  

This project is in super-alpha. Expect things to break and agent to get stuck.

This project uses [`uv`](https://github.com/astral-sh/uv), which can be installed in one line: 
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To get started with the project:

```
git clone https://github.com/marksverdhei/python-template.git
```

Then install project with
```
uv sync --dev
```

Launch browser:
```
brave-browser --remote-debugging-port=9222 --user-data-dir=/tmp/brave-profile
```
On the browser, you want to navigate to and log in onto pokerogue.net or the 
instance of locally running pokerogue. Here you want to manually start the game
and navigate to the first battle

Set ANTHROPIC_API_KEY:
```
export ANTHROPIC_API_KEY=xxx
```
And finally, run the program

```
uv run main.py
```
