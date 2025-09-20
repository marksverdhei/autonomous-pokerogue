# autonomous-pokerogue

This project uses [`uv`](https://github.com/astral-sh/uv), which can be installed in one line: 
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To get started with the project:

```
git clone https://github.com/marksverdhei/autonomous-pokerogue.git
cd autonomous-pokerogue
```

Then install project with
```
uv sync --dev
source .venv/bin/activate
```

For development, make sure to install the project with the `-e` flag 
so the source code isn't distributed to `.venv`:
```
uv pip install -e .
```

Then you can run the default cli, which can be customized in `cli.py` and `pyproject.toml`  
```
$ autonomous-pokerogue
Hello from autonomous-pokerogue
```

How to install:
```bash
pip install git+https://github.com/marksverdhei/autonomous-pokerogue.git
```
