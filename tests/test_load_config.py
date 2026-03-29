"""Tests for main.load_config — YAML loading and token resolution."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, content: dict) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(content))
    return p


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_tuple_of_conf_and_token(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "MY_TOKEN",
            "llm": {"model": "gpt-4"},
            "agent": {"task": "play"},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MY_TOKEN", "secret123")

        import importlib
        import main as m
        conf, token = m.load_config()

        assert isinstance(conf, dict)
        assert isinstance(token, str)

    def test_token_read_from_env_var(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "OPENROUTER_API_KEY",
            "llm": {},
            "agent": {},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("OPENROUTER_API_KEY", "my-key-xyz")

        import main as m
        _, token = m.load_config()
        assert token == "my-key-xyz"

    def test_missing_env_var_returns_empty_string(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "NONEXISTENT_KEY",
            "llm": {},
            "agent": {},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("NONEXISTENT_KEY", raising=False)

        import main as m
        _, token = m.load_config()
        assert token == ""

    def test_conf_contains_expected_keys(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "MY_TOKEN",
            "llm": {"base_url": "http://localhost"},
            "agent": {"task": "do something"},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("MY_TOKEN", "")

        import main as m
        conf, _ = m.load_config()
        assert "api_token_var" in conf
        assert "llm" in conf
        assert "agent" in conf

    def test_llm_conf_preserved(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "T",
            "llm": {"base_url": "https://api.example.com", "model": "gpt-5"},
            "agent": {},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("T", "")

        import main as m
        conf, _ = m.load_config()
        assert conf["llm"]["model"] == "gpt-5"
        assert conf["llm"]["base_url"] == "https://api.example.com"

    def test_agent_conf_preserved(self, tmp_path, monkeypatch):
        cfg = _write_config(tmp_path, {
            "api_token_var": "T",
            "llm": {},
            "agent": {"task": "win the game"},
        })
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("T", "key")

        import main as m
        conf, _ = m.load_config()
        assert conf["agent"]["task"] == "win the game"
