#!/bin/bash
while true; do
    uv run main.py
    echo "Program crashed with exit code $? — restarting..." >&2
    sleep 1
done
