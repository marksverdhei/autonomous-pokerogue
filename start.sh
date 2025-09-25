#!/bin/bash
while true; do
    uv run src/main.py $1
    echo "Program crashed with exit code $? — restarting..." >&2
    sleep 1
done
