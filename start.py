"""Render entrypoint: build the local index, then start FastAPI."""

from __future__ import annotations

import os
import subprocess
import sys

import ingest


def main() -> None:
    ingest.ensure_index()
    port = os.getenv("PORT", "8501")
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--host",
        "0.0.0.0",
        "--port",
        port,
    ]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
