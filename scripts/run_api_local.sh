#!/usr/bin/env bash
set -e

export ENABLE_LOCAL_INFERENCE=1
python -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
