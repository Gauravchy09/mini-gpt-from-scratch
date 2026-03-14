@echo off
python -m uvicorn api.index:app --host 0.0.0.0 --port 8000 --reload
