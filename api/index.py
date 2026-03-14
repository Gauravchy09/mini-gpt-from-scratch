from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from infrence.runtime import find_export_dirs, generate_from_export, pick_default_export


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_new_tokens: int = Field(default=100, ge=1, le=300)
    temperature: float = Field(default=0.9, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=0, le=200)
    top_p: float = Field(default=0.92, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.15, ge=1.0, le=2.0)
    export_dir: str | None = None
    device: str = "auto"


app = FastAPI(title="Mini-GPT API", version="1.0.0")


def _resolve_export_dir(user_export_dir: str | None) -> Path:
    if user_export_dir:
        chosen = Path(user_export_dir)
        if not chosen.is_absolute():
            chosen = ROOT_DIR / chosen
        return chosen

    exports = find_export_dirs(ROOT_DIR)
    default_export = pick_default_export(exports)
    if default_export is None:
        raise FileNotFoundError("No export folder found. Expected run_02-*/run_02 with mini_gpt_state.pt")
    return default_export


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "mini-gpt-api", "status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/exports")
def exports() -> dict[str, list[str]]:
    export_dirs = [str(path) for path in find_export_dirs(ROOT_DIR)]
    return {"export_dirs": export_dirs}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, str]:
    try:
        export_dir = _resolve_export_dir(req.export_dir)
        output = generate_from_export(
            export_dir=export_dir,
            prompt=req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            device_name=req.device,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "export_dir": str(export_dir),
        "prompt": req.prompt,
        "generated_text": output,
    }
