from __future__ import annotations

import os
from pathlib import Path
import sys

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


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


def _is_local_inference_enabled() -> bool:
    return os.getenv("ENABLE_LOCAL_INFERENCE", "0") == "1"


def _backend_url() -> str:
    return os.getenv("MODEL_API_URL", "").strip().rstrip("/")


def _resolve_local_export_dir(user_export_dir: str | None) -> Path:
    # Import lazily so Vercel deploy does not require torch/tokenizers.
    from infrence.runtime import find_export_dirs, pick_default_export

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
    mode = "proxy" if _backend_url() else ("local" if _is_local_inference_enabled() else "disabled")
    return {"service": "mini-gpt-api", "status": "ok", "mode": mode}


@app.get("/health")
def health() -> dict[str, str]:
    mode = "proxy" if _backend_url() else ("local" if _is_local_inference_enabled() else "disabled")
    return {"status": "healthy", "mode": mode}


@app.get("/exports")
def exports() -> dict[str, list[str]]:
    if not _is_local_inference_enabled():
        return {"export_dirs": []}

    from infrence.runtime import find_export_dirs

    export_dirs = [str(path) for path in find_export_dirs(ROOT_DIR)]
    return {"export_dirs": export_dirs}


@app.post("/generate")
def generate(req: GenerateRequest) -> dict[str, str]:
    backend = _backend_url()

    if backend:
        # Vercel-friendly mode: proxy request to external inference service.
        try:
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{backend}/generate",
                    json=req.model_dump(),
                    headers={"Content-Type": "application/json"},
                )
            response.raise_for_status()
            payload = response.json()
            return {
                "export_dir": str(payload.get("export_dir", "remote")),
                "prompt": req.prompt,
                "generated_text": str(payload.get("generated_text", "")),
            }
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Failed to reach MODEL_API_URL backend: {exc}") from exc

    if not _is_local_inference_enabled():
        raise HTTPException(
            status_code=503,
            detail=(
                "No inference backend configured. Set MODEL_API_URL for proxy mode on Vercel, "
                "or set ENABLE_LOCAL_INFERENCE=1 for local torch inference."
            ),
        )

    try:
        from infrence.runtime import generate_from_export

        export_dir = _resolve_local_export_dir(req.export_dir)
        generated_text = generate_from_export(
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
        "generated_text": generated_text,
    }
