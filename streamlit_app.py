from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
import torch
from tokenizers import Tokenizer

from models.gpt_model import GPTConfig, MiniGPT
from utils.device import get_device

ROOT_DIR = Path(__file__).resolve().parent
PREFERRED_EXPORT_NAMES = ["run_02", "run_01"]

DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.9
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.92
DEFAULT_REPETITION_PENALTY = 1.15


def find_artifact_dirs() -> list[Path]:
    found: list[Path] = []

    default_export = ROOT_DIR / "artifacts" / "mini_gpt_demo_export"
    if (default_export / "mini_gpt_state.pt").exists():
        found.append(default_export)

    for parent in sorted(ROOT_DIR.glob("run_*"), reverse=True):
        # First, try preferred export subfolder names explicitly.
        for export_name in PREFERRED_EXPORT_NAMES:
            candidate = parent / export_name
            if (candidate / "mini_gpt_state.pt").exists():
                found.append(candidate)

        # Then, discover any run_* child folder that contains model artifacts.
        for child in sorted(parent.glob("run_*"), reverse=True):
            if (child / "mini_gpt_state.pt").exists():
                found.append(child)

    # Deduplicate while preserving order.
    unique: list[Path] = []
    seen = set()
    for item in found:
        key = str(item.resolve())
        if key not in seen:
            unique.append(item)
            seen.add(key)

    # Prefer run_02 when available, then the rest.
    unique.sort(key=lambda path: (0 if path.name == "run_02" else 1, str(path)), reverse=False)
    return unique


def remap_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    # Notebook export used token_emb/pos_emb names; repository model uses token_embedding/position_embedding.
    key_map = {
        "embedding.token_emb.weight": "embedding.token_embedding.weight",
        "embedding.pos_emb.weight": "embedding.position_embedding.weight",
    }
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        remapped[key_map.get(key, key)] = value
    return remapped


@st.cache_resource(show_spinner=False)
def load_model_bundle(export_dir: str, device_name: str) -> tuple[MiniGPT, Tokenizer, torch.device, GPTConfig, dict[str, Any]]:
    export_path = Path(export_dir)
    model_path = export_path / "mini_gpt_state.pt"
    config_path = export_path / "mini_gpt_config.json"
    tokenizer_path = export_path / "tokenizer" / "tokenizer.json"

    with config_path.open("r", encoding="utf-8") as file:
        model_cfg = json.load(file)
    config = GPTConfig(**model_cfg)

    tokenizer = Tokenizer.from_file(tokenizer_path.as_posix())
    model = MiniGPT(config)
    device = get_device(device_name)

    loaded = torch.load(model_path, map_location=device)
    state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    state_dict = remap_state_dict_keys(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    load_report = {
        "missing": missing,
        "unexpected": unexpected,
        "model_path": str(model_path),
    }
    return model, tokenizer, device, config, load_report


def apply_top_p(next_logits: torch.Tensor, top_p: float) -> torch.Tensor:
    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_remove = cumulative_probs > top_p
    sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
    sorted_remove[..., 0] = False

    remove_mask = torch.zeros_like(next_logits, dtype=torch.bool)
    remove_mask.scatter_(1, sorted_indices, sorted_remove)
    return next_logits.masked_fill(remove_mask, float("-inf"))


def generate_text(
    prompt: str,
    export_dir: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device_name: str,
) -> str:
    model, tokenizer, device, _, _ = load_model_bundle(export_dir, device_name)

    encoded = tokenizer.encode(prompt)
    input_ids = encoded.ids
    if not input_ids:
        return ""

    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = x[:, -model.config.context_length :]
            logits, _ = model(context)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)

            if repetition_penalty > 1.0:
                for batch_idx in range(x.size(0)):
                    seen_tokens = torch.unique(x[batch_idx])
                    next_logits[batch_idx, seen_tokens] = next_logits[batch_idx, seen_tokens] / repetition_penalty

            if top_k > 0:
                values, _ = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
                cutoff = values[:, [-1]]
                next_logits = torch.where(
                    next_logits < cutoff,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            if 0.0 < top_p < 1.0:
                next_logits = apply_top_p(next_logits, top_p)

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist(), skip_special_tokens=True)


def render_sidebar(artifact_dirs: list[Path]) -> tuple[str, str, int, float, int, float, float]:
    st.sidebar.header("Hyperparameters")

    choices = [str(path) for path in artifact_dirs]
    if not choices:
        st.sidebar.warning("No exported model folder found.")
        return "", "cpu", DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY

    preferred_index = 0
    for idx, path_str in enumerate(choices):
        if path_str.replace("\\", "/").endswith("/run_02"):
            preferred_index = idx
            break
    export_dir = st.sidebar.selectbox("Trained Model Folder", choices, index=preferred_index)

    device_name = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=20, max_value=240, value=DEFAULT_MAX_NEW_TOKENS, step=10)
    temperature = st.sidebar.slider("Temperature", min_value=0.2, max_value=1.5, value=DEFAULT_TEMPERATURE, step=0.05)
    top_k = st.sidebar.slider("Top K", min_value=0, max_value=100, value=DEFAULT_TOP_K, step=5)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=DEFAULT_TOP_P, step=0.01)
    repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=1.5, value=DEFAULT_REPETITION_PENALTY, step=0.01)
    return export_dir, device_name, max_new_tokens, temperature, top_k, top_p, repetition_penalty


def main() -> None:
    st.set_page_config(page_title="Mini-GPT Demo", page_icon="GPT", layout="wide")
    st.title("Mini-GPT Streamlit Demo")
    st.caption("Simple frontend for prompt-based text generation using your trained run_02 export.")

    artifact_dirs = find_artifact_dirs()
    if not artifact_dirs:
        st.error("No exported model found. Expected a folder containing mini_gpt_state.pt.")
        st.write("Export folders searched under project root, for example:")
        st.code("run_02-*/run_02")
        return

    export_dir, device_name, max_new_tokens, temperature, top_k, top_p, repetition_penalty = render_sidebar(artifact_dirs)

    col1, col2 = st.columns([3, 2])
    with col1:
        prompt = st.text_area(
            "Prompt",
            value="When the astronaut landed on Mars, she discovered",
            height=160,
        )
        generate_clicked = st.button("Generate Text", type="primary", use_container_width=True)

        if generate_clicked:
            if not export_dir:
                st.warning("Select a trained model folder from the sidebar.")
            elif not prompt.strip():
                st.warning("Enter a prompt before generating text.")
            else:
                with st.spinner("Generating text..."):
                    generated = generate_text(
                        prompt=prompt,
                        export_dir=export_dir,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        device_name=device_name,
                    )
                st.subheader("Generated Output")
                st.text_area("Result", value=generated, height=220)

    with col2:
        st.subheader("Current Hyperparameters")
        st.write(f"Model folder: {Path(export_dir).name if export_dir else 'N/A'}")
        st.write(f"Device: {device_name}")
        st.write(f"Max new tokens: {max_new_tokens}")
        st.write(f"Temperature: {temperature:.2f}")
        st.write(f"Top-k: {top_k}")
        st.write(f"Top-p: {top_p:.2f}")
        st.write(f"Repetition penalty: {repetition_penalty:.2f}")

        st.subheader("Quick Guide")
        st.caption("Lower temperature gives safer output; higher gives more creative output.")
        st.caption("Top-k and top-p control how many candidate tokens are sampled.")

        st.subheader("Run Command")
        st.code("streamlit run streamlit_app.py", language="bash")
        st.info("UI is intentionally minimal for demo use.")


if __name__ == "__main__":
    main()
