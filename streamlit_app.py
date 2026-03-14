from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch
import yaml

from models.gpt_model import GPTConfig, MiniGPT
from tokenizer.tokenizer import SimpleCharTokenizer
from utils.checkpoint import load_checkpoint
from utils.device import get_device

ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / "experiments" / "checkpoints"
VOCAB_PATH = ROOT_DIR / "tokenizer" / "vocab.json"
MODEL_CONFIG_PATH = ROOT_DIR / "configs" / "model_config.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def available_checkpoints() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    checkpoints = sorted(path.name for path in CHECKPOINT_DIR.glob("*.pt"))
    return checkpoints


@st.cache_resource(show_spinner=False)
def load_model_bundle(checkpoint_name: str, device_name: str) -> tuple[MiniGPT, SimpleCharTokenizer, torch.device, GPTConfig]:
    tokenizer = SimpleCharTokenizer.load(VOCAB_PATH)

    model_cfg = load_yaml(MODEL_CONFIG_PATH)
    model_cfg["vocab_size"] = tokenizer.vocab_size
    config = GPTConfig(**model_cfg)

    model = MiniGPT(config)
    device = get_device(device_name)
    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    load_checkpoint(checkpoint_path, model=model, optimizer=None, map_location=device)
    model.to(device)
    model.eval()
    return model, tokenizer, device, config


def generate_text(
    prompt: str,
    checkpoint_name: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device_name: str,
) -> str:
    model, tokenizer, device, _ = load_model_bundle(checkpoint_name, device_name)
    input_ids = tokenizer.encode(prompt)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_ids=x,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
        )

    return tokenizer.decode(output[0].tolist())


def render_sidebar() -> tuple[str, str, int, float, int]:
    st.sidebar.header("Generation Settings")

    checkpoints = available_checkpoints()
    if not checkpoints:
        st.sidebar.warning("No checkpoints found yet.")
        return "", "cpu", 60, 1.0, 20

    default_checkpoint = "best.pt" if "best.pt" in checkpoints else checkpoints[0]
    checkpoint_name = st.sidebar.selectbox("Checkpoint", checkpoints, index=checkpoints.index(default_checkpoint))
    device_name = st.sidebar.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
    max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=20, max_value=200, value=60, step=10)
    temperature = st.sidebar.slider("Temperature", min_value=0.2, max_value=1.5, value=1.0, step=0.1)
    top_k = st.sidebar.slider("Top K", min_value=0, max_value=50, value=20, step=5)
    return checkpoint_name, device_name, max_new_tokens, temperature, top_k


def main() -> None:
    st.set_page_config(page_title="Mini-GPT Demo", page_icon="GPT", layout="wide")
    st.title("Mini-GPT Demo")
    st.caption("A small Streamlit interface for demonstrating prompt-based text generation.")

    if not VOCAB_PATH.exists():
        st.error("Tokenizer vocabulary not found. Run training once before opening the demo UI.")
        st.code("python training/train.py --epochs 1 --max_steps_per_epoch 20")
        return

    checkpoint_name, device_name, max_new_tokens, temperature, top_k = render_sidebar()

    col1, col2 = st.columns([3, 2])
    with col1:
        prompt = st.text_area(
            "Prompt",
            value="deep learning is",
            height=160,
            help="This text is encoded into tokens and used as the starting context for generation.",
        )
        generate_clicked = st.button("Generate Text", type="primary", use_container_width=True)

        if generate_clicked:
            if not checkpoint_name:
                st.warning("Train the model first so the UI can load a checkpoint.")
            elif not prompt.strip():
                st.warning("Enter a prompt before generating text.")
            else:
                with st.spinner("Generating text..."):
                    generated = generate_text(
                        prompt=prompt,
                        checkpoint_name=checkpoint_name,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        device_name=device_name,
                    )
                st.subheader("Generated Output")
                st.text_area("Result", value=generated, height=220)

    with col2:
        st.subheader("Model Details")
        checkpoints = available_checkpoints()
        st.write(f"Available checkpoints: {len(checkpoints)}")
        if checkpoint_name:
            _, tokenizer, device, config = load_model_bundle(checkpoint_name, device_name)
            st.write(f"Checkpoint: {checkpoint_name}")
            st.write(f"Vocabulary size: {tokenizer.vocab_size}")
            st.write(f"Embedding dim: {config.embedding_dim}")
            st.write(f"Heads: {config.num_heads}")
            st.write(f"Layers: {config.num_layers}")
            st.write(f"Context length: {config.context_length}")
            st.write(f"Device used: {device}")

        st.subheader("Demo Commands")
        st.code("python training/train.py --epochs 1 --max_steps_per_epoch 20", language="bash")
        st.code("streamlit run streamlit_app.py", language="bash")
        st.info("Because the model is intentionally small, short training runs will generate imperfect text. That is expected for a learning demo.")


if __name__ == "__main__":
    main()
