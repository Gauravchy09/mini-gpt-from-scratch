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
PREFERRED_EXPORT_NAMES = ["run_03", "run_02", "run_01"]
TINY_SHAKESPEARE_CHECKPOINTS = ["mini_gpt_tinyshakespeare.pt", "mini_gpt_tinyshakesp.pt"]

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

    # Prefer run_03, then run_02, then run_01 when available.
    priority = {name: index for index, name in enumerate(PREFERRED_EXPORT_NAMES)}
    unique.sort(key=lambda path: (priority.get(path.name, len(priority)), str(path)), reverse=False)
    return unique


def infer_model_label(path: Path, kind: str) -> str:
    lower_name = path.name.lower()

    if kind == "checkpoint" and "shakes" in lower_name:
        return "Shakespeare Model"

    if kind == "export":
        if path.name == "run_03":
            return "QA Model"
        if path.name == "run_01":
            return "Story Model"
        if path.name == "run_02":
            return "Base Model"

    return path.stem if kind == "checkpoint" else path.name


def build_model_options() -> list[dict[str, str]]:
    options: list[dict[str, str]] = []

    for export_path in find_artifact_dirs():
        options.append(
            {
                "kind": "export",
                "path": str(export_path),
                "label": infer_model_label(export_path, "export"),
            }
        )

    for filename in TINY_SHAKESPEARE_CHECKPOINTS:
        checkpoint_path = ROOT_DIR / filename
        if checkpoint_path.exists():
            options.append(
                {
                    "kind": "checkpoint",
                    "path": str(checkpoint_path),
                    "label": infer_model_label(checkpoint_path, "checkpoint"),
                }
            )

    return options


@st.cache_data(show_spinner=False)
def get_model_summary(export_dir: str) -> dict[str, Any]:
    config_path = Path(export_dir) / "mini_gpt_config.json"
    with config_path.open("r", encoding="utf-8") as file:
        model_cfg = json.load(file)

    config = GPTConfig(**model_cfg)
    param_count = sum(param.numel() for param in MiniGPT(config).parameters())
    return {
        "param_count": param_count,
    }


@st.cache_data(show_spinner=False)
def get_checkpoint_summary(checkpoint_path: str) -> dict[str, Any]:
    loaded = torch.load(checkpoint_path, map_location="cpu")
    state_dict = loaded.get("model_state_dict", loaded) if isinstance(loaded, dict) else loaded
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format")

    param_count = sum(tensor.numel() for tensor in state_dict.values() if torch.is_tensor(tensor))
    return {
        "param_count": int(param_count),
    }


def format_param_count(param_count: int) -> str:
    if param_count >= 1_000_000:
        return f"~{param_count / 1_000_000:.1f}M"
    if param_count >= 1_000:
        return f"~{param_count / 1_000:.1f}K"
    return str(param_count)


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


def remap_tiny_shakespeare_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("token_embed."):
            new_key = new_key.replace("token_embed.", "embedding.token_embedding.", 1)
        elif new_key.startswith("pos_embed."):
            new_key = new_key.replace("pos_embed.", "embedding.position_embedding.", 1)
        elif ".attn.qkv." in new_key:
            new_key = new_key.replace(".attn.qkv.", ".attn.qkv_proj.")
        elif ".attn.proj." in new_key:
            new_key = new_key.replace(".attn.proj.", ".attn.out_proj.")
        elif new_key.startswith("ln_f."):
            new_key = new_key.replace("ln_f.", "final_ln.", 1)

        remapped[new_key] = value

    return remapped


def normalize_itos(raw_itos: Any) -> dict[int, str]:
    if not isinstance(raw_itos, dict):
        raise ValueError("Checkpoint does not contain valid itos mapping")
    return {int(key): str(value) for key, value in raw_itos.items()}


def encode_with_vocab(text: str, stoi: dict[str, int]) -> list[int]:
    ids = [stoi[ch] for ch in text if ch in stoi]
    if ids:
        return ids

    # Fallback when prompt has no known characters.
    fallback = stoi.get("\n")
    if fallback is not None:
        return [fallback]
    return [next(iter(stoi.values()))]


def decode_with_vocab(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos.get(int(token_id), "") for token_id in ids)


@st.cache_resource(show_spinner=False)
def load_model_bundle(
    model_kind: str,
    model_path: str,
    device_name: str,
) -> tuple[MiniGPT, Any, torch.device, GPTConfig, dict[str, Any]]:
    device = get_device(device_name)

    if model_kind == "export":
        export_path = Path(model_path)
        state_path = export_path / "mini_gpt_state.pt"
        config_path = export_path / "mini_gpt_config.json"
        tokenizer_path = export_path / "tokenizer" / "tokenizer.json"

        with config_path.open("r", encoding="utf-8") as file:
            model_cfg = json.load(file)
        config = GPTConfig(**model_cfg)

        tokenizer = Tokenizer.from_file(tokenizer_path.as_posix())
        model = MiniGPT(config)

        loaded = torch.load(state_path, map_location=device)
        state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
        state_dict = remap_state_dict_keys(state_dict)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        load_report = {
            "missing": missing,
            "unexpected": unexpected,
            "model_path": str(state_path),
        }
        return model, tokenizer, device, config, load_report

    if model_kind == "checkpoint":
        checkpoint = torch.load(model_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("Checkpoint format is not supported for UI loading.")

        vocab_size = int(checkpoint["vocab_size"])
        context_length = int(checkpoint["block_size"])
        num_layers = int(checkpoint["n_layer"])
        num_heads = int(checkpoint["n_head"])
        embedding_dim = int(checkpoint["n_embd"])

        config = GPTConfig(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            context_length=context_length,
            dropout=0.1,
        )

        model = MiniGPT(config)
        state_dict = remap_tiny_shakespeare_keys(checkpoint["model_state_dict"])
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        stoi = checkpoint.get("stoi")
        itos = normalize_itos(checkpoint.get("itos"))
        if not isinstance(stoi, dict):
            raise ValueError("Checkpoint does not contain valid stoi mapping")
        vocab_bundle = {"stoi": {str(key): int(value) for key, value in stoi.items()}, "itos": itos}

        load_report = {
            "missing": missing,
            "unexpected": unexpected,
            "model_path": model_path,
        }
        return model, vocab_bundle, device, config, load_report

    raise ValueError(f"Unsupported model kind: {model_kind}")


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
    model_kind: str,
    model_path: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    device_name: str,
) -> str:
    model, tokenizer_like, device, _, _ = load_model_bundle(model_kind, model_path, device_name)

    if model_kind == "export":
        encoded = tokenizer_like.encode(prompt)
        input_ids = encoded.ids
    else:
        stoi = tokenizer_like["stoi"]
        input_ids = encode_with_vocab(prompt, stoi)

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

    output_ids = x[0].tolist()
    if model_kind == "export":
        return tokenizer_like.decode(output_ids, skip_special_tokens=True)

    itos = tokenizer_like["itos"]
    return decode_with_vocab(output_ids, itos)


def render_sidebar(
    model_options: list[dict[str, str]],
) -> tuple[dict[str, str], str, int, float, int, float, float]:
    st.sidebar.header("Hyperparameters")

    if not model_options:
        st.sidebar.warning("No exported model folder found.")
        return {}, "cpu", DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, DEFAULT_REPETITION_PENALTY

    preferred_index = 0
    for preferred_name in PREFERRED_EXPORT_NAMES:
        match_index = next(
            (
                idx
                for idx, option in enumerate(model_options)
                if option["kind"] == "export" and option["path"].replace("\\", "/").endswith(f"/{preferred_name}")
            ),
            None,
        )
        if match_index is not None:
            preferred_index = match_index
            break

    selected_index = st.sidebar.selectbox(
        "Model",
        list(range(len(model_options))),
        index=preferred_index,
        format_func=lambda idx: f"{model_options[idx]['label']} ({Path(model_options[idx]['path']).name})",
    )
    selected_model = model_options[selected_index]

    device_name = st.sidebar.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        index=0,
        help="auto: uses CUDA GPU when available, else CPU. cuda: force GPU (falls back to CPU if unavailable). cpu: always run on CPU.",
    )
    st.sidebar.caption("Use auto for most cases. Choose cpu only if GPU memory is limited or CUDA is unstable.")
    max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=20, max_value=240, value=DEFAULT_MAX_NEW_TOKENS, step=10)
    temperature = st.sidebar.slider("Temperature", min_value=0.2, max_value=1.5, value=DEFAULT_TEMPERATURE, step=0.05)
    top_k = st.sidebar.slider("Top K", min_value=0, max_value=100, value=DEFAULT_TOP_K, step=5)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=DEFAULT_TOP_P, step=0.01)
    repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=1.5, value=DEFAULT_REPETITION_PENALTY, step=0.01)
    return selected_model, device_name, max_new_tokens, temperature, top_k, top_p, repetition_penalty


def main() -> None:
    st.set_page_config(page_title="Mini-GPT Demo", page_icon="GPT", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1100px;
            padding-top: 1.2rem;
            padding-bottom: 1.2rem;
        }
        .model-pill {
            border: 1px solid rgba(49, 51, 63, 0.2);
            border-radius: 14px;
            padding: 0.8rem 1rem;
            margin-bottom: 0.8rem;
            background: rgba(255, 255, 255, 0.03);
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 0.8rem;
                padding-right: 0.8rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Mini-GPT Streamlit Demo")
    st.caption("Simple frontend for prompt-based text generation using your trained run exports.")

    model_options = build_model_options()
    if not model_options:
        st.error("No exported model found. Expected a folder containing mini_gpt_state.pt.")
        st.write("Export folders searched under project root, for example:")
        st.code("run_02-*/run_02")
        return

    selected_model, device_name, max_new_tokens, temperature, top_k, top_p, repetition_penalty = render_sidebar(model_options)
    model_kind = selected_model.get("kind", "")
    model_path = selected_model.get("path", "")

    model_folder_name = Path(model_path).name if model_path else "N/A"
    model_label = selected_model.get("label", model_folder_name)
    model_params_text = "~20M"
    if model_path:
        try:
            summary = get_model_summary(model_path) if model_kind == "export" else get_checkpoint_summary(model_path)
            model_params_text = format_param_count(summary["param_count"])
        except Exception:
            model_params_text = "~20M"

    st.markdown(
        f"""
        <div class="model-pill">
            <strong>Model:</strong> {model_label} &nbsp; | &nbsp; <strong>Source:</strong> {model_folder_name} &nbsp; | &nbsp; <strong>Parameters:</strong> {model_params_text}
        </div>
        """,
        unsafe_allow_html=True,
    )

    default_prompt = "### Instruction:\nWhat is machine learning?\n\n### Response:\n"
    if "story" in model_label.lower():
        default_prompt = "Once upon a time,"
    if "shakespeare" in model_label.lower():
        default_prompt = "ROMEO:\n"

    prompt = st.text_area(
        "Prompt",
        value=default_prompt,
        height=170,
    )
    generate_clicked = st.button("Generate Text", type="primary", use_container_width=True)

    if model_path:
        try:
            _, _, resolved_device, _, _ = load_model_bundle(model_kind, model_path, device_name)
            st.caption(f"Running on: {resolved_device.type.upper()} (selected: {device_name})")
        except Exception:
            st.caption(f"Selected device mode: {device_name}")

    if generate_clicked:
        if not model_path:
            st.warning("Select a trained model from the sidebar.")
        elif not prompt.strip():
            st.warning("Enter a prompt before generating text.")
        else:
            with st.spinner("Generating text..."):
                generated = generate_text(
                    prompt=prompt,
                    model_kind=model_kind,
                    model_path=model_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    device_name=device_name,
                )
            st.subheader("Generated Output")
            st.text_area("Result", value=generated, height=260)


if __name__ == "__main__":
    main()
