# Mini-GPT From Scratch

Mini-GPT is a small GPT-style language model built in plain PyTorch for learning and demo purposes. The code is intentionally simple: a character-level tokenizer, readable causal self-attention, a compact transformer stack, and straightforward training and generation scripts.

This repository is a good fit if you want to explain how a GPT model works in a viva, demo, or classroom setting without hiding the logic behind a large framework.

## What This Project Covers

- Character-level tokenization
- Token and positional embeddings
- Causal multi-head self-attention
- Transformer blocks with residual connections
- Next-token training with cross-entropy loss
- Autoregressive text generation
- Checkpoint saving and loading

## Project Structure

```text
Mini-GPT/
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset.py
├── models/
│   ├── attention.py
│   ├── embedding.py
│   ├── gpt_model.py
│   └── transformer_block.py
├── inference/
│   ├── generate.py
│   └── sampler.py
├── tokenizer/
│   ├── tokenizer.py
│   ├── vocab.json
│   └── vocab.py
├── training/
│   ├── loss.py
│   ├── train.py
│   └── trainer.py
├── utils/
│   ├── checkpoint.py
│   ├── device.py
│   ├── logger.py
│   └── seed.py
├── tests/
│   ├── test_model.py
│   ├── test_tokenizer.py
│   └── test_train_cli.py
├── scripts/
│   ├── download_data.sh
│   ├── run_training.bat
│   └── run_training.sh
├── notebooks/
│   └── training_demo.ipynb
├── experiments/
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## How The Model Works

### 1. Tokenizer

The tokenizer in `tokenizer/tokenizer.py` is character-based.

- It scans the training text.
- It builds a vocabulary of unique characters.
- It converts text into integer token ids.
- It converts token ids back into text.

This is simpler than BPE or WordPiece, which makes it easier to explain in a demo.

### 2. Dataset

The dataset in `data/dataset.py` creates training examples for next-token prediction.

If the context length is 128, each sample looks like this:

- Input: 128 tokens
- Target: the same sequence shifted by 1 token

That is how GPT learns to predict the next character.

### 3. Embeddings

The embedding layer in `models/embedding.py` creates:

- Token embeddings: meaning for each token id
- Positional embeddings: position information for each token in the sequence

The model adds them together before sending them into the transformer blocks.

### 4. Attention

The attention layer in `models/attention.py` implements causal self-attention.

- Each token can attend only to itself and previous tokens.
- A lower-triangular mask blocks future tokens.
- Attention scores are converted into probabilities with softmax.
- Weighted values are combined to produce the output.

This is the core idea behind GPT.

### 5. Transformer Block

Each transformer block in `models/transformer_block.py` contains:

- Layer normalization
- Multi-head causal self-attention
- Feed-forward network
- Residual connections

This block is repeated multiple times in the full model.

### 6. Full GPT Model

The full model in `models/gpt_model.py` does three main jobs:

- Build embeddings
- Pass them through transformer blocks
- Convert final hidden states into vocabulary logits

During training, it also computes cross-entropy loss.

During generation, it repeatedly predicts one next token at a time.

## Installation

Create and activate a virtual environment, then install requirements.

### Windows

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Linux or macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training

The training entrypoint is `training/train.py`.

If `data/raw/input.txt` does not exist, the script automatically creates a tiny demo dataset so the project runs out of the box.

### Basic training

```bash
python training/train.py
```

### Quick demo training

Use a short run when you need a fast classroom demo.

```bash
python training/train.py --epochs 1 --max_steps_per_epoch 20 --batch_size 16 --log_interval 5 --eval_interval 10
```

### Windows helper script

```bat
scripts\run_training.bat --epochs 1 --max_steps_per_epoch 20
```

### Bash helper script

```bash
./scripts/run_training.sh --epochs 1 --max_steps_per_epoch 20
```

### Useful CLI options

You can override YAML settings from the command line.

```bash
python training/train.py \
	--epochs 2 \
	--batch_size 16 \
	--learning_rate 0.001 \
	--context_length 64 \
	--embedding_dim 64 \
	--num_layers 2 \
	--device cpu
```

Available training overrides:

- `--epochs`
- `--batch_size`
- `--learning_rate`
- `--eval_interval`
- `--log_interval`
- `--grad_clip`
- `--train_split`
- `--device`
- `--max_steps_per_epoch`

Available model overrides:

- `--embedding_dim`
- `--num_heads`
- `--num_layers`
- `--context_length`
- `--dropout`

Other useful arguments:

- `--data_path`
- `--model_config`
- `--training_config`
- `--checkpoint_dir`
- `--log_file`
- `--skip_demo_data`

## Text Generation

After training, generate text with:

```bash
python inference/generate.py --prompt "deep learning is" --max_new_tokens 80 --checkpoint experiments/checkpoints/best.pt
```

Example output from a tiny demo run:

```text
Prompt: deep learning is
Generated Text:
deep learning is fun. transfors arere powerful motalsas
```

The output is imperfect because the model is intentionally small and the demo dataset is tiny. That is normal for this learning project.

## Streamlit Demo UI

The repository also includes a simple web UI in `streamlit_app.py` so you can demo the model without using only the terminal.

Start the UI with:

```bash
streamlit run streamlit_app.py
```

Or use the helper scripts:

```bat
scripts\run_demo_ui.bat
```

```bash
./scripts/run_demo_ui.sh
```

The UI lets you:

- choose a checkpoint
- enter a prompt
- adjust temperature, top-k, and output length
- generate text in the browser

Before using the UI, run training at least once so `tokenizer/vocab.json` and a checkpoint file exist.

## Configuration Files

### `configs/model_config.yaml`

Default model settings:

```yaml
vocab_size: 0
embedding_dim: 128
num_heads: 4
num_layers: 4
context_length: 128
dropout: 0.1
```

`vocab_size` is filled automatically from the tokenizer.

### `configs/training_config.yaml`

Default training settings:

```yaml
seed: 42
batch_size: 32
learning_rate: 0.0003
epochs: 5
eval_interval: 200
log_interval: 50
grad_clip: 1.0
train_split: 0.9
device: auto
max_steps_per_epoch: null
```

## Output Files

Training creates or updates these artifacts:

- `tokenizer/vocab.json`
- `experiments/logs/train.log`
- `experiments/checkpoints/best.pt`
- `experiments/checkpoints/final.pt`

## Demo Flow

If you need to explain the project in 5 to 7 minutes, this order works well:

1. Start with `tokenizer/tokenizer.py` and explain how text becomes numbers.
2. Show `data/dataset.py` and explain input-target shifting.
3. Show `models/embedding.py` and explain token plus position embeddings.
4. Show `models/attention.py` and explain causal masking.
5. Show `models/transformer_block.py` and explain the repeated block structure.
6. Show `models/gpt_model.py` and explain forward pass plus generation loop.
7. Show `training/train.py` and `training/trainer.py` and explain the training flow.
8. End with `inference/generate.py` and run one prompt live.

## Tests

Run the tests with:

```bash
pytest -q
```

## Future Improvements

- Add a subword tokenizer such as BPE
- Train on a larger dataset
- Add mixed precision support
- Add a small web interface
- Add saveable experiment metadata

## License

This project is available under the MIT License.