# Mini-GPT From Scratch

A lightweight implementation of a GPT-style language model built from scratch using **PyTorch**.  
This project is designed to understand the core components of **transformer-based language models**, including tokenization, self-attention, transformer blocks, training pipelines, and autoregressive text generation.

The project follows a **production-style machine learning repository structure** and can be trained using **Google Colab GPUs**.

---

# Project Goals

- Understand how GPT-style language models work internally
- Implement the Transformer architecture from scratch
- Train a small language model using PyTorch
- Generate text using autoregressive decoding
- Build a clean and modular deep learning project structure

---

# Features

- Transformer architecture implementation
- Multi-head self-attention
- Token and positional embeddings
- Autoregressive text generation
- Training pipeline with checkpoints
- Modular and production-style code structure
- Compatible with Google Colab GPU training

---

# Project Structure

```
mini-gpt-from-scratch/
│
├── configs/
│   ├── model_config.yaml
│   └── training_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── dataset.py
│
├── models/
│   ├── attention.py
│   ├── transformer_block.py
│   ├── embedding.py
│   └── gpt_model.py
│
├── training/
│   ├── train.py
│   ├── trainer.py
│   └── loss.py
│
├── inference/
│   ├── generate.py
│   └── sampler.py
│
├── tokenizer/
│   ├── tokenizer.py
│   └── vocab.py
│
├── utils/
│   ├── logger.py
│   ├── seed.py
│   ├── checkpoint.py
│   └── device.py
│
├── experiments/
│   ├── logs/
│   ├── checkpoints/
│   └── outputs/
│
├── notebooks/
│   └── training_demo.ipynb
│
├── tests/
│   ├── test_model.py
│   └── test_tokenizer.py
│
├── scripts/
│   ├── download_data.sh
│   └── run_training.sh
│
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

# Core Components

## Tokenizer

The tokenizer converts raw text into numerical tokens that the model can process.

Responsibilities:

- Vocabulary creation
- Text encoding
- Token decoding

Location:

```
tokenizer/tokenizer.py
```

---

## Transformer Model

The GPT architecture consists of multiple transformer blocks.

Components include:

- Token embeddings
- Positional embeddings
- Multi-head self-attention
- Feed-forward networks
- Residual connections
- Layer normalization

Location:

```
models/gpt_model.py
```

---

## Training Pipeline

The training pipeline manages the model training process.

Responsibilities:

- Data loading
- Loss computation
- Gradient updates
- Model checkpointing
- Training logs

Location:

```
training/train.py
```

---

## Inference

After training, the model can generate text using autoregressive decoding.

Example:

```
Input Prompt:
Deep learning is

Output:
Deep learning is transforming artificial intelligence by enabling machines to learn complex patterns from data.
```

Location:

```
inference/generate.py
```

---

# Installation

Clone the repository:

```
git clone https://github.com/your-username/mini-gpt-from-scratch.git
cd mini-gpt-from-scratch
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Training the Model

Run the training script:

```
python training/train.py
```

Training outputs will be saved inside:

```
experiments/checkpoints
experiments/logs
```

---

# Generating Text

After training, generate text using:

```
python inference/generate.py
```

Example:

```
Prompt: Artificial intelligence

Generated Text:
Artificial intelligence is rapidly evolving and shaping the future of technology.
```

---

# Configuration

Model and training parameters are defined inside configuration files.

## Model Config

```
configs/model_config.yaml
```

Example parameters:

```
vocab_size: 5000
embedding_dim: 256
num_heads: 4
num_layers: 4
context_length: 128
dropout: 0.1
```

---

## Training Config

```
configs/training_config.yaml
```

Example parameters:

```
batch_size: 32
learning_rate: 0.0003
epochs: 10
device: cuda
```

---

# Experiments

All experiment outputs are stored inside the experiments folder.

```
experiments/
 ├── logs/
 ├── checkpoints/
 └── outputs/
```

This helps track training runs and generated outputs.

---

# Running in Google Colab

This project can be trained using Google Colab GPUs.

Steps:

1. Upload the repository to Colab
2. Install dependencies
3. Run the training script

Notebook location:

```
notebooks/training_demo.ipynb
```

---

# Technologies Used

- Python
- PyTorch
- NumPy
- YAML
- Google Colab GPU

---

# Learning Outcomes

This project helps understand:

- Transformer architecture
- Self-attention mechanism
- Language model training
- Text generation
- Deep learning project structuring

---

# Future Improvements

- Byte Pair Encoding tokenizer
- Larger datasets
- Flash Attention
- Mixed precision training
- Web interface for text generation
- Fine-tuning support

---

# License

This project is open-source and available under the MIT License.

---

# Author

Gaurav Kumar Choudhary  
Computer Science Student | AI Systems | Deep Learning | LLMs