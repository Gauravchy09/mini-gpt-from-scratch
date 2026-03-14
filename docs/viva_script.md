# 2-Minute Viva Script

## Short Script

Hello everyone. This project is a small GPT-style language model built from scratch in PyTorch. The main goal was not to build a state-of-the-art model, but to understand the complete pipeline of how GPT works in a simple and readable way.

I start with [tokenizer/tokenizer.py](../tokenizer/tokenizer.py). This file uses a character-level tokenizer. It reads text, builds a vocabulary of unique characters, and converts characters into integer token ids. I kept this simple because it is easy to explain in a demo.

Next, in [data/dataset.py](../data/dataset.py), I create training samples for next-token prediction. Each input sequence is paired with a target sequence shifted by one token. So the model learns to predict the next character at every position.

Then I move to the model files. In [models/embedding.py](../models/embedding.py), token embeddings and positional embeddings are added together. In [models/attention.py](../models/attention.py), I implement causal self-attention. The important idea here is the causal mask, which prevents the model from seeing future tokens. In [models/transformer_block.py](../models/transformer_block.py), I combine attention, feed-forward layers, normalization, and residual connections into one transformer block.

The complete model is in [models/gpt_model.py](../models/gpt_model.py). It stacks multiple transformer blocks, produces logits over the vocabulary, computes training loss, and also contains the autoregressive generate function.

For training, [training/train.py](../training/train.py) loads configs, builds the tokenizer and dataset, creates the model, and starts the trainer. [training/trainer.py](../training/trainer.py) handles the training loop, validation, and checkpoint saving.

Finally, [inference/generate.py](../inference/generate.py) and [streamlit_app.py](../streamlit_app.py) are used for demo output. After training, I can enter a prompt and the model generates text one token at a time.

So overall, this project shows the complete GPT pipeline: tokenization, dataset creation, transformer model, training, checkpointing, and generation, all in a small codebase that is easy to study.

## 1-Line Backup Answers

- Why character-level tokenization: it is simpler to implement and easier to explain.
- Why causal mask: it stops the model from cheating by seeing future tokens.
- Why shifted targets: GPT is trained as a next-token prediction model.
- Why positional embeddings: attention alone does not know token order.
- Why generated text looks rough: the model is very small and trained on a tiny dataset.
