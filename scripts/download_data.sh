#!/usr/bin/env bash
set -e

mkdir -p data/raw
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o data/raw/input.txt
echo "Downloaded dataset to data/raw/input.txt"
