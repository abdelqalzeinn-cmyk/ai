# Tiny Chat 42M

A from-scratch, decoder-only Transformer chatbot using PyTorch. The default model has about 42 million trainable parameters and uses a compact byte-level BPE tokenizer with UTF-8 fallback.

The model uses mobile-oriented ideas found in efficient Gemma-style designs: RMSNorm, rotary position embeddings, SwiGLU, grouped-query attention with 2 key/value heads, tied input/output embeddings, and a KV cache during generation. This is not Gemma 4 code or a claim of matching Gemma quality; it is a compact, trainable architecture using similar efficiency principles.

## 1. Create the environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell blocks activation, run the project commands with `.venv\Scripts\python.exe` instead.

## 2. Download and prepare Hugging Face data

This uses the public `HuggingFaceH4/ultrachat_200k` `train_sft` split and converts its messages to `User:` / `Assistant:` training text:

```powershell
python -m tiny_chat.data
```

To publish the prepared rows as your own Hugging Face dataset, authenticate with `huggingface-cli login`, then run:

```powershell
python -m tiny_chat.data --hub-repo YOUR_USERNAME/tiny-chat-data
```

Check the dataset license and Hugging Face terms before redistributing a derivative dataset.

## 3. Run a smoke test

```powershell
python -m pytest -q
```

## 4. Train

For a quick test:

```powershell
python -m tiny_chat.train --steps 50 --batch-size 8 --max-chars 100000
```

For a longer run, increase `--steps`. By default, local training uses the first 100,000 characters so startup is fast. For a larger training corpus, use `--max-chars 20000000`; use `--max-chars 0` only when you have enough RAM for the complete corpus. The BPE tokenizer and encoded token IDs are cached under `data/processed/.cache`, so repeating the same command starts much faster. A CUDA GPU is strongly recommended; CPU training is useful only for smoke tests. The script writes checkpoints under `checkpoints/`.

For phone deployment, keep the context window small, run the model with a mobile PyTorch/ExecuTorch build, and apply post-training weight-only quantization (INT8 or 4-bit). The KV cache avoids recomputing the full prompt on every generated token, while grouped-query attention reduces the cache size by 4x compared with 8 key/value heads.

## 5. Chat

```powershell
python -m tiny_chat.chat
```

The first model will be a learning/demo chatbot, not a production assistant. Quality depends heavily on dataset size, cleaning, training tokens, and hardware. To make it safer and more useful, add a curated instruction dataset and evaluation set after the baseline works.
