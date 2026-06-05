# Financial Sentiment Classification with QLoRA

Fine-tuning **Qwen2.5-0.5B-Instruct** with QLoRA to classify financial news
sentiment as negative / neutral / positive. Full workflow: baseline → LoRA training
→ honest before/after evaluation on a held-out test set.

## Results

| Model                 | Accuracy | Macro-F1 |
|-----------------------|:--------:|:--------:|
| Qwen2.5-0.5B (base)   |  0.422   |  0.479   |
| Qwen2.5-0.5B + LoRA   | **0.792**| **0.793**|

*20% held-out split, fixed seed (no leakage). Accuracy nearly doubled; gains are
balanced across all three classes.*

## Method

QLoRA (4-bit NF4 base + LoRA adapters), classification framed as instruction tuning.
Dataset: financial_phrasebank (~4.8k labelled sentences).

Key hyperparameters: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, lr `2e-4`,
3 epochs, effective batch 16.

## Usage

```bash
pip install -r requirements.txt
python train.py       # trains + saves the LoRA adapter
python evaluate.py    # base vs LoRA -> results/metrics.json
```

Needs a CUDA GPU (free Colab/Kaggle T4 works).

**Stack:** transformers · peft · bitsandbytes · datasets · scikit-learn · torch
