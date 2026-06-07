"""Compare base vs LoRA-tuned Qwen on the held-out test set.
 
Produces the before/after accuracy + macro-F1 table and writes results/metrics.json.
"""

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, f1_score
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = "qwen-fin-sentiment-lora"
LABELS = ["negative", "neutral", "positive"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def build_prompt(sentence: str) -> str:
    messages = [{"role": "user", "content":
        "Classify the sentiment of this financial sentence as "
        "negative, neutral, or positive. Answer with one word only.\n\n"
        f"Sentence: {sentence}"}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def parse_label(text: str) -> int:
    text = text.lower()
    for i,lab in enumerate(LABELS):
        if lab in text:
            return i
    return 1  # return 1 if no label found

def evaluate(model: PeftModel, test_ds: list, name: str, n: int = 400)->dict:
    preds, golds = [], []
    for ex in test_ds.select(range(min(n,len(test_ds)))):
        inp = tokenizer(build_prompt(ex["sentence"]), return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=5,do_sample=False)
        gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        preds.append(parse_label(gen))
        golds.append(int(ex["label"]))
    acc = round(accuracy_score(golds, preds), 4)
    f1 = round(f1_score(golds, preds, average="macro"), 4)
    print(f"[{name}] accuracy={acc}  macro-F1={f1}")
    return {"accuracy": acc, "macro_f1": f1}

def main():
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")
    ds = ds["train"].flatten()
    ds = ds.rename_columns({"train.sentence": "sentence", "train.label": "label"})
    test_ds = ds.train_test_split(test_size=0.2, seed=42)["test"]  # same split, no leakage
 
    # base model (fp16 inference - no quantization needed for a 0.5B model)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map={"": 0},
    )
    base_metrics = evaluate(base, test_ds, "base")
 
    # LoRA-tuned
    tuned = PeftModel.from_pretrained(base, ADAPTER_DIR)
    lora_metrics = evaluate(tuned, test_ds, "lora")
 
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump({"base": base_metrics, "lora": lora_metrics}, f, indent=2)
    print("Saved results/metrics.json")
 
 
if __name__ == "__main__":
    main()
 