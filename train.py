"""Fine-tune Qwen2.5-0.5B with QLoRA on financial sentiment classification.
 
Run on a CUDA GPU (free Colab/Kaggle T4 works). Saves a LoRA adapter to OUTPUT_DIR.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use GPU 0

from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,prepare_model_for_kbit_training


# Hyperparameters
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "qwen-fin-sentiment-lora"
LABELS = ["negative", "neutral", "positive"]


def main(): 
    #data
    ds = load_dataset("descartes100/enhanced-financial-phrasebank")
    ds = ds["train"].flatten() # the dataset has a nested structure, flatten it for easier access 
    ds = ds.rename_columns({"train.sentence": "sentence", "train.label": "label"})
    ds = ds.train_test_split(test_size=0.2, seed=42)  

    #tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None: # some models don't have a pad token, set it to eos token
        tokenizer.pad_token = tokenizer.eos_token

    def preprocess(example: dict) -> dict:
        messages = [
            {"role": "user", "content":
                "Classify the sentiment of this financial sentence as "
                "negative, neutral, or positive. Answer with one word only.\n\n"
                f"Sentence: {example['sentence']}"},
            {"role": "assistant", "content": LABELS[int(example["label"])]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        return tokenizer(text, truncation=True, max_length=256)

    train_ds = ds["train"].map(preprocess, remove_columns=ds["train"].column_names)


    #model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)


    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # rank
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # should show only LoRA params as trainable 


    # Training arguments
    args= TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}")



if __name__ == "__main__":
    main()



