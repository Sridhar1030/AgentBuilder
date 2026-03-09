"""
QLoRA Fine-Tuning Script — runs inside a RayJob on RHOAI.

Reads gold training data, fine-tunes the 1B student with QLoRA,
merges the adapters, and pushes the merged model to S3.
"""

import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "unsloth/Llama-3.2-1B-Instruct")
GOLD_DATA_PATH = os.getenv("GOLD_DATA_PATH", "/data/gold_pairs.jsonl")
OUTPUT_DIR = os.getenv("OUTPUT_MODEL_PATH", "/tmp/student-merged")


def load_data(path: str) -> Dataset:
    if path.startswith("s3://"):
        import boto3
        parts = path.replace("s3://", "").split("/", 1)
        bucket, key = parts[0], parts[1]
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        lines = obj["Body"].read().decode().strip().split("\n")
        records = [json.loads(line) for line in lines]
    else:
        with open(path, "r") as f:
            content = f.read().strip()
            if content.startswith("["):
                records = json.loads(content)
            else:
                records = [json.loads(line) for line in content.split("\n")]
    return Dataset.from_list(records)


def main():
    print(f"Loading base model: {BASE_MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    print(f"Loading gold data from: {GOLD_DATA_PATH}")
    dataset = load_data(GOLD_DATA_PATH)
    print(f"Training on {len(dataset)} samples")

    training_args = SFTConfig(
        output_dir="/tmp/sft-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        warmup_ratio=0.1,
        dataset_text_field="text",
        max_seq_length=512,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    print("Merging LoRA adapters into base model...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if OUTPUT_DIR.startswith("s3://") or os.getenv("PUSH_TO_S3", "false") == "true":
        upload_to_s3(OUTPUT_DIR)

    print("Fine-tuning complete!")


def upload_to_s3(local_dir: str):
    import boto3
    s3_path = os.getenv("OUTPUT_MODEL_PATH", "")
    if not s3_path.startswith("s3://"):
        return
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, prefix = parts[0], parts[1]
    s3 = boto3.client("s3")
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            key = os.path.join(prefix, os.path.relpath(local_path, local_dir))
            print(f"Uploading {local_path} -> s3://{bucket}/{key}")
            s3.upload_file(local_path, bucket, key)


if __name__ == "__main__":
    main()
