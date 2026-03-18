"""
Standalone QLoRA fine-tuning script for Kubeflow TrainJob.

All configuration is read from environment variables so it can be
driven by the TrainJob spec without code changes.

Environment Variables:
    GOLD_DATA_PATH          S3 path to the gold training JSONL (e.g. s3://bucket/key)
    MODEL_OUTPUT_S3_PATH    S3 destination for the merged model
    BASE_MODEL_ID           HuggingFace model ID (default: unsloth/Llama-3.2-1B-Instruct)
    NUM_EPOCHS              Number of training epochs (default: 3)
    BATCH_SIZE              Per-device train batch size (default: 4)
    LEARNING_RATE           Learning rate (default: 2e-4)
    LORA_R                  LoRA rank (default: 16)
    LORA_ALPHA              LoRA alpha (default: 32)
    S3_ENDPOINT             MinIO / S3 endpoint URL
    S3_ACCESS_KEY           S3 access key
    S3_SECRET_KEY           S3 secret key
"""

import json
import os

import boto3
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def get_env(key: str, default: str = None) -> str:
    val = os.environ.get(key, default)
    if val is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return val


def main():
    gold_data_path = get_env("GOLD_DATA_PATH")
    model_output_s3_path = get_env("MODEL_OUTPUT_S3_PATH")
    base_model_id = get_env("BASE_MODEL_ID", "unsloth/Llama-3.2-1B-Instruct")
    num_epochs = int(get_env("NUM_EPOCHS", "3"))
    batch_size = int(get_env("BATCH_SIZE", "4"))
    learning_rate = float(get_env("LEARNING_RATE", "2e-4"))
    lora_r = int(get_env("LORA_R", "16"))
    lora_alpha = int(get_env("LORA_ALPHA", "32"))
    s3_endpoint = get_env("S3_ENDPOINT")
    s3_access_key = get_env("S3_ACCESS_KEY")
    s3_secret_key = get_env("S3_SECRET_KEY")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    # --- Load gold data from S3 ---
    parts = gold_data_path.replace("s3://", "").split("/", 1)
    bucket, key = parts[0], parts[1]
    obj = s3.get_object(Bucket=bucket, Key=key)
    lines = obj["Body"].read().decode().strip().split("\n")
    records = [json.loads(line) for line in lines]
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} training samples from {gold_data_path}")

    # --- Load base model with 4-bit quantization ---
    print(f"Loading base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 512

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)

    # --- Apply LoRA ---
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # --- Train ---
    training_args = SFTConfig(
        output_dir="/tmp/sft-checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        fp16=False,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.1,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting QLoRA fine-tuning...")
    trainer.train()

    # --- Save LoRA adapter, then merge in full precision ---
    adapter_dir = "/tmp/lora-adapter"
    output_dir = "/tmp/student-merged"
    print("Saving LoRA adapter...")
    model.save_pretrained(adapter_dir)

    print(f"Reloading base model in full precision for merge: {base_model_id}")
    del model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    print("LoRA merged into full-precision model")
    merged_model.save_pretrained(output_dir)

    # Save clean tokenizer and strip fields that break older vLLM/transformers
    clean_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    clean_tokenizer.save_pretrained(output_dir)

    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tc_path, "r") as f:
        tc = json.load(f)
    removed = []
    for field in ["tokenizer_class", "auto_map"]:
        if field in tc:
            removed.append(f"{field}={tc.pop(field)}")
    if removed:
        with open(tc_path, "w") as f:
            json.dump(tc, f, indent=2)
        print(f"Patched tokenizer_config.json: removed {removed}")

    # --- Upload merged model to S3 ---
    dest_parts = model_output_s3_path.replace("s3://", "").split("/", 1)
    dest_bucket, dest_prefix = dest_parts[0], dest_parts[1]

    for root, _, files in os.walk(output_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            s3_key = os.path.join(dest_prefix, os.path.relpath(local_path, output_dir))
            print(f"Uploading {fname} -> s3://{dest_bucket}/{s3_key}")
            s3.upload_file(local_path, dest_bucket, s3_key)

    print(f"Model uploaded to {model_output_s3_path}")


if __name__ == "__main__":
    main()
