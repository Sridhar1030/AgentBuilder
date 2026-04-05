"""
Standalone QLoRA fine-tuning script for Kubeflow TrainJob.
Supports both SFT and DPO training modes.

All configuration is read from environment variables so it can be
driven by the TrainJob spec without code changes.

Environment Variables:
    TRAINING_MODE           "sft" (default) or "dpo"
    GOLD_DATA_PATH          S3 path to the gold training JSONL (SFT mode)
    PREF_DATA_PATH          S3 path to preference JSONL (DPO mode)
    MODEL_OUTPUT_S3_PATH    S3 destination for the merged model
    BASE_MODEL_ID           HuggingFace model ID or S3 path to SFT model (DPO mode)
    NUM_EPOCHS              Number of training epochs (default: 3)
    BATCH_SIZE              Per-device train batch size (default: 4)
    LEARNING_RATE           Learning rate (default: 2e-4 for SFT, 5e-5 for DPO)
    LORA_R                  LoRA rank (default: 16)
    LORA_ALPHA              LoRA alpha (default: 32)
    DPO_BETA                DPO beta parameter (default: 0.1)
    S3_ENDPOINT             MinIO / S3 endpoint URL
    S3_ACCESS_KEY           S3 access key
    S3_SECRET_KEY           S3 secret key
"""

import json
import os
import shutil

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


def build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=get_env("S3_ENDPOINT"),
        aws_access_key_id=get_env("S3_ACCESS_KEY"),
        aws_secret_access_key=get_env("S3_SECRET_KEY"),
    )


def load_s3_jsonl(s3, s3_path: str) -> list[dict]:
    parts = s3_path.replace("s3://", "").split("/", 1)
    obj = s3.get_object(Bucket=parts[0], Key=parts[1])
    lines = obj["Body"].read().decode().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def download_s3_dir(s3, s3_path: str, local_dir: str):
    """Download an entire S3 prefix to a local directory."""
    parts = s3_path.replace("s3://", "").split("/", 1)
    bucket, prefix = parts[0], parts[1].rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel:
                continue
            local_path = os.path.join(local_dir, rel)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(bucket, key, local_path)
            print(f"  Downloaded {key} -> {local_path}")


def upload_model_to_s3(s3, output_dir: str, model_output_s3_path: str):
    dest_parts = model_output_s3_path.replace("s3://", "").split("/", 1)
    dest_bucket, dest_prefix = dest_parts[0], dest_parts[1]
    for root, _, files in os.walk(output_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            s3_key = os.path.join(dest_prefix, os.path.relpath(local_path, output_dir))
            print(f"Uploading {fname} -> s3://{dest_bucket}/{s3_key}")
            s3.upload_file(local_path, dest_bucket, s3_key)
    print(f"Model uploaded to {model_output_s3_path}")


def load_model_and_tokenizer(model_id: str):
    """Load a model in 4-bit QLoRA mode with tokenizer."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model = prepare_model_for_kbit_training(model)
    return model, tokenizer


def merge_and_save(model, base_model_id: str, output_dir: str):
    """Save LoRA adapter, reload full-precision base, merge, patch tokenizer."""
    adapter_dir = "/tmp/lora-adapter"
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


# =========================================================================
# SFT Training
# =========================================================================

def run_sft(s3):
    gold_data_path = get_env("GOLD_DATA_PATH")
    model_output_s3_path = get_env("MODEL_OUTPUT_S3_PATH")
    base_model_id = get_env("BASE_MODEL_ID", "unsloth/Llama-3.2-1B-Instruct")
    num_epochs = int(get_env("NUM_EPOCHS", "3"))
    batch_size = int(get_env("BATCH_SIZE", "4"))
    learning_rate = float(get_env("LEARNING_RATE", "2e-4"))
    lora_r = int(get_env("LORA_R", "16"))
    lora_alpha = int(get_env("LORA_ALPHA", "32"))

    records = load_s3_jsonl(s3, gold_data_path)
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} training samples from {gold_data_path}")

    print(f"Loading base model: {base_model_id}")
    model, tokenizer = load_model_and_tokenizer(base_model_id)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

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

    print("Starting SFT QLoRA fine-tuning...")
    trainer.train()

    output_dir = "/tmp/student-merged"
    merge_and_save(model, base_model_id, output_dir)
    upload_model_to_s3(s3, output_dir, model_output_s3_path)


# =========================================================================
# DPO Training
# =========================================================================

def _format_pref_records(records: list[dict]) -> list[dict]:
    """Convert plain-text preference pairs to chat format for DPOTrainer.

    DPOTrainer in trl >= 0.12 expects chosen/rejected as lists of message dicts,
    not raw strings.
    """
    formatted = []
    for r in records:
        prompt = r.get("prompt", "")
        chosen = r.get("chosen", "")
        rejected = r.get("rejected", "")
        if not (prompt and chosen and rejected):
            continue
        formatted.append({
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": [{"role": "assistant", "content": chosen}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return formatted


def run_dpo(s3):
    from trl import DPOConfig, DPOTrainer

    pref_data_path = get_env("PREF_DATA_PATH")
    model_output_s3_path = get_env("MODEL_OUTPUT_S3_PATH")
    base_model_id = get_env("BASE_MODEL_ID")
    num_epochs = int(get_env("NUM_EPOCHS", "1"))
    batch_size = int(get_env("BATCH_SIZE", "1"))
    learning_rate = float(get_env("LEARNING_RATE", "5e-5"))
    lora_r = int(get_env("LORA_R", "16"))
    lora_alpha = int(get_env("LORA_ALPHA", "32"))
    dpo_beta = float(get_env("DPO_BETA", "0.1"))

    records = load_s3_jsonl(s3, pref_data_path)
    if not records:
        print("No preference data found -- skipping DPO")
        return

    records = _format_pref_records(records)
    dataset = Dataset.from_list(records)
    print(f"Loaded {len(dataset)} preference pairs from {pref_data_path}")

    # If BASE_MODEL_ID is an S3 path, download it locally first
    local_model_id = base_model_id
    if base_model_id.startswith("s3://"):
        local_model_id = "/tmp/sft-model"
        if os.path.exists(local_model_id):
            shutil.rmtree(local_model_id)
        os.makedirs(local_model_id, exist_ok=True)
        print(f"Downloading SFT model from {base_model_id}...")
        download_s3_dir(s3, base_model_id, local_model_id)

    print(f"Loading SFT model for DPO: {local_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(local_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 1024
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = AutoModelForCausalLM.from_pretrained(
        local_model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dpo_args = DPOConfig(
        output_dir="/tmp/dpo-checkpoints",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        beta=dpo_beta,
        fp16=True,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.1,
        remove_unused_columns=False,
        max_length=512,
        max_prompt_length=256,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # Let DPOTrainer handle PEFT via peft_config (don't apply get_peft_model manually).
    # With ref_model=None + peft_config, trl creates an implicit reference model
    # by toggling LoRA adapters on/off, saving GPU memory.
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        peft_config=lora_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print(f"Starting DPO training (beta={dpo_beta})...")
    trainer.train()

    output_dir = "/tmp/student-dpo-merged"
    print("Saving DPO adapter and merging...")
    trainer.save_model("/tmp/dpo-adapter")

    del trainer
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    peft_model = PeftModel.from_pretrained(base_model, "/tmp/dpo-adapter")
    merged_model = peft_model.merge_and_unload()
    print("DPO LoRA merged into full-precision model")
    merged_model.save_pretrained(output_dir)

    clean_tokenizer = AutoTokenizer.from_pretrained(local_model_id)
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

    upload_model_to_s3(s3, output_dir, model_output_s3_path)


# =========================================================================
# Entrypoint
# =========================================================================

def main():
    training_mode = get_env("TRAINING_MODE", "sft")
    print(f"Training mode: {training_mode}")

    s3 = build_s3_client()

    if training_mode == "dpo":
        run_dpo(s3)
    else:
        run_sft(s3)


if __name__ == "__main__":
    main()
