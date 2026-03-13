"""
KFP Component 2 — QLoRA Fine-Tune

Reads gold training data from MinIO, fine-tunes the 1B student with QLoRA,
merges the adapters, and uploads the merged model back to MinIO.
"""

from kfp import dsl


@dsl.component(
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    packages_to_install=[
        "peft==0.15.2",
        "trl==0.17.0",
        "bitsandbytes>=0.46.1",
        "datasets==3.6.0",
        "accelerate==1.6.0",
        "boto3",
        "sentencepiece",
        "protobuf",
    ],
)
def finetune(
    gold_data_path: str,
    model_output_s3_path: str,
    base_model_id: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
) -> str:
    """Run QLoRA SFT on the student model using extracted gold data."""
    import json
    import os
    import torch
    import boto3
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

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

    import os
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

    from peft import PeftModel
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload()
    print("LoRA merged into full-precision model")
    merged_model.save_pretrained(output_dir)

    # Save tokenizer then strip fields that break older vLLM/transformers.
    clean_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    clean_tokenizer.save_pretrained(output_dir)

    import json as _json
    tc_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(tc_path, "r") as f:
        tc = _json.load(f)
    removed = []
    for key in ["tokenizer_class", "auto_map"]:
        if key in tc:
            removed.append(f"{key}={tc.pop(key)}")
    if removed:
        with open(tc_path, "w") as f:
            _json.dump(tc, f, indent=2)
        print(f"Patched tokenizer_config.json: removed {removed}")

    dest_parts = model_output_s3_path.replace("s3://", "").split("/", 1)
    dest_bucket, dest_prefix = dest_parts[0], dest_parts[1]

    for root, _, files in os.walk(output_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            s3_key = os.path.join(dest_prefix, os.path.relpath(local_path, output_dir))
            print(f"Uploading {fname} -> s3://{dest_bucket}/{s3_key}")
            s3.upload_file(local_path, dest_bucket, s3_key)

    print(f"Model uploaded to {model_output_s3_path}")
    return model_output_s3_path
