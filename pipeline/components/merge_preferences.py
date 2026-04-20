"""
KFP Component -- Merge Preferences

Merges DPO preference pairs from three sources:
  1. Static pre-generated bank (bulk signal, sampled per run)
  2. Live extract_preferences output (fresh student-vs-teacher signal)
  3. Human feedback pairs (real-world corrections from VS Code extension)
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=["boto3"],
)
def merge_preferences(
    pipeline_pref_path: str,
    human_feedback_path: str,
    gold_data_path: str,
    s3_endpoint: str,
    s3_access_key: str,
    s3_secret_key: str,
    bucket: str = "mlflow-artifacts",
    static_bank_key: str = "preferences/static-bank/preference-bank.jsonl",
    static_bank_sample_size: int = 300,
    sft_mix_ratio: float = 0.15,
) -> str:
    """Merge static bank (sampled), live pipeline, human feedback, and SFT regularization pairs."""
    import json
    import random
    import re
    import uuid

    import boto3

    random.seed(42)

    print("=" * 60)
    print("MERGE PREFERENCES STEP")
    print("=" * 60)
    print(f"  Static bank:     s3://{bucket}/{static_bank_key}")
    print(f"  Sample size:     {static_bank_sample_size}")
    print(f"  Pipeline prefs:  {pipeline_pref_path}")
    print(f"  Human feedback:  {human_feedback_path or '(none)'}")
    print(f"  Gold data:       {gold_data_path or '(none)'}")
    print(f"  SFT mix ratio:   {sft_mix_ratio}")

    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=s3_access_key,
        aws_secret_access_key=s3_secret_key,
    )

    all_pairs = []

    def read_jsonl_from_s3(s3_path: str, label: str) -> int:
        if not s3_path:
            print(f"  {label}: empty path, skipping")
            return 0
        parts = s3_path.replace("s3://", "").split("/", 1)
        try:
            obj = s3.get_object(Bucket=parts[0], Key=parts[1])
        except s3.exceptions.NoSuchKey:
            print(f"  {label}: key not found, skipping")
            return 0
        lines = [l for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]
        count = 0
        for line in lines:
            pair = json.loads(line)
            if pair.get("prompt") and pair.get("chosen") and pair.get("rejected"):
                all_pairs.append(pair)
                count += 1
        print(f"  {label}: {count} pairs")
        return count

    # Source 1: Static pre-generated bank (sample a subset)
    static_count = 0
    try:
        obj = s3.get_object(Bucket=bucket, Key=static_bank_key)
        lines = [l for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]
        bank_pairs = []
        for line in lines:
            pair = json.loads(line)
            if pair.get("prompt") and pair.get("chosen") and pair.get("rejected"):
                bank_pairs.append(pair)

        print(f"  Static bank total: {len(bank_pairs)} pairs")
        if len(bank_pairs) > static_bank_sample_size:
            sampled = random.sample(bank_pairs, static_bank_sample_size)
        else:
            sampled = bank_pairs
        all_pairs.extend(sampled)
        static_count = len(sampled)
        print(f"  Static bank sampled: {static_count} pairs")
    except Exception as e:
        print(f"  Static bank not found or error: {e}")
        print(f"  (Run generate_preference_bank job first to create it)")

    # Source 2: Live pipeline preferences (student vs teacher)
    pipeline_count = read_jsonl_from_s3(pipeline_pref_path, "Pipeline live prefs")

    # Source 3: Human feedback
    human_count = read_jsonl_from_s3(human_feedback_path, "Human feedback")

    # Source 4: SFT regularization data (prevents forgetting SFT task format)
    # Takes gold SFT examples and reformats as DPO pairs:
    #   chosen = gold reviewer comment, rejected = generic unhelpful response
    sft_mix_count = 0
    if gold_data_path:
        try:
            parts = gold_data_path.replace("s3://", "").split("/", 1)
            obj = s3.get_object(Bucket=parts[0], Key=parts[1])
            lines = [l for l in obj["Body"].read().decode().strip().split("\n") if l.strip()]

            sft_records = []
            for line in lines:
                rec = json.loads(line)
                text = rec.get("text", "")
                if not text:
                    continue
                sft_records.append(text)

            # Sample 15% of SFT data
            dpo_pair_count = len(all_pairs) if all_pairs else 300
            mix_count = max(10, int(dpo_pair_count * sft_mix_ratio))
            if len(sft_records) > mix_count:
                sampled_sft = random.sample(sft_records, mix_count)
            else:
                sampled_sft = sft_records

            generic_rejects = [
                "The code looks fine to me.",
                "No issues found.",
                "LGTM.",
                "This change is acceptable.",
                "I don't see any problems with this code.",
            ]

            for text in sampled_sft:
                # Extract user message (the diff prompt) and assistant message (the review)
                user_match = re.search(
                    r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL
                )
                asst_match = re.search(
                    r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", text, re.DOTALL
                )
                if user_match and asst_match:
                    prompt = user_match.group(1).strip()
                    chosen = asst_match.group(1).strip()
                    if len(chosen) > 20:
                        all_pairs.append({
                            "prompt": prompt,
                            "chosen": chosen,
                            "rejected": random.choice(generic_rejects),
                            "source": "sft_regularization",
                        })
                        sft_mix_count += 1

            print(f"  SFT regularization: {sft_mix_count} pairs (from {len(sft_records)} gold records)")
        except Exception as e:
            print(f"  SFT mixing failed: {e}")

    print(f"\n  Total merged: {len(all_pairs)} pairs")
    print(f"    Static bank: {static_count}")
    print(f"    Pipeline:    {pipeline_count}")
    print(f"    Human:       {human_count}")
    print(f"    SFT mix:     {sft_mix_count}")

    if not all_pairs:
        print("  No pairs to merge. Returning pipeline path as-is.")
        return pipeline_pref_path

    random.shuffle(all_pairs)

    run_id = uuid.uuid4().hex[:12]
    out_key = f"preferences/merged-{run_id}.jsonl"
    body = "\n".join(json.dumps(p) for p in all_pairs)
    s3.put_object(Bucket=bucket, Key=out_key, Body=body.encode())

    out_path = f"s3://{bucket}/{out_key}"
    print(f"  Wrote merged preferences to {out_path}")
    return out_path
