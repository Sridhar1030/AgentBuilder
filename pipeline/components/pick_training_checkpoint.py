"""Pick SFT or DPO checkpoint for downstream GRPO based on harness correctness."""

from kfp import dsl


@dsl.component(base_image="python:3.11-slim")
def pick_training_checkpoint(
    sft_model_s3_path: str,
    dpo_model_s3_path: str,
    eval_sft_results: dict,
    eval_dpo_results: dict,
) -> str:
    """Return DPO path only if correctness did not regress vs SFT; otherwise SFT."""
    print("=" * 60)
    print("PICK TRAINING CHECKPOINT")
    print("=" * 60)

    def correctness_pass_rate(results: dict) -> float:
        agg = (results or {}).get("judge_aggregates", {})
        corr = agg.get("correctness", {})
        rate = corr.get("pass_rate")
        if rate is None:
            rate = corr.get("mean")
        return float(rate) if rate is not None else 0.0

    sft_rate = correctness_pass_rate(eval_sft_results)
    dpo_rate = correctness_pass_rate(eval_dpo_results)

    print(f"  SFT correctness pass rate: {sft_rate:.1%}")
    print(f"  DPO correctness pass rate: {dpo_rate:.1%}")

    if dpo_rate >= sft_rate:
        print(f"  -> Using DPO checkpoint (no regression)")
        return dpo_model_s3_path

    print(f"  -> DPO regressed vs SFT; using SFT checkpoint for GRPO")
    return sft_model_s3_path
