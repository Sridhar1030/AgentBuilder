"""
KFP Component -- Quality Gate

Reads the structured evaluation results produced by the evaluate component
and returns a simple "pass" / "fail" string for pipeline branching.

The evaluate component does the heavy lifting (running judges, checking
thresholds).  This component exists solely to expose the gate decision
as a primitive string output that dsl.Condition can branch on.
"""

from kfp import dsl


@dsl.component(base_image="python:3.11-slim")
def quality_gate(eval_results: dict) -> str:
    """Extract the quality gate decision from evaluation results.

    Returns:
        "pass" if all harness thresholds were met, "fail" otherwise.
    """
    gate_result = eval_results.get("gate_result", "fail")
    composite = eval_results.get("composite_score", 0.0)
    regressions = eval_results.get("regressions", [])
    judge_agg = eval_results.get("judge_aggregates", {})
    num_questions = eval_results.get("num_questions", 0)

    print("=" * 60)
    print("QUALITY GATE")
    print("=" * 60)
    print(f"  Decision:        {gate_result.upper()}")
    print(f"  Composite score: {composite:.4f}")
    print(f"  Questions eval'd:{num_questions}")

    if judge_agg:
        print("\n  Judge scores:")
        for name, agg in judge_agg.items():
            if agg.get("pass_rate") is not None:
                print(f"    {name}: pass_rate={agg['pass_rate']:.1%}")
            elif agg.get("mean") is not None:
                print(f"    {name}: mean={agg['mean']:.2f}/5")

    if regressions:
        print(f"\n  {len(regressions)} threshold(s) FAILED:")
        for r in regressions:
            print(f"    {r['judge']}: {r['metric']}="
                  f"{r['actual']:.4f} < {r['threshold']:.4f}")
    else:
        print("\n  All thresholds passed.")

    print("=" * 60)
    return gate_result
