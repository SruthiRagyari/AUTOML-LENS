import json
data = json.load(open("benchmarks/results/benchmark_results.json"))
for r in data:
    if r.get("dataset_name") == "winequality_red" and r.get("seed") == 456:
        print("=" * 60)
        print("condition:", r.get("condition"))
        print("overall_winner:", r.get("overall_winner"), "| best_ensemble_model:", r.get("best_ensemble_model"))
        print("overall_holdout_score:", r.get("overall_holdout_score"))
        print("best_ensemble_holdout_score:", r.get("best_ensemble_holdout_score"))
        print("candidate_models:", [m.get("model_name") for m in r.get("candidate_models", [])])
        for e in r.get("ensemble_candidates", []):
            print("  ensemble:", e.get("model_name"), "| strategy:", e.get("strategy"),
                  "| holdout:", e.get("holdout_score"), "| weights:", e.get("weights"))
