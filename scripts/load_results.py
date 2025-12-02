import json
from pathlib import Path
from collections import defaultdict


def load_results(runs_dir="../runs/GSM8K"):
    runs_dir = Path(runs_dir)
    results = defaultdict(lambda: {"answers": [], "correct": [], "reward": [], "explanations": []})

    for jsonl_file in runs_dir.glob("*.jsonl"):
        filename = jsonl_file.stem   # e.g. "llama3b-temp0.7--samples128"
        model_name = filename.split("--")[0]  # take everything before first "--"

        with jsonl_file.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)

                explanations = obj.get("explanations", [])
                correct_list = obj.get("correct", [])
                reward_list = obj.get("rewards", [])

                results[model_name]["explanations"].append(explanations)
                results[model_name]["correct"].append(correct_list)
                results[model_name]["reward"].append(reward_list)

    return results
