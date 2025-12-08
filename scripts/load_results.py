import json
from pathlib import Path
from collections import defaultdict


def load_results(runs_dir="../runs/GSM8K"):
    runs_dir = Path(runs_dir)
    results = defaultdict(lambda: {"answers": [], "correct": [], "reward": [], "explanations": [], "num_tokens": []})

    for jsonl_file in runs_dir.glob("*.jsonl"):
        print(f"Processing file: {jsonl_file.name}")
        filename = jsonl_file.stem   # e.g. "llama3b-temp0.7--samples128"
        model_name = filename.split("--samples")[0]  # take everything before first "--"

        with jsonl_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):

                line = line.strip()
                if not line:
                    continue

                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Bad JSON at line {i}: {e}")
                    print("Offending line repr:", repr(line[:200]))
                    continue

                explanations = obj.get("explanations", [])
                correct_list = obj.get("correct", [])
                reward_list = obj.get("rewards", [])
                num_tokens = obj.get("num_tokens", [])

                results[model_name]["explanations"].append(explanations)
                results[model_name]["correct"].append(correct_list)
                results[model_name]["reward"].append(reward_list)
                results[model_name]["num_tokens"].append(num_tokens)

    return results
