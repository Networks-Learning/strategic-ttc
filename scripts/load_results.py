from collections import defaultdict
from pathlib import Path
import json

def load_results(runs_dir="../runs/GSM8K", keep_ids=None):
    runs_dir = Path(runs_dir)

    results = defaultdict(lambda: {
        "answers": [],
        "correct": [],
        "reward": [],
        "explanations": [],
        "num_tokens": [],
    })

    seen_qids = defaultdict(set)
    
    for jsonl_file in runs_dir.glob("*.jsonl"):
        total_processed = 0

        filename = jsonl_file.stem
        model_name = filename.split("--temp")[0]
        # print(f"Processing {model_name}")

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

                qid = obj.get("qid")
                if qid is None or (keep_ids is not None and qid not in keep_ids):
                    continue

                if qid in seen_qids[model_name]:
                    continue
                seen_qids[model_name].add(qid)

                explanations = obj.get("explanations", [])
                correct_list = obj.get("correct", [])
                reward_list = obj.get("rewards", [])
                num_tokens = obj.get("num_tokens", [])

                results[model_name]["explanations"].append(explanations)
                results[model_name]["correct"].append(correct_list)
                results[model_name]["reward"].append(reward_list)
                results[model_name]["num_tokens"].append(num_tokens)

                total_processed += 1

        # print(f"Total processed: {total_processed}")

    results = dict(sorted(results.items(), key=lambda item: item[0]))

    return results
