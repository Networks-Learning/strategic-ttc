import argparse
from pathlib import Path
from typing import Optional

from strategic_ttc.config import load_yaml, build_config, build_components
from strategic_ttc.core.generation import generate_and_save_jsonl
from strategic_ttc.models.reward_armor import ArmoRewardModel

import random
import numpy as np
import torch

def set_global_seed(seed: int = 1234) -> None:
    print(f"[strategic-ttc] Setting global seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model answers on a benchmark and save them to JSONL."
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> None:
    if args is None:
        args = _parse_args()

    set_global_seed(1234)

    cfg_path = Path(args.config).resolve()
    raw_cfg = load_yaml(str(cfg_path))
    run_cfg = build_config(raw_cfg)

    # resolve output path from YAML
    out_path = Path(run_cfg.output_path)

    model, benchmark, verifier = build_components(run_cfg)
    n_samples = run_cfg.n_samples

    print(f"[strategic-ttc] Config: {cfg_path}")
    print(f"[strategic-ttc] Output: {out_path}")
    print(f"[strategic-ttc] Model: {getattr(model, 'name', 'unknown')}")
    print(f"[strategic-ttc] Benchmark: {benchmark.name} (n_samples={n_samples})")
    print(f"[strategic-ttc] Verifier: {verifier.name}")
    print(f"[strategic-ttc] Reasoning: {run_cfg.reasoning}")
    if run_cfg.system_prompt is not None:
        print(f"[strategic-ttc] System prompt: {run_cfg.system_prompt[:80]}...")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    reward_model = ArmoRewardModel(
        model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device="cuda:1",       
        dtype="bfloat16",
        local_files_only=False,
    )

    generate_and_save_jsonl(
        model=model,
        benchmark=benchmark,
        verifier=verifier,
        n_samples=n_samples,
        output_path=str(out_path),
        reward_model=reward_model,
        system_prompt=run_cfg.system_prompt,   
        reasoning=run_cfg.reasoning,
    )

    print(f"[strategic-ttc] Done. JSONL saved to: {out_path}")


if __name__ == "__main__":
    main()
