import argparse
from pathlib import Path
from typing import Optional

from strategic_ttc.config import load_yaml, build_config, build_components
from strategic_ttc.core.generation import generate_and_save_jsonl
from strategic_ttc.models.reward_armor import ArmoRewardModel


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

    # ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # TODO: make this configurable later
    reward_model = ArmoRewardModel(
        model_id="RLHFlow/ArmoRM-Llama3-8B-v0.1",
        device="cuda:1",          # or "auto"
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
    )

    print(f"[strategic-ttc] Done. JSONL saved to: {out_path}")


if __name__ == "__main__":
    main()
