from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple

import yaml

from strategic_ttc.interfaces.benchmark import BenchmarkProtocol
from strategic_ttc.interfaces.verifier import VerifierProtocol
from strategic_ttc.models.hf_chat import HFChatModel

BenchmarkFactory = Callable[[Dict[str, Any]], BenchmarkProtocol]
VerifierFactory = Callable[[Dict[str, Any]], VerifierProtocol]

BENCHMARKS: Dict[str, BenchmarkFactory] = {}
VERIFIERS: Dict[str, VerifierFactory] = {}


def register_benchmark(name: str):
    def deco(fn: BenchmarkFactory):
        BENCHMARKS[name] = fn
        return fn
    return deco


def register_verifier(name: str):
    def deco(fn: VerifierFactory):
        VERIFIERS[name] = fn
        return fn
    return deco


def _ensure_registries_loaded() -> None:
    import importlib

    importlib.import_module("strategic_ttc.benchmarks")
    importlib.import_module("strategic_ttc.verifiers")


@dataclass(frozen=True)
class RunConfig:
    # model block
    model_id: str
    device: str
    dtype: Optional[str]

    # experiment block
    max_tokens: int
    temperature: float
    n_samples: int
    output_path: str  # resolved to absolute in load_yaml

    # benchmark block
    benchmark_type: str
    benchmark_params: Dict[str, Any]

    # verifier block
    verifier_type: str
    verifier_params: Dict[str, Any]


def build_config(cfg: Dict[str, Any]) -> RunConfig:
    """
    Convert raw YAML dict into a typed RunConfig.

    Expected YAML structure:

    model:
      model_id: ...
      device: ...
      dtype: ...

    experiment:
      max_tokens: ...
      temperature: ...
      n_samples: ...
      output_path: ...   # path to JSONL, may be relative to this YAML

    benchmark:
      type: ...
      params:
        path: ...        # may be relative to this YAML

    verifier:
      type: ...
      params: ...
    """
    return RunConfig(
        model_id=cfg["model"]["model_id"],
        device=cfg["model"].get("device", "auto"),
        dtype=cfg["model"].get("dtype"),
        max_tokens=cfg["experiment"]["max_tokens"],
        temperature=cfg["experiment"]["temperature"],
        n_samples=cfg["experiment"]["n_samples"],
        output_path=cfg["experiment"]["output_path"],
        benchmark_type=cfg["benchmark"]["type"],
        benchmark_params=cfg["benchmark"].get("params", {}),
        verifier_type=cfg["verifier"]["type"],
        verifier_params=cfg["verifier"].get("params", {}),
    )


def build_components(
    run_cfg: RunConfig,
) -> Tuple[HFChatModel, BenchmarkProtocol, VerifierProtocol]:
    _ensure_registries_loaded()

    model = HFChatModel(
        model_id=run_cfg.model_id,
        device=run_cfg.device,
        dtype=run_cfg.dtype,
        max_tokens=run_cfg.max_tokens,
        temperature=run_cfg.temperature,
    )

    try:
        benchmark = BENCHMARKS[run_cfg.benchmark_type](run_cfg.benchmark_params)
    except KeyError:
        raise KeyError(
            f"Unknown benchmark '{run_cfg.benchmark_type}'. "
            f"Available: {list(BENCHMARKS.keys())}"
        )

    try:
        verifier = VERIFIERS[run_cfg.verifier_type](run_cfg.verifier_params)
    except KeyError:
        raise KeyError(
            f"Unknown verifier '{run_cfg.verifier_type}'. "
            f"Available: {list(VERIFIERS.keys())}"
        )

    return model, benchmark, verifier


def _resolve_paths_in_cfg(
    cfg: Dict[str, Any],
    cfg_path: Path,
) -> Dict[str, Any]:
    """
    Resolve any relative paths in the config to be absolute, relative to the YAML file.

    Currently:
      - benchmark.params.path
      - experiment.output_path
    """
    # benchmark path
    try:
        p = cfg["benchmark"]["params"].get("path")
    except Exception:
        p = None

    if p:
        p = Path(p)
        if not p.is_absolute():
            cfg["benchmark"]["params"]["path"] = str((cfg_path.parent / p).resolve())

    # experiment output_path
    try:
        out = cfg["experiment"].get("output_path")
    except Exception:
        out = None

    if out:
        out_p = Path(out)
        if not out_p.is_absolute():
            cfg["experiment"]["output_path"] = str((cfg_path.parent / out_p).resolve())

    return cfg


def load_yaml(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _resolve_paths_in_cfg(cfg, cfg_path)
    return cfg
