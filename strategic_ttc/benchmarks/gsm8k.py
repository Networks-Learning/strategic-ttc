import json
from pathlib import Path
from typing import Iterable, Optional, Dict, Any, List

from strategic_ttc.interfaces.benchmark import BenchmarkProtocol, Question
from strategic_ttc.config import register_benchmark


class GSM8KJsonlBenchmark(BenchmarkProtocol):
    def __init__(
        self,
        path: str,
        name: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        self._path = Path(path)
        self.name = name or self._path.stem
        self._limit = limit

    def iter_questions(self) -> Iterable[Question]:
        """
        Expects a JSONL file where each line is a dict like:
        {
            "question": <str>,
            "answer": <str>,   # full rationale with "#### <final>"
            "final": <str>     # canonical short answer, e.g. "72"
        }
        """
        with self._path.open("r", encoding="utf-8") as f:
            yielded = 0
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj: Dict[str, Any] = json.loads(line)

                q_text = obj.get("question")
                prompt = (q_text or "").strip()
                if not prompt:
                    continue

                ans = obj.get("final")

                qid = str(obj.get("id") or obj.get("qid") or f"gsm8k-{i+1}")

                yield Question(
                    qid=qid,
                    prompt=prompt,
                    ground_truths=[str(ans)],
                    meta=obj,  
                )

                yielded += 1
                if self._limit is not None and yielded >= self._limit:
                    break

    def __len__(self) -> int:
        """
        Count usable questions, respecting the same filters + limit.
        """
        cnt = 0
        with self._path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                obj = json.loads(line)

                prompt = (obj.get("question") or "").strip()
                if not prompt:
                    continue

                ans = obj.get("final")

                cnt += 1
                if self._limit is not None and cnt >= self._limit:
                    break
        return cnt


@register_benchmark("gsm8k_jsonl")
def _gsm8k_factory(params: Dict[str, Any]) -> BenchmarkProtocol:
    return GSM8KJsonlBenchmark(
        path=params["path"],
        name=params.get("name"),
        limit=params.get("limit"),
    )
