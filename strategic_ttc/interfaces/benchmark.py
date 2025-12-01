from typing import Protocol, Iterable, Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    prompt: str
    ground_truths: Optional[List[str]] = None  
    meta: Optional[Dict[str, Any]] = None  


class BenchmarkProtocol(Protocol):
    name: str

    def iter_questions(self) -> Iterable[Question]: ...

    def __len__(self) -> int: ...
