from dataclasses import dataclass
from typing import Protocol, Optional, List, Dict, Any


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    explanation: Optional[str] = None


class VerifierProtocol(Protocol):
    name: str

    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult: ...
