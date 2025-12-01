from dataclasses import dataclass
from typing import Protocol, List, Dict, Any, Optional

from strategic_ttc.interfaces.benchmark import Question
from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult


@dataclass(frozen=True)
class SampleOutcome:
    answer: str
    verification: VerificationResult
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class TTCResult:
    question_id: str
    method_name: str

    chosen_answer: str
    chosen_verification: VerificationResult

    samples: List[SampleOutcome]
    meta: Optional[Dict[str, Any]] = None  


class TTCMethodProtocol(Protocol):
    """
    Test-time compute method that aggregates *already generated* answers
    for a single question.
    """
    name: str

    def aggregate(
        self,
        *,
        question: Question,
        answers: List[str],
        verifier: VerifierProtocol,
        meta_list: Optional[List[Dict[str, Any]]] = None,
    ) -> TTCResult:
        ...
