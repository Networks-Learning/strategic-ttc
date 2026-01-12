import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier

_answer_pat = re.compile(
    r"Answer\s*:\s*[^\w]*([ABCD])\b",
    flags=re.IGNORECASE,
)

@dataclass
class GPQAPer(VerifierProtocol):
    name: str = "gpqa_per"
    
    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        truth = (ground_truths or [""])[0]
        truth = str(truth).strip().upper()

        text = model_answer or ""
        
        matches = _answer_pat.findall(text)

        if not matches:

            if not matches:
                 return VerificationResult(
                    correct=False,
                    explanation=f"no 'Answer: [A-D]' pattern found; truth='{truth}'",
                )

        pred = matches[-1].strip().upper()
        
        ok = (pred == truth)

        return VerificationResult(
            correct=ok,
            explanation=f"parsed='{pred}'; truth='{truth}'",
        )

@register_verifier("gpqa_per")
def _gpqa_per_factory(params: Dict[str, Any]) -> GPQAPer:
    return GPQAPer()