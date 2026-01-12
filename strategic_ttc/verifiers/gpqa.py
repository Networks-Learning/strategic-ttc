import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier

_answer_pat = re.compile(
    r"Answer\s*:\s*\$?\s*([ABCD])\s*\$?",
    flags=re.IGNORECASE,
)

@dataclass
class GPQAMCVerifier(VerifierProtocol):
    name: str = "gpqa_mc"
    
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
            return VerificationResult(
                correct=False,
                explanation=f"no Answer: <LETTER> found; truth='{truth}'",
            )

        pred = matches[-1].strip().upper()
        ok = pred == truth

        return VerificationResult(
            correct=ok,
            explanation=f"parsed='{pred}'; truth='{truth}'",
        )

def parse_pred_from_explanation_gpqa(expl: Optional[str]) -> Optional[str]:
    if not expl:
        return None

    first = expl.split(";", 1)[0]

    m = re.search(r"parsed\s*=\s*'([^']+)'", first)
    if not m:
        return None

    pred = m.group(1).strip().upper()
    return pred if pred else None

@register_verifier("gpqa_mc")
def _gpqa_mc_factory(params: Dict[str, Any]) -> GPQAMCVerifier:
    return GPQAMCVerifier()
