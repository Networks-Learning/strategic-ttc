import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from hdetect.interfaces.verifier import VerifierProtocol, VerificationResult
from hdetect.config import register_verifier

CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def _last_boxed_content(text: str) -> Optional[str]:
    """
    Return the content of the LAST \boxed{...} occurrence.
    Uses a brace counter to handle nested braces.
    """

    starts = [m.end() for m in re.finditer(r'\\boxed\s*\{', text)]
    if not starts:
        return None

    start = starts[-1] 
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
        i += 1
    if depth != 0:
        return None 

    inner = text[start : i - 1]
    return inner.strip()


def _extract_choice_letter(s: str, max_choice: str = "E") -> Optional[str]:
    """
    Try to extract a single choice letter (A..max_choice) from text s.
    - Ignores letters that are part of words (e.g., 'Paris').
    - Accepts forms like 'B', '(B)', 'B.', 'Option B', 'choice: B', 'B) ..'.
    - Supports digits '1'..'5' -> 'A'.. (if present).
    """

    s_norm = s.replace("$", " ")

    hi = CHOICES.index(max_choice.upper()) + 1
    allowed = CHOICES[:hi]

    # Direct letter: 'B', '(B)', 'B.', 'B)', etc.
    m = re.search(rf'(?<![A-Za-z])([{allowed}])(?![A-Za-z])', s_norm, flags=re.IGNORECASE)
    
    if m:
        return m.group(1).upper()

    # Fallback: 'option B', 'choice B', etc.
    m = re.search(rf'(?:option|choice)\s*[:\-\s]*([{allowed}])(?![A-Za-z])',
                  s_norm, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Numeric fallback: 1->A, 2->B, ...
    m = re.search(r'(?<!\d)([1-9])(?!\d)', s_norm)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(allowed):
            return CHOICES[idx]

    return None


@dataclass
class BoxedChoiceVerifier(VerifierProtocol):
    name: str = "boxed_choice"
    max_choice: str = "E"  

    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        truth = (ground_truths or [""])[0].strip().upper()[:1]

        inner = _last_boxed_content(model_answer or "")
        if inner is None:
            return VerificationResult(
                correct=False,
                explanation=f"no \\boxed{{...}} found; truth={truth}"
            )

        pred = _extract_choice_letter(inner, max_choice=self.max_choice)
        if pred is None:
            return VerificationResult(
                correct=False,
                explanation=f"boxed='{inner}' -> could not parse choice; truth={truth}"
            )

        correct = (pred == truth)
        return VerificationResult(
            correct=correct,
            explanation=f"pred={pred}; truth={truth}; boxed='{inner}'"
        )


@register_verifier("boxed_choice")
def _boxed_choice_factory(params):
    return BoxedChoiceVerifier(max_choice=params.get("max_choice", "E"))
