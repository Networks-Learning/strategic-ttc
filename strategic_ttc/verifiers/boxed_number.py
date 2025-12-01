# strategic_ttc/verifiers/boxed_number.py

import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier


def _last_boxed_content(text: str) -> Optional[str]:
    text = text or ""

    open_pat = r"(?:\\boxed|\\fbox)\s*\{"
    starts = [m.end() for m in re.finditer(open_pat, text)]
    if not starts:
        return None

    start = starts[-1]
    depth = 1
    i = start

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth == 0:
        return text[start : i - 1].strip()

    return None



_number_token_re = re.compile(
    r"""
    (?P<num>
        [\(\[]?-?        # optional leading bracket and sign
        (?:\$)?          # optional currency
        (?:\d{1,3}(?:,\d{3})+|\d+)?   # integer part with optional thousands
        (?:\.\d+)?       # decimal part
        (?:\s+)?         # optional space (for mixed numbers)
        (?:\d+/\d+)?     # optional fraction (simple)
        [\)\]]?          # optional trailing bracket
        (?:\s*%)?        # optional percent
    )
    """,
    re.VERBOSE,
)

def _strip_wrappers(s: str) -> str:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1].strip()
    return s


def _parse_simple_number(s: str) -> Optional[float]:
    if not s:
        return None
    s = _strip_wrappers(s)
    s = s.replace("$", "").replace("₹", "").replace("€", "").replace("£", "")
    s = s.replace(",", " ")
    s = re.sub(r"\s+", " ", s).strip()

    negative = False
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()

    # mixed number "a b/c"
    m = re.match(r"^(\d+)\s+(\d+)/(\d+)$", s)
    if m:
        a, b, c = map(int, m.groups())
        val = a + (b / c if c != 0 else 0.0)
    else:
        # pure fraction "b/c"
        m = re.match(r"^(\d+)/(\d+)$", s)
        if m:
            b, c = map(int, m.groups())
            val = (b / c) if c != 0 else None
        else:
            try:
                s_plain = s.replace(" ", "")
                if not s_plain:
                    return None
                val = float(s_plain)
            except ValueError:
                return None

    if val is None:
        return None
    if negative:
        val = -val
    if is_percent:
        val = val / 100.0
    return val


def _extract_numeric(text: str) -> Optional[str]:
    candidates = [m.group("num") for m in _number_token_re.finditer(text)]
    candidates = [c.strip() for c in candidates if c and re.search(r"\d", c)]
    return candidates[-1] if candidates else None


def _compare_numbers(pred: str, truth: str, tol: float = 1e-9) -> Tuple[bool, str]:
    p_val = _parse_simple_number(pred)
    t_val = _parse_simple_number(truth)
    if p_val is not None and t_val is not None:
        ok = abs(p_val - t_val) <= tol * max(1.0, abs(t_val))
        return ok, f"numeric compare: pred={p_val} truth={t_val} tol={tol}"

    p_norm = pred.strip()
    t_norm = truth.strip()
    return p_norm == t_norm, f"string compare: pred='{p_norm}' truth='{t_norm}'"


@dataclass
class BoxedNumberVerifier(VerifierProtocol):
    name: str = "boxed_number"
    tol: float = 1e-5

    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        truth = (ground_truths or [""])[0]
        truth = str(truth)

        # Prefer \boxed{...} or variants if present
        boxed = _last_boxed_content(model_answer or "")
        if boxed is None or not re.search(r"\d", boxed):
            # Fallback: last numeric token in the whole answer
            extracted = _extract_numeric(model_answer or "")
            if extracted is None:
                return VerificationResult(
                    correct=False,
                    explanation=f"no numeric content found; truth='{truth}'",
                )
            pred_raw = extracted
            source = "extracted"
        else:
            pred_raw = boxed
            source = "boxed"

        correct, detail = _compare_numbers(pred_raw, truth, tol=self.tol)
        return VerificationResult(
            correct=correct,
            explanation=f"{source}='{pred_raw}'; {detail}; truth='{truth}'",
        )


@register_verifier("boxed_number")
def _boxed_number_factory(params: Dict[str, Any]) -> BoxedNumberVerifier:
    return BoxedNumberVerifier(tol=params.get("tol", 1e-5))
