import re
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

def extract_last_boxed(text: str) -> Optional[str]:
    if not text:
        return None

    starts = [m.start() for m in re.finditer(r"\\(?:boxed|fbox)\s*\{", text)]
    if not starts:
        return None

    start_idx = starts[-1]
    

    i = start_idx
    while i < len(text) and text[i] != "{":
        i += 1
    
    if i >= len(text):
        return None

    balance = 0
    content_start = i + 1
    for j in range(content_start, len(text)):
        char = text[j]
        if char == "{":
            balance += 1
        elif char == "}":
            if balance == 0:
                return text[content_start:j]
            balance -= 1
            
    return None  

def normalize_math_str(s: str) -> str:
    if not s: 
        return ""
    
    s = re.sub(r"\\(text|mbox|mathrm|textbf)\s*\{([^\}]+)\}", r"\2", s)
    
    for cmd in [r"\$", r"\%", r"\degree", r"^\circ", r"\,", r"\!"]:
        s = s.replace(cmd, "")
        
    s = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}", 
        r"(\1)/(\2)", 
        s
    )
    
    for char in ["$", "€", "£", ",", " "]:
        s = s.replace(char, "")
        
    return s.strip()

import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier


def is_equiv_with_detail(pred: str, truth: str) -> Tuple[bool, str]:

    p_norm = normalize_math_str(pred)
    t_norm = normalize_math_str(truth)

    if p_norm == t_norm:
        return True, f"exact string match: '{p_norm}' == '{t_norm}'"

    try:
        p_float = float(p_norm)
        t_float = float(t_norm)
        if abs(p_float - t_float) < 1e-6:
            return True, f"float match: abs({p_float} - {t_float}) < 1e-6"
    except ValueError:
        pass

    if SYMPY_AVAILABLE:
        try:
            transforms = standard_transformations + (implicit_multiplication_application,)
            
            p_sym = parse_expr(p_norm, transformations=transforms)
            t_sym = parse_expr(t_norm, transformations=transforms)
            
            diff = sympy.simplify(p_sym - t_sym)
            if diff == 0:
                return True, "sympy match: simplify(pred - truth) == 0"
        except Exception:
            pass 

    return False, f"mismatch: '{p_norm}' != '{t_norm}'"


@dataclass
class UniversalMathVerifier(VerifierProtocol):
    name: str = "universal_math"

    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        truth = (ground_truths or [""])[0]
        truth = str(truth)
        
        boxed_content = extract_last_boxed(model_answer or "")
        
        if boxed_content:
            candidate = boxed_content
            source = "boxed"
        else:
            candidates = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", model_answer or "")
            if candidates:
                candidate = candidates[-1]
                source = "extracted" 
            else:
                return VerificationResult(
                    correct=False, 
                    explanation=f"no numeric content found; truth='{truth}'"
                )

        is_correct, detail = is_equiv_with_detail(candidate, truth)
        
        return VerificationResult(
            correct=is_correct,
            explanation=f"{source}='{candidate}'; {detail}; truth='{truth}'"
        )

@register_verifier("universal_math")
def _universal_math_factory(params: Dict[str, Any]) -> UniversalMathVerifier:
    return UniversalMathVerifier()