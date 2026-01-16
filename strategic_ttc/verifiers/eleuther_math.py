import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult
from strategic_ttc.config import register_verifier

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    SYMPY_AVAILABLE = False # i dont want sympy, if you want change that
except ImportError:
    SYMPY_AVAILABLE = False

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
    
    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    
    return s

def last_boxed_only_string(string: str) -> Optional[str]:
    if not string:
        return None
        
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_int = int(a)
        b_int = int(b)
        if string == "{}/{}".format(a_int, b_int):
            new_string = "\\frac{" + str(a_int) + "}{" + str(b_int) + "}"
            return new_string
    except (ValueError, AssertionError):
        pass
    return string

def remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string

def fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string: str) -> str:
    """
    EleutherAI's aggressive normalization logic.
    """
    if not string:
        return ""

    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "") 

    # " 0." equivalent to " ." and "{0." equivalent to "{." 
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # fix fractions
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # X/Y changed to \frac{X}{Y}
    string = fix_a_slash_b(string)

    return string



def is_equiv_with_detail(pred: str, truth: str) -> Tuple[bool, str]:
    """
    1. Try EleutherAI's 'strip_string' normalization (Exact Match).
    2. Fallback to SymPy for mathematical equivalence.
    """
    p_norm = strip_string(pred)
    t_norm = strip_string(truth)

    if p_norm == t_norm:
        return True, f"eleuther_strip_match: '{p_norm}' == '{t_norm}'"

    try:
        if abs(float(p_norm) - float(t_norm)) < 1e-6:
            return True, "float_match"
    except ValueError:
        pass


    if SYMPY_AVAILABLE:
        try:
            transforms = standard_transformations + (implicit_multiplication_application,)
            
            if not p_norm or not t_norm:
                return False, "empty_string"

            p_sym = parse_expr(p_norm, transformations=transforms)
            t_sym = parse_expr(t_norm, transformations=transforms)
            
            diff = sympy.simplify(p_sym - t_sym)
            if diff == 0:
                return True, "sympy_match"
        except Exception:
            pass 

    return False, f"mismatch: '{p_norm}' != '{t_norm}'"



@dataclass
class UniversalMathVerifier(VerifierProtocol):
    name: str = "eleuther_math"

    def verify(
        self,
        model_answer: str,
        ground_truths: Optional[List[str]] = None,
        prompt: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> VerificationResult:
        truth = (ground_truths or [""])[0]
        truth = str(truth)
        
        boxed_content = last_boxed_only_string(model_answer or "")
        
        if boxed_content:

            candidate = remove_boxed(boxed_content)
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
        
        truth_clean = remove_boxed(truth)

        is_correct, detail = is_equiv_with_detail(candidate, truth_clean)
        
        return VerificationResult(
            correct=is_correct,
            explanation=f"{source}='{candidate}'; {detail}; truth='{truth_clean}'"
        )

@register_verifier("eleuther_math")
def _eleuther_math_factory(params: Dict[str, Any]) -> UniversalMathVerifier:
    return UniversalMathVerifier()