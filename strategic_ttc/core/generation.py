import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, List
from tqdm.auto import tqdm

from strategic_ttc.interfaces.benchmark import BenchmarkProtocol, Question
from strategic_ttc.interfaces.verifier import VerifierProtocol, VerificationResult


def _compute_reward(
    reward_model: Any,
    *,
    question: Question,
    answer: str,
) -> Optional[float]:
    if reward_model is None:
        return None

    if hasattr(reward_model, "score"):
        return float(
            reward_model.score(
                question=question,
                answer=answer,
            )
        )

    raise TypeError(
        "reward_model must implement a .score(question=..., answer=...) -> float method"
    )


def _count_think_tokens(
    *,
    model: Any,
    text: str,
) -> int:
    m = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return 0

    think_segment = m.group(1)

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        return len(think_segment.split())

    ids = tokenizer(think_segment, return_tensors="pt")["input_ids"][0]
    return int(ids.shape[0])

def _count_existing_lines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n

def generate_and_save_jsonl(
    *,
    model: Any,
    benchmark: BenchmarkProtocol,
    verifier: VerifierProtocol,
    n_samples: int,
    output_path: str,
    reward_model: Optional[Any] = None,
    system_prompt: Optional[str] = None,
    reasoning: bool = False,
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "name", None)
    reward_model_name = getattr(reward_model, "name", None) if reward_model is not None else None

    already_done = _count_existing_lines(out_path)
    total = len(benchmark)
    print(f"[strategic-ttc] Resuming: {already_done}/{total} questions already in {out_path.name}")

    with out_path.open("a", encoding="utf-8") as f:
        for idx, question in enumerate(tqdm(benchmark.iter_questions(), total=total, desc="Generating samples")):
            if idx < already_done:
                continue

            answers: List[str] = []
            correct: List[bool] = []
            explanations: List[Optional[str]] = []
            num_tokens: List[Any] = []
            rewards: List[Optional[float]] = []

            for _ in range(n_samples):
                question_prompt = question.prompt
                if system_prompt is not None:
                    question_prompt = question_prompt + "\n" + system_prompt  

                gen = model.generate(question_prompt)

                verification = verifier.verify(
                    model_answer=gen.text,
                    ground_truths=question.ground_truths,
                    prompt=question.prompt,
                    meta=question.meta,
                )

                reward = _compute_reward(
                    reward_model,
                    question=question,
                    answer=gen.text,
                )

                answers.append(gen.text)
                correct.append(verification.correct)
                explanations.append(verification.explanation)
                rewards.append(reward)

                if reasoning:
                    think_tokens = _count_think_tokens(model=model, text=gen.text)
                    total_tokens = int(gen.tokens)
                    num_tokens.append([think_tokens, total_tokens])
                else:
                    num_tokens.append(int(gen.tokens))

            record: Dict[str, Any] = {
                "qid": question.qid,
                "prompt": question_prompt,
                "answers": answers,
                "correct": correct,
                "explanations": explanations,
                "num_tokens": num_tokens,
                "rewards": rewards,
                "ground_truths": question.ground_truths,
                "question_meta": question.meta,
                "model_name": model_name,
                "reward_model_name": reward_model_name,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()