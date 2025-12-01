import json
from pathlib import Path
from typing import Any, Dict, Optional
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


def generate_and_save_jsonl(
    *,
    model: Any,
    benchmark: BenchmarkProtocol,
    verifier: VerifierProtocol,
    n_samples: int,
    output_path: str,
    reward_model: Optional[Any] = None,
) -> None:
    """
    Run n_samples generations per question in `benchmark` using `model`,
    verify each with `verifier`, optionally score with `reward_model`, and
    save everything to a JSONL file at `output_path`.

    JSONL schema (one line per (question, sample)):

        {
          "qid": <str>,
          "sample_id": <int>,
          "answer": <str>,
          "verification": {
            "correct": <bool>,
            "explanation": <str or null>
          },
          "reward": <float or null>,
          "ground_truths": [<str>, ...] or null,
          "question_meta": { ... } or null,
          "model_name": <str or null>,
          "reward_model_name": <str or null>
        }

    We assume `model` has a .generate(prompt: str) -> str method.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "name", None)
    reward_model_name = getattr(reward_model, "name", None) if reward_model is not None else None

    with out_path.open("w", encoding="utf-8") as f:
        for question in tqdm(benchmark.iter_questions(), desc="Generating samples"):
            answers: list[str] = []
            correct: list[bool] = []
            explanations: list[Optional[str]] = []
            num_tokens: list[int] = []
            rewards: list[Optional[float]] = []

            for _ in range(n_samples):
                gen = model.generate(question.prompt)  

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
                num_tokens.append(gen.tokens)
                rewards.append(reward)

            record: Dict[str, Any] = {
                "qid": question.qid,
                "prompt": question.prompt,
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
