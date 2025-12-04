import json
from pathlib import Path
from typing import Any, Dict, Optional, List
from tqdm import tqdm

from strategic_ttc.interfaces.benchmark import BenchmarkProtocol, Question
from strategic_ttc.interfaces.verifier import VerifierProtocol


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
    wait: int = 0,
    feedback_model: Optional[Any] = None,
) -> None:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "name", None)
    reward_model_name = getattr(reward_model, "name", None) if reward_model is not None else None
    feedback_model_name = getattr(feedback_model, "name", None) if feedback_model is not None else None

    with out_path.open("w", encoding="utf-8") as f:
        for question in tqdm(benchmark.iter_questions(), desc="Generating samples"):
            answers: List[List[str]] = []
            correct: List[List[bool]] = []
            explanations: List[List[Optional[str]]] = []
            num_tokens: List[List[int]] = []
            rewards: List[List[Optional[float]]] = []

            for _ in range(n_samples):
                gen = model.generate(
                    question.prompt,
                    wait=wait,
                    feedback_model=feedback_model,
                )

                sample_answers: List[str] = []
                sample_correct: List[bool] = []
                sample_explanations: List[Optional[str]] = []
                sample_tokens: List[int] = []
                sample_rewards: List[Optional[float]] = []

                for text, tok_count in zip(gen.texts, gen.tokens):
                    verification = verifier.verify(
                        model_answer=text,
                        ground_truths=question.ground_truths,
                        prompt=question.prompt,
                        meta=question.meta,
                    )

                    reward = _compute_reward(
                        reward_model,
                        question=question,
                        answer=text,
                    )

                    sample_answers.append(text)
                    sample_correct.append(verification.correct)
                    sample_explanations.append(verification.explanation)
                    sample_tokens.append(tok_count)
                    sample_rewards.append(reward)

                answers.append(sample_answers)
                correct.append(sample_correct)
                explanations.append(sample_explanations)
                num_tokens.append(sample_tokens)
                rewards.append(sample_rewards)

            record: Dict[str, Any] = {
                "qid": question.qid,
                "prompt": question.prompt,
                "answers": answers,
                "correct": correct,
                "explanations": explanations,
                "num_tokens": num_tokens,
                "rewards": rewards,
                "wait": wait,
                "ground_truths": question.ground_truths,
                "question_meta": question.meta,
                "model_name": model_name,
                "reward_model_name": reward_model_name,
                "feedback_model_name": feedback_model_name,
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
