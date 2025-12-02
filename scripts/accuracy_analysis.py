import re
import random
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

def majority_vote_correct(
    preds: List[Optional[str]],
    correct: List[bool],
) -> Optional[bool]:
    """
    Given a list of predicted values (strings) and their correctness flags,
    compute whether the *majority prediction* is correct.

    Strategy:
      - ignore None preds
      - find most common pred (ties broken arbitrarily but deterministically)
      - consider majority pred correct if any sample with that pred is correct.
    """
    valid = [(p, c) for p, c in zip(preds, correct) if p is not None]
    if not valid:
        return None 

    values = [p for p, _ in valid]
    counts = Counter(values)
    majority_value, _ = counts.most_common(1)[0]

    for p, c in valid:
        if p == majority_value and c:
            return True
    return False

def compute_curves_for_model(
    data: Dict[str, Any],
    parse_pred_fn,               
    sample_size: int = 50,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """
    data: dict with keys "answers", "explanations", "correct", "reward"
          each a list[num_questions] of list[num_samples].
    sample_size: how many Monte Carlo trials per θ.

    Returns:
      thetas, maj_mean, maj_std, rew_mean, rew_std
    """
    random.seed(rng_seed)
    np.random.seed(rng_seed)

    answers_all = data["answers"]
    expl_all = data["explanations"]
    corr_all = data["correct"]
    rew_all = data["reward"]

    num_questions = len(corr_all)
    if num_questions == 0:
        raise ValueError("No questions found in data")

    n_samples_list = [len(c) for c in corr_all]
    N = min(n_samples_list)  # use the minimum over questions
    if N == 0:
        raise ValueError("Some questions have zero samples")

    thetas = np.array([2**i for i in range(int(np.log2(N)) + 1)], dtype=int)

    maj_means = []
    maj_stds = []
    rew_means = []
    rew_stds = []

    for theta in tqdm(thetas):
        maj_acc_trials = []
        rew_acc_trials = []

        for _ in range(sample_size):
            # per trial, per question
            maj_correct_flags = []
            rew_correct_flags = []

            for q_idx in range(num_questions):
                n_q = len(corr_all[q_idx])
                if theta > n_q:
                    # not enough samples for this question -> skip it
                    continue

                idxs = random.sample(range(n_q), theta)

                # slice per-question lists
                expl_q = [expl_all[q_idx][i] for i in idxs]
                corr_q = [corr_all[q_idx][i] for i in idxs]
                rew_q  = [rew_all[q_idx][i] for i in idxs]

                preds_q = [parse_pred_fn(e) for e in expl_q]
                maj_corr = majority_vote_correct(preds_q, corr_q)

                if maj_corr is not None:
                    maj_correct_flags.append(1.0 if maj_corr else 0.0)

                if all(r is None for r in rew_q):
                    continue

                rew_numeric = [(-float("inf") if r is None else float(r)) for r in rew_q]
                best_idx_local = int(np.argmax(rew_numeric))
                rew_correct_flags.append(1.0 if corr_q[best_idx_local] else 0.0)

            if maj_correct_flags:
                maj_acc_trials.append(np.mean(maj_correct_flags))
            if rew_correct_flags:
                rew_acc_trials.append(np.mean(rew_correct_flags))

        maj_means.append(np.mean(maj_acc_trials) if maj_acc_trials else np.nan)
        maj_stds.append(np.std(maj_acc_trials) if maj_acc_trials else np.nan)
        rew_means.append(np.mean(rew_acc_trials) if rew_acc_trials else np.nan)
        rew_stds.append(np.std(rew_acc_trials) if rew_acc_trials else np.nan)

    return thetas, np.array(maj_means), np.array(maj_stds), np.array(rew_means), np.array(rew_stds)

import matplotlib.pyplot as plt
import numpy as np

def plot_model_curves(
    model_name: str,
    thetas: np.ndarray,
    maj_mean: np.ndarray,
    maj_std: np.ndarray,
    rew_mean: np.ndarray,
    rew_std: np.ndarray,
    num_questions: int,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    ax = axes[0]
    ax.errorbar(
        thetas,
        maj_mean,
        yerr=maj_std,
        fmt="-o",
        capsize=3,
    )
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Majority vote\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(
        thetas,
        rew_mean,
        yerr=rew_std,
        fmt="-o",
        capsize=3,
    )
    ax.set_xlabel("Number of samples")
    ax.set_title(f"Reward-max\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name}", fontsize=14)
    plt.tight_layout()
    plt.show()
