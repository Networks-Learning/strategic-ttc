import re
import random
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Callable
import numpy as np
from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def _ci_percentile(xs, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    lo, hi = np.quantile(xs, [alpha/2, 1 - alpha/2])
    lo = np.mean(xs) - lo
    hi = hi - np.mean(xs)
    return (lo, hi)

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



def compute_curves_for_model_fast(
    data: Dict[str, Any],
    parse_pred_fn: Callable,
    sample_size: int = 50,
    rng_seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(rng_seed)

    expl_all = data["explanations"]
    corr_all = data["correct"]
    rew_all = data["reward"]
    token_all = data["num_tokens"]

    num_questions = len(corr_all)
    if num_questions == 0:
        raise ValueError("No questions found in data")

    n_samples_list = [len(c) for c in corr_all]
    N = min(n_samples_list)
    if N == 0:
        raise ValueError("Some questions have zero samples")

    def truncate_and_stack(source_list, dtype):
        return np.array([item[:N] for item in source_list], dtype=dtype)

    rew_matrix = np.array(
        [[(float(x) if x is not None else -np.inf) for x in q_rewards[:N]] for q_rewards in rew_all],
        dtype=np.float32,
    )
    corr_matrix = truncate_and_stack(corr_all, dtype=bool)
    token_matrix = truncate_and_stack(token_all, dtype=np.float32)

    flat_expl = [item for sublist in expl_all for item in sublist[:N]]
    flat_parsed = [parse_pred_fn(e) for e in flat_expl]

    NONE_SENTINEL = "__NONE__"
    flat_parsed_str = [str(p) if p is not None else NONE_SENTINEL for p in flat_parsed]

    uniq, pred_ids = np.unique(flat_parsed_str, return_inverse=True)
    preds_matrix = pred_ids.reshape(num_questions, N)

    none_locs = np.where(uniq == NONE_SENTINEL)[0]
    none_id = int(none_locs[0]) if len(none_locs) else -1  

    rand_noise = np.random.rand(sample_size, num_questions, N)
    permutations = np.argsort(rand_noise, axis=-1)

    thetas = np.array([2**i for i in range(int(np.log2(N))) if 2**i != 2], dtype=int)

    maj_means, maj_stds = [], []
    rew_means, rew_stds = [], []
    token_means, token_stds = [], []

    def _sem(xs):
        return np.std(xs) / np.sqrt(len(xs)) if len(xs) > 1 else np.nan

    for theta in tqdm(thetas):
        curr_indices = permutations[:, :, :theta]  

        batch_preds = np.take_along_axis(preds_matrix[None, ...], curr_indices, axis=-1)    
        batch_corrs = np.take_along_axis(corr_matrix[None, ...], curr_indices, axis=-1)      
        batch_rews  = np.take_along_axis(rew_matrix[None, ...], curr_indices, axis=-1)       
        batch_tokens = np.take_along_axis(token_matrix[None, ...], curr_indices, axis=-1)    

        best_idx_local = np.argmax(batch_rews, axis=-1)  
        best_corr = np.take_along_axis(batch_corrs, best_idx_local[..., None], axis=-1).squeeze(-1)  

        trial_acc_rew = np.mean(best_corr, axis=1)  
        rew_means.append(np.nanmean(trial_acc_rew))
        rew_stds.append(_ci_percentile(trial_acc_rew))

        if none_id >= 0:
            valid_mask = (batch_preds != none_id)  
        else:
            valid_mask = np.ones_like(batch_preds, dtype=bool)

        has_valid = np.any(valid_mask, axis=-1)  

        eq = batch_preds[..., :, None] == batch_preds[..., None, :] 
        counts = np.sum(eq & valid_mask[..., :, None] & valid_mask[..., None, :], axis=-1)  

        if none_id >= 0:
            counts = np.where(batch_preds == none_id, -1, counts)


        maj_pos = np.argmax(counts, axis=-1)  
        maj_label = np.take_along_axis(batch_preds, maj_pos[..., None], axis=-1).squeeze(-1)  

        is_majority = (batch_preds == maj_label[..., None])  
        maj_and_correct = is_majority & batch_corrs & valid_mask
        maj_correct_flags = np.any(maj_and_correct, axis=-1)  

        maj_correct_float = np.where(has_valid, maj_correct_flags.astype(np.float32), np.nan)  
        trial_acc_maj = np.nanmean(maj_correct_float, axis=1)  

        maj_means.append(np.nanmean(trial_acc_maj))
        maj_stds.append(_ci_percentile(trial_acc_maj))

        total_tokens_per_q = np.sum(batch_tokens, axis=-1)   
        trial_tokens = np.mean(total_tokens_per_q, axis=1)   
        token_means.append(np.mean(trial_tokens))
        token_stds.append(_ci_percentile(trial_tokens))

    return (
        thetas,
        np.array(maj_means), np.array(maj_stds),
        np.array(rew_means), np.array(rew_stds),
        np.array(token_means), np.array(token_stds),
    )

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


    maj_mean = np.asarray(maj_mean)
    rew_mean = np.asarray(rew_mean)

    maj_yerr = np.array(maj_std).T
    rew_yerr = np.array(rew_std).T

    ax = axes[0]
    ax.errorbar(
        thetas,
        maj_mean,
        yerr=maj_yerr,
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
        yerr=rew_yerr,
        fmt="-o",
        capsize=3,
    )
    ax.set_xlabel("Number of samples")
    ax.set_title(f"Reward-max\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"{model_name}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_model_curves_with_tokens(
    model_name: str,
    thetas: np.ndarray,
    maj_mean: np.ndarray,
    maj_std: np.ndarray,      
    rew_mean: np.ndarray,
    rew_std: np.ndarray,     
    token_mean: np.ndarray,
    token_std: np.ndarray,
    num_questions: int,
):
    thetas = np.asarray(thetas)
    maj_mean = np.asarray(maj_mean, dtype=float)
    rew_mean = np.asarray(rew_mean, dtype=float)
    token_mean = np.asarray(token_mean, dtype=float)

    maj_yerr = np.asarray(maj_std, dtype=float).T
    rew_yerr = np.asarray(rew_std, dtype=float).T
    tok_yerr = np.asarray(token_std, dtype=float).T

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)

    ax = axes[0]
    ax.errorbar(thetas, maj_mean, yerr=maj_yerr, fmt="-o", capsize=3)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Majority vote\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.errorbar(thetas, rew_mean, yerr=rew_yerr, fmt="-o", capsize=3)
    ax.set_xlabel("Number of samples")
    ax.set_title(f"Reward-max\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.errorbar(thetas, token_mean, yerr=tok_yerr, fmt="-o", capsize=3)
    ax.set_xlabel("Number of samples")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Tokens\n({num_questions} questions)")
    ax.grid(True, alpha=0.3)

    fig.suptitle(model_name, fontsize=14)
    fig.tight_layout()
    plt.show()


def _split_asym_err(err, n):

    err = np.asarray(err, dtype=float)

    if err.ndim == 1:
        if err.shape[0] != n:
            raise ValueError(f"err length {err.shape[0]} != n {n}")
        return err, err

    if err.ndim == 2:
        if err.shape == (n, 2):
            return err[:, 0], err[:, 1]
        if err.shape == (2, n):
            return err[0, :], err[1, :]

    raise ValueError(f"Unsupported err shape {err.shape}; expected ({n},), ({n},2), or (2,{n})")

def plot_all_curves(
    thetas_dict,
    maj_mean_dict,
    maj_std_dict,
    best_mean_dict,
    best_std_dict,
    filename="gsm8k_results.pdf"
):
    fig, (ax_maj, ax_best) = plt.subplots(2, 1, figsize=(6.75, 5), sharex=True)
    colors = plt.cm.tab10.colors

    for i, model_name in enumerate(sorted(thetas_dict)):
        th = np.asarray(thetas_dict[model_name])
        n = th.size
        color = colors[i % len(colors)]

        display_name = model_name
        if "llama" in display_name.lower():
            display_name = display_name.replace("llama-", "llama ")

        clean_label = display_name.split("--")[0].replace("_", " ")

        maj_mean = np.asarray(maj_mean_dict[model_name], dtype=float)
        best_mean = np.asarray(best_mean_dict[model_name], dtype=float)

        maj_lo, maj_hi = _split_asym_err(maj_std_dict[model_name], n)
        best_lo, best_hi = _split_asym_err(best_std_dict[model_name], n)

        ax_maj.plot(
            th, maj_mean,
            label=clean_label,
            marker="s", markersize=4, linewidth=1.2,
            color=color
        )
        ax_maj.fill_between(
            th,
            maj_mean - maj_lo,
            maj_mean + maj_hi,
            alpha=0.15, color=color, edgecolor="none"
        )

        ax_best.plot(
            th, best_mean,
            marker="o", markersize=4, linewidth=1.2,
            color=color
        )
        ax_best.fill_between(
            th,
            best_mean - best_lo,
            best_mean + best_hi,
            alpha=0.15, color=color, edgecolor="none"
        )

    ax_maj.set_ylabel("Majority Accuracy")
    ax_best.set_ylabel(r"Best-of-$N$ Accuracy")
    ax_best.set_xlabel(r"Number of Samples ($N$)")

    ax_maj.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def plot_all_curves_delta(
    thetas_dict,
    maj_mean_dict,
    maj_std_dict,
    best_mean_dict,
    best_std_dict,
    filename="gsm8k_results_delta.pdf"
):
    fig, (ax_maj, ax_best) = plt.subplots(2, 1, figsize=(6.75, 5), sharex=True)
    colors = plt.cm.tab10.colors

    tick_th = None

    for i, model_name in enumerate(sorted(thetas_dict)):
        th = np.asarray(thetas_dict[model_name])
        if tick_th is None:
            tick_th = th

        color = colors[i % len(colors)]

        display_name = model_name
        if "llama" in display_name.lower():
            display_name = display_name.replace("llama-", "llama ")

        clean_label = display_name.split("--")[0].replace("_", " ")

        maj_mean = np.asarray(maj_mean_dict[model_name], dtype=float)
        best_mean = np.asarray(best_mean_dict[model_name], dtype=float)

        maj_base = maj_mean[0]
        best_base = best_mean[0]

        maj_delta = maj_mean - maj_base
        best_delta = best_mean - best_base

        maj_err = np.asarray(maj_std_dict[model_name], dtype=float)
        best_err = np.asarray(best_std_dict[model_name], dtype=float)

        if maj_err.shape[0] == 2 and maj_err.shape[1] == th.size:
            maj_err = maj_err.T
        if best_err.shape[0] == 2 and best_err.shape[1] == th.size:
            best_err = best_err.T

        if maj_err.shape != (th.size, 2):
            raise ValueError(f"{model_name}: maj_err shape {maj_err.shape}, expected ({th.size}, 2)")
        if best_err.shape != (th.size, 2):
            raise ValueError(f"{model_name}: best_err shape {best_err.shape}, expected ({th.size}, 2)")

        maj_lo, maj_hi = maj_err[:, 0], maj_err[:, 1]
        best_lo, best_hi = best_err[:, 0], best_err[:, 1]

        ax_maj.plot(
            th, maj_delta,
            label=clean_label,
            marker="s", markersize=4, linewidth=1.2,
            color=color,
        )
        ax_maj.fill_between(
            th,
            maj_delta - maj_lo,
            maj_delta + maj_hi,
            alpha=0.15, color=color, edgecolor="none",
        )

        ax_best.plot(
            th, best_delta,
            marker="o", markersize=4, linewidth=1.2,
            color=color,
        )
        ax_best.fill_between(
            th,
            best_delta - best_lo,
            best_delta + best_hi,
            alpha=0.15, color=color, edgecolor="none",
        )

    ax_maj.set_ylabel(r"$\Delta$ Majority Accuracy")
    ax_best.set_ylabel(r"$\Delta$ Best-of-$N$ Accuracy")
    ax_best.set_xlabel(r"Number of Samples ($N$)")

    ax_maj.set_xscale("log", base=2)
    ax_best.set_xscale("log", base=2)

    if tick_th is not None:
        ax_best.set_xticks(tick_th)
    ax_best.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    ax_maj.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.35),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.show()


def _ci_percentile_mean_bootstrap(xs, alpha=0.05, B=2000, rng=None):
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if xs.size == 0:
        return (np.nan, np.nan)

    mu = xs.mean()

    if xs.size == 1:
        return (0.0, 0.0)

    if rng is None:
        rng = np.random.default_rng(0)

    idx = rng.integers(0, xs.size, size=(B, xs.size))
    boot_means = xs[idx].mean(axis=1)

    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return (mu - lo, hi - mu)


def reasoning_accuracy_curve_second_custom(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)

    num_bins = len(qs) - 1
    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]

    all_think = []
    for correct_list, token_list in zip(model_results["correct"], model_results["num_tokens"]):
        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[0] > 5:
                all_think.append(t[0])

    if len(all_think) == 0:
        nan_means = [np.nan] * num_bins
        nan_errs = [(np.nan, np.nan)] * num_bins
        return nan_means, nan_errs, nan_means, nan_errs

    all_think = np.asarray(all_think, dtype=float)
    boundaries = np.quantile(all_think, qs)

    for correct_list, token_list in zip(model_results["correct"], model_results["num_tokens"]):
        think = []
        corr = []
        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[0] > 5:
                think.append(t[0])
                corr.append(c)

        if len(think) == 0:
            continue

        think = np.asarray(think, dtype=float)
        corr = np.asarray(corr, dtype=float)

        for b in range(num_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            if b == num_bins - 1:
                mask = (think >= lo) & (think <= hi)
            else:
                mask = (think >= lo) & (think < hi)

            if mask.any():
                all_bin_accs[b].append(corr[mask].mean())
                all_bin_think[b].append(think[mask].mean())

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []

    for b in range(num_bins):
        xs_acc = np.asarray(all_bin_accs[b], dtype=float)
        xs_th  = np.asarray(all_bin_think[b], dtype=float)

        if xs_acc.size == 0:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))
        else:
            acc_means.append(xs_acc.mean())
            acc_errs.append(_ci_percentile_mean_bootstrap(xs_acc, alpha=alpha, B=B, rng=rng))

        if xs_th.size == 0:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))
        else:
            think_means.append(xs_th.mean())
            think_errs.append(_ci_percentile_mean_bootstrap(xs_th, alpha=alpha, B=B, rng=rng))

    return acc_means, acc_errs, think_means, think_errs


# def plot_reasoning_curve_2(
#     means,
#     stds,
#     title="Accuracy vs Reasoning Level",
#     num_questions=None,
#     filename=None,
#     color=None,
# ):
#     means = np.asarray(means, dtype=float)
#     n = means.size
#     bins = np.arange(1, n + 1)

#     if color is None:
#         color = plt.cm.tab10.colors[0]

#     stds = np.asarray(stds, dtype=float)

#     if stds.ndim == 1:
#         lo = hi = stds
#     elif stds.ndim == 2 and stds.shape == (n, 2):
#         lo, hi = stds[:, 0], stds[:, 1]
#     elif stds.ndim == 2 and stds.shape == (2, n):
#         lo, hi = stds[0, :], stds[1, :]
#     else:
#         raise ValueError(f"Unsupported stds shape {stds.shape}; expected ({n},), ({n},2) or (2,{n})")

#     fig, ax = plt.subplots(figsize=(6.75, 3.5))

#     ax.plot(
#         bins,
#         means,
#         marker="o",
#         markersize=4,
#         linewidth=1.2,
#         color=color,
#     )

#     ax.fill_between(
#         bins,
#         means - lo,
#         means + hi,
#         alpha=0.15,
#         color=color,
#         edgecolor="none",
#     )

#     ax.set_xticks(bins)
#     ax.set_xlabel("Reasoning Level")
#     ax.set_ylabel("Accuracy")

#     if num_questions is not None:
#         ax.set_title(f"{title}\nQuestions: {num_questions}")
#     else:
#         ax.set_title(title)

#     plt.tight_layout()

#     if filename is not None:
#         plt.savefig(filename, bbox_inches="tight")

#     plt.show()


def plot_reasoning_curves_accuracy_and_tokens(
    acc_means,
    acc_errs,
    tok_means,
    tok_errs,
    title="Reasoning curves",
    num_questions=None,
    filename=None,
    color=None,
):
    acc_means = np.asarray(acc_means, dtype=float)
    tok_means = np.asarray(tok_means, dtype=float)

    n = acc_means.size
    if tok_means.size != n:
        raise ValueError(f"tok_means length {tok_means.size} != acc_means length {n}")

    bins = np.arange(1, n + 1)

    if color is None:
        color = plt.cm.tab10.colors[0]

    acc_lo, acc_hi = _split_asym_err(acc_errs, n)
    tok_lo, tok_hi = _split_asym_err(tok_errs, n)

    fig, (ax_acc, ax_tok) = plt.subplots(2, 1, figsize=(6.75, 5.5), sharex=True)

    ax_acc.plot(bins, acc_means, marker="o", markersize=4, linewidth=1.2, color=color)
    ax_acc.fill_between(bins, acc_means - acc_lo, acc_means + acc_hi, alpha=0.15, color=color, edgecolor="none")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, alpha=0.3)

    ax_tok.plot(bins, tok_means, marker="o", markersize=4, linewidth=1.2, color=color)
    ax_tok.fill_between(bins, tok_means - tok_lo, tok_means + tok_hi, alpha=0.15, color=color, edgecolor="none")
    ax_tok.set_xlabel("Reasoning Level")
    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xticks(bins)
    ax_tok.grid(True, alpha=0.3)

    if num_questions is not None:
        fig.suptitle(f"{title}\nQuestions: {num_questions}", y=0.98)
    else:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()
