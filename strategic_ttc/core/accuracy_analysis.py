import re
import random
from collections import Counter
from typing import List, Optional, Tuple, Dict, Any, Callable
import os
import logging
import numpy as np
from scipy import stats
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as ticker
import os
from typing import Dict, Optional, Tuple
from matplotlib.ticker import FuncFormatter, LogLocator

def _ci_percentile(xs, alpha=0.05):
    xs = np.asarray(xs, dtype=float)
    xs = xs[~np.isnan(xs)]
    if xs.size == 0:
        return np.nan, np.nan
    
    mean_val = np.mean(xs)
    lo_val, hi_val = np.quantile(xs, [alpha/2, 1 - alpha/2])
    
    lo_diff = max(0.0, mean_val - lo_val)
    hi_diff = max(0.0, hi_val - mean_val)
    
    return (lo_diff, hi_diff)

def majority_vote_correct(
    preds: List[Optional[str]],
    correct: List[bool],
) -> Optional[bool]:
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

        maj_correct_float = np.where(has_valid, maj_correct_flags.astype(np.float32), 0.0)
        
        trial_acc_maj = np.mean(maj_correct_float, axis=1)  

        maj_means.append(np.mean(trial_acc_maj))
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
    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Majority vote\n({num_questions} questions)")
    ax.grid(True)

    ax = axes[1]
    ax.errorbar(
        thetas,
        rew_mean,
        yerr=rew_yerr,
        fmt="-o",
        capsize=3,
    )
    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_title(f"Best-of-$N$\n({num_questions} questions)")
    ax.grid(True)

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
    log_scale: bool = False,
    colors: Dict[str, str] = None,
    filename: Optional[str] = None,
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
    ax.errorbar(thetas, maj_mean, yerr=maj_yerr, fmt="-o", capsize=3, color=colors.get(model_name, None) if colors else None)
    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Majority vote")
    ax.grid(True)

    ax = axes[1]
    ax.errorbar(thetas, rew_mean, yerr=rew_yerr, fmt="-o", capsize=3, color=colors.get(model_name, None) if colors else None)
    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_title(f"Best-of-$N$")
    ax.grid(True)

    ax = axes[2]
    ax.errorbar(thetas, token_mean, yerr=tok_yerr, fmt="-o", capsize=3, color=colors.get(model_name, None) if colors else None)
    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_ylabel("Tokens")
    ax.set_title(f"Tokens")
    ax.grid(True)

    if log_scale:
        for ax in axes:
            ax.set_xscale("log", base=2)

    fig.suptitle(model_name, fontsize=14)
    fig.tight_layout()

    if filename:
        out_path = os.path.abspath(filename)
    else:
        raise ValueError("No filename provided for saving plot; please provide a valid filename.")
    
    parent_dir = os.path.dirname(out_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    try:
        fig.savefig(out_path, bbox_inches="tight")
    except Exception as e:
        logging.warning("Failed to save figure %s with current rcParams (error: %s). Trying fallback by disabling text.usetex.", out_path, e)
        prev_usetex = plt.rcParams.get("text.usetex", False)
        try:
            plt.rcParams["text.usetex"] = False
            fig.savefig(out_path, bbox_inches="tight")
        finally:
            plt.rcParams["text.usetex"] = prev_usetex
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
    vertical=True,
    colors: Dict[str, str] = None,
    filename="gsm8k_results.pdf",
):  

    if vertical:
        fig, (ax_maj, ax_best) = plt.subplots(2, 1, figsize=(6.75, 5), sharex=True)
    else:
        fig, (ax_maj, ax_best) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for i, model_name in enumerate(sorted(thetas_dict)):
        th = np.asarray(thetas_dict[model_name])
        n = th.size
        color = colors.get(model_name, None) if colors else None

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

    ax_maj.set_ylabel(r"$\Delta$ Majority Accuracy")
    ax_best.set_ylabel(r"$\Delta$ Best-of-$N$ Accuracy")
    if not vertical:
        ax_maj.set_xlabel(r"Number of Samples ($N$)")
    ax_best.set_xlabel(r"Number of Samples ($N$)")

    ax_maj.set_xscale("log", base=2)
    ax_best.set_xscale("log", base=2)

    # if tick_th is not None:
    #     ax_best.set_xticks(tick_th)
    # ax_best.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        plt.savefig(filename, bbox_inches="tight")
    except Exception as e:
        logging.warning("Failed to save plot_all_curves to %s (error: %s). Trying fallback by disabling text.usetex.", filename, e)
        prev_usetex = plt.rcParams.get("text.usetex", False)
        try:
            plt.rcParams["text.usetex"] = False
            plt.savefig(filename, bbox_inches="tight")
        finally:
            plt.rcParams["text.usetex"] = prev_usetex
    plt.show()


def plot_all_curves_delta(
    thetas_dict,
    maj_mean_dict,
    maj_std_dict,
    best_mean_dict,
    best_std_dict,
    vertical=True,
    colors: Dict[str, str] = None,
    filename="gsm8k_results_delta.pdf"
):  

    if vertical:
        fig, (ax_maj, ax_best) = plt.subplots(2, 1, figsize=(6.75, 5), sharex=True, constrained_layout=True)
    else:
        fig, (ax_maj, ax_best) = plt.subplots(1, 2, figsize=(12, 4), sharey=True, constrained_layout=True)

    tick_th = None

    for i, model_name in enumerate(sorted(thetas_dict)):
        th = np.asarray(thetas_dict[model_name])
        if tick_th is None:
            tick_th = th

        color = colors.get(model_name, None) if colors else None
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
    if not vertical:
        ax_maj.set_xlabel(r"Number of Samples ($N$)")
    ax_best.set_xlabel(r"Number of Samples ($N$)")

    ax_maj.set_xscale("log", base=2)
    ax_best.set_xscale("log", base=2)

    # if tick_th is not None:
    #     ax_best.set_xticks(tick_th)
    # ax_best.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    fig.legend(
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.1),
        frameon=False,
    )

    plt.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(filename)) or "."
    os.makedirs(out_dir, exist_ok=True)
    try:
        plt.savefig(filename, bbox_inches="tight")
    except Exception as e:
        logging.warning("Failed to save plot_all_curves_delta to %s (error: %s). Trying fallback by disabling text.usetex.", filename, e)
        prev_usetex = plt.rcParams.get("text.usetex", False)
        try:
            plt.rcParams["text.usetex"] = False
            plt.savefig(filename, bbox_inches="tight")
        finally:
            plt.rcParams["text.usetex"] = prev_usetex
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

import pandas as pd

def plot_difficulty_composition(diff_distributions, boundaries):
    sanitized_dists = []
    for dist in diff_distributions:
        new_dist = {}
        for k, v in dist.items():
            key_name = "Unknown" if k is None or k == "" else k
            new_dist[key_name] = new_dist.get(key_name, 0) + v
        sanitized_dists.append(new_dist)

    df = pd.DataFrame(sanitized_dists).fillna(0)
    
    total_counts = df.sum(axis=1)

    df_percent = df.div(total_counts, axis=0) * 100

    ax = df_percent.plot(
        kind='bar', 
        stacked=True, 
        figsize=(10, 6), 

        colormap='tab10', 
        edgecolor='black',
        alpha=0.85
    )
    
    for i, count in enumerate(total_counts):
        ax.text(i, 102, f"N={int(count)}", ha='center', fontsize=9, fontweight='bold')

    plt.title("Question Difficulty Composition", fontsize=14)
    plt.xlabel("Test-time compute ($\\theta$)", fontsize=12)
    plt.ylabel("Percentage of Questions (%)", fontsize=12)
    plt.ylim(0, 110)
    plt.legend(title="Difficulty", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

import json

def reasoning_accuracy_curve_per_question_quantiles(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
    difficulty_path=None,
):
    writers_diff = []
    if difficulty_path is not None:
        try:
            with open(difficulty_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    writers_diff.append(item.get("writer_difficulty", "Unknown"))
        except Exception as e:
            print(f"Warning: Could not load difficulty file: {e}")
            writers_diff = [None] * len(model_results["correct"])
    else:
        writers_diff = [None] * len(model_results["correct"])

    if len(writers_diff) != len(model_results["correct"]):
        min_len = min(len(writers_diff), len(model_results["correct"]))
        writers_diff = writers_diff[:min_len]

    rng = np.random.default_rng(rng_seed)
    num_bins = len(qs) - 1

    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]
    all_bin_difs = [[] for _ in range(num_bins)]

    iterator = zip(model_results["correct"], model_results["num_tokens"], writers_diff)


    num_used = 0

    for correct_list, token_list, q_diff in iterator:
        
        num_correct = sum(1 for c in correct_list if c is True)

        if not (num_correct > 5 and num_correct < 25):
            continue

        num_used += 1
        q_thinks = []
        q_corrs = []
        q_difs = []

        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[0] > 5 and t[1] < 2040:
                q_thinks.append(t[0] / t[1])
                q_corrs.append(c)
                q_difs.append(q_diff) 
        
        if len(q_thinks) < 1:
            continue

        q_thinks = np.asarray(q_thinks, dtype=float)
        q_corrs = np.asarray(q_corrs, dtype=float)
        q_difs = np.asarray(q_difs, dtype=float)

        q_thinks_jittered = q_thinks + rng.uniform(-0.1, 0.1, size=q_thinks.shape)
        
        local_boundaries = np.quantile(q_thinks_jittered, qs)

        for b in range(num_bins):
            lo, hi = local_boundaries[b], local_boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered <= hi)
            else:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered < hi)

            if mask.any():
                all_bin_accs[b].extend(q_corrs[mask])
                all_bin_think[b].extend(q_thinks[mask])
                all_bin_difs[b].extend(q_difs[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []
    diff_distributions = []

    for b in range(num_bins):
        data_acc = np.array(all_bin_accs[b], dtype=float)
        if data_acc.size > 0:
            acc_means.append(np.mean(data_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(data_acc, alpha=alpha, B=B, rng=rng))
        else:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))

        data_think = np.array(all_bin_think[b], dtype=float)
        if data_think.size > 0:
            think_means.append(np.mean(data_think))
            think_errs.append(_ci_percentile_mean_bootstrap(data_think, alpha=alpha, B=B, rng=rng))
        else:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))

        data_diff = all_bin_difs[b]
        if len(data_diff) > 0:
            sanitized_diffs = ["Unknown" if d is None else str(d) for d in data_diff]
            unique, counts = np.unique(sanitized_diffs, return_counts=True)
            diff_distributions.append(dict(zip(unique, counts)))
        else:
            diff_distributions.append({})

    print(f"Used {num_used} questions for analysis.")

    return acc_means, acc_errs, think_means, think_errs, diff_distributions, None

def reasoning_accuracy_curve_second_custom_2_boxed(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
):
    rng = np.random.default_rng(rng_seed)

    accuracy_list = []

    for token_list, correct_list in zip(model_results["num_tokens"], model_results["correct"]):
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue
        accuracy_list.append(num_correct / len(valid_tokens))


        for t, c in zip(token_list, correct_list):
            if not (isinstance(t, list) and len(t) == 2 and t[1] < 2048) and c is True:
                print(f"Inconsistent data found. Token info: {t}, Correct: {c}")
                # raise ValueError("Inconsistent data: correct answer with invalid token info.")

    print(accuracy_list)
    quantile_25 = min(max(0.2, np.quantile(accuracy_list, 0.25)), 0.8)
    quantile_75 = min(np.quantile(accuracy_list, 0.75), 0.9)

    accuracy_list_sorted = sorted(accuracy_list)

    plt.hist(accuracy_list_sorted, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(quantile_25, color='red', linestyle='dashed', linewidth=1, label='25th Percentile')
    plt.axvline(quantile_75, color='green', linestyle='dashed', linewidth=1, label='75th Percentile')
    plt.legend()
    plt.show()

    print(f"25th percentile accuracy: {quantile_25:.3f}, 75th percentile accuracy: {quantile_75:.3f}")

    all_think_values = []

    num_used = 0

    for token_list, correct_list in zip(model_results["num_tokens"], model_results["correct"]):
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue

        if not (num_correct / len(valid_tokens) >= quantile_25 and num_correct / len(valid_tokens) <= quantile_75):
            continue
        num_used += 1

        for t in token_list:
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                all_think_values.append(t[0]/t[1])

    print(f"Used {num_used} questions for analysis.")
    num_bins = len(qs) - 1
    if len(all_think_values) == 0:
        nan_vec = [np.nan] * num_bins
        nan_err = [(np.nan, np.nan)] * num_bins
        return nan_vec, nan_err, nan_vec, nan_err, [{} for _ in range(num_bins)], []

    all_think_values = np.asarray(all_think_values, dtype=float)
    boundaries = np.quantile(all_think_values, qs)

    print("Boundaries:", boundaries)
    
    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]

    iterator = zip(model_results["correct"], model_results["num_tokens"])

    for correct_list, token_list in iterator:
        q_thinks = []
        q_corrs = []

        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue

        if not (num_correct / len(valid_tokens) >= quantile_25 and num_correct / len(valid_tokens) <= quantile_75):
            continue

        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                q_thinks.append(t[0]/t[1])
                q_corrs.append(c)
        
        if not q_thinks:
            continue

        q_thinks = np.asarray(q_thinks, dtype=float)
        q_corrs = np.asarray(q_corrs, dtype=float)

        for b in range(num_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_thinks >= lo) & (q_thinks <= hi)
            else:
                mask = (q_thinks >= lo) & (q_thinks < hi)

            if mask.any():
                all_bin_accs[b].extend(q_corrs[mask])
                all_bin_think[b].extend(q_thinks[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []

    for b in range(num_bins):
        data_acc = np.array(all_bin_accs[b], dtype=float)
        if data_acc.size > 0:
            acc_means.append(np.mean(data_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(data_acc, alpha=alpha, B=B, rng=rng))
        else:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))

        data_think = np.array(all_bin_think[b], dtype=float)
        if data_think.size > 0:
            think_means.append(np.mean(data_think))
            think_errs.append(_ci_percentile_mean_bootstrap(data_think, alpha=alpha, B=B, rng=rng))
        else:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))

    return acc_means, acc_errs, think_means, think_errs, all_think_values

def reasoning_accuracy_curve_second_custom_2(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
    difficulty_path=None,
):
    writers_diff = []
    if difficulty_path is not None:
        try:
            with open(difficulty_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    writers_diff.append(item.get("writer_difficulty", "Unknown"))
        except Exception as e:
            print(f"Warning: Could not load difficulty file: {e}")
            writers_diff = [None] * len(model_results["correct"])
    else:
        writers_diff = [None] * len(model_results["correct"])

    if len(writers_diff) != len(model_results["correct"]):
        print(f"Warning: Diff len ({len(writers_diff)}) != Results len ({len(model_results['correct'])}). truncating to shorter.")
        min_len = min(len(writers_diff), len(model_results["correct"]))
        writers_diff = writers_diff[:min_len]

    rng = np.random.default_rng(rng_seed)

    accuracy_list = []

    for token_list, correct_list in zip(model_results["num_tokens"], model_results["correct"]):
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue
        accuracy_list.append(num_correct / len(valid_tokens))


        for t, c in zip(token_list, correct_list):
            if not (isinstance(t, list) and len(t) == 2 and t[1] < 2048) and c is True:
                print(f"Inconsistent data found. Token info: {t}, Correct: {c}")
                # raise ValueError("Inconsistent data: correct answer with invalid token info.")

    print(accuracy_list)
    quantile_25 = max(0.2, np.quantile(accuracy_list, 0.25))
    quantile_75 = min(np.quantile(accuracy_list, 0.75), 0.9)

    accuracy_list_sorted = sorted(accuracy_list)

    plt.hist(accuracy_list_sorted, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(quantile_25, color='red', linestyle='dashed', linewidth=1, label='25th Percentile')
    plt.axvline(quantile_75, color='green', linestyle='dashed', linewidth=1, label='75th Percentile')
    plt.legend()
    plt.show()

    print(f"25th percentile accuracy: {quantile_25:.3f}, 75th percentile accuracy: {quantile_75:.3f}")

    all_think_values = []

    num_used = 0

    for token_list, correct_list in zip(model_results["num_tokens"], model_results["correct"]):
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue

        if not (num_correct / len(valid_tokens) >= quantile_25 and num_correct / len(valid_tokens) <= quantile_75):
            continue
        num_used += 1

        for t in token_list:
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                all_think_values.append(t[0]/t[1])

    print(f"Used {num_used} questions for analysis.")
    num_bins = len(qs) - 1
    if len(all_think_values) == 0:
        nan_vec = [np.nan] * num_bins
        nan_err = [(np.nan, np.nan)] * num_bins
        return nan_vec, nan_err, nan_vec, nan_err, [{} for _ in range(num_bins)], []

    all_think_values = np.asarray(all_think_values, dtype=float)
    boundaries = np.quantile(all_think_values, qs)

    print("Boundaries:", boundaries)
    
    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]
    all_bin_difs = [[] for _ in range(num_bins)]

    iterator = zip(model_results["correct"], model_results["num_tokens"], writers_diff)

    for correct_list, token_list, q_diff in iterator:
        q_thinks = []
        q_corrs = []
        q_difs = []

        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue

        if not (num_correct / len(valid_tokens) >= quantile_25 and num_correct / len(valid_tokens) <= quantile_75):
            continue

        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                q_thinks.append(t[0]/t[1])
                q_corrs.append(c)
                q_difs.append(q_diff) 
        
        if not q_thinks:
            continue

        q_thinks = np.asarray(q_thinks, dtype=float)
        q_corrs = np.asarray(q_corrs, dtype=float)
        q_difs = np.asarray(q_difs) 

        for b in range(num_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_thinks >= lo) & (q_thinks <= hi)
            else:
                mask = (q_thinks >= lo) & (q_thinks < hi)

            if mask.any():
                all_bin_accs[b].extend(q_corrs[mask])
                all_bin_think[b].extend(q_thinks[mask])
                all_bin_difs[b].extend(q_difs[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []
    diff_distributions = []

    for b in range(num_bins):
        data_acc = np.array(all_bin_accs[b], dtype=float)
        if data_acc.size > 0:
            acc_means.append(np.mean(data_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(data_acc, alpha=alpha, B=B, rng=rng))
        else:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))

        data_think = np.array(all_bin_think[b], dtype=float)
        if data_think.size > 0:
            think_means.append(np.mean(data_think))
            think_errs.append(_ci_percentile_mean_bootstrap(data_think, alpha=alpha, B=B, rng=rng))
        else:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))

        data_diff = all_bin_difs[b]
        if len(data_diff) > 0:
            sanitized_diffs = ["Unknown" if d is None else str(d) for d in data_diff]
            
            unique, counts = np.unique(sanitized_diffs, return_counts=True)
            diff_distributions.append(dict(zip(unique, counts)))
        else:
            diff_distributions.append({})

    return acc_means, acc_errs, think_means, think_errs, diff_distributions, boundaries

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

    all_think = []
    cnt = 0
    sum_cor = 0


    for token_list, corr in zip(model_results["num_tokens"], model_results["correct"]):
        for t, c in zip(token_list, corr):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2045:
                all_think.append(t[1])
            else:
                sum_cor += 1 if c else 0
                cnt += 1
    print(f"Skipped {cnt} invalid token entries., accuracy on those: {sum_cor}/{cnt} = {sum_cor/cnt if cnt>0 else 0.0:.4f}")

    num_bins = len(qs) - 1
    if len(all_think) == 0:
        nan_vec = [np.nan] * num_bins
        nan_err = [(np.nan, np.nan)] * num_bins
        return nan_vec, nan_err, nan_vec, nan_err, []

    all_think = np.asarray(all_think, dtype=float)
    
    jitter_all = all_think + rng.uniform(-0.1, 0.1, size=all_think.shape)
    boundaries = np.quantile(jitter_all, qs)

    print(boundaries)

    bin_question_means = [[] for _ in range(num_bins)]
    bin_question_thinks = [[] for _ in range(num_bins)]

    for correct_list, token_list in zip(model_results["correct"], model_results["num_tokens"]):
        q_think = []
        q_corr = []
        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2045:
                q_think.append(t[1])
                q_corr.append(c)
        
        if not q_think:
            continue

        q_think = np.asarray(q_think, dtype=float)
        q_corr = np.asarray(q_corr, dtype=float)
        
        q_think_jittered = q_think + rng.uniform(-0.1, 0.1, size=q_think.shape)

        for b in range(num_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_think_jittered >= lo) & (q_think_jittered <= hi)
            else:
                mask = (q_think_jittered >= lo) & (q_think_jittered < hi)

            if mask.any():
                bin_question_means[b].extend(q_corr[mask])
                bin_question_thinks[b].extend(q_think[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []

    for b in range(num_bins):
        qs_acc = np.array(bin_question_means[b], dtype=float)
        
        if qs_acc.size == 0:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))
        else:
            acc_means.append(np.mean(qs_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(qs_acc, alpha=alpha, B=B, rng=rng))

        qs_th = np.array(bin_question_thinks[b], dtype=float)
        
        if qs_th.size == 0:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))
        else:
            think_means.append(np.mean(qs_th))
            think_errs.append(_ci_percentile_mean_bootstrap(qs_th, alpha=alpha, B=B, rng=rng))

    return acc_means, acc_errs, think_means, think_errs, all_think

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
    ax_acc.grid(True)

    ax_tok.plot(bins, tok_means, marker="o", markersize=4, linewidth=1.2, color=color)
    ax_tok.fill_between(bins, tok_means - tok_lo, tok_means + tok_hi, alpha=0.15, color=color, edgecolor="none")
    ax_tok.set_xlabel("Reasoning Level")
    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xticks(bins)
    ax_tok.grid(True)

    if num_questions is not None:
        fig.suptitle(f"{title}\nQuestions: {num_questions}", y=0.98)
    else:
        fig.suptitle(title, y=0.98)

    plt.tight_layout()
    if filename is not None:
        out_dir = os.path.dirname(os.path.abspath(filename)) or "."
        os.makedirs(out_dir, exist_ok=True)
        try:
            plt.savefig(filename, bbox_inches="tight")
        except Exception as e:
            logging.warning("Failed to save reasoning curves to %s (error: %s). Trying fallback by disabling text.usetex.", filename, e)
            prev_usetex = plt.rcParams.get("text.usetex", False)
            try:
                plt.rcParams["text.usetex"] = False
                plt.savefig(filename, bbox_inches="tight")
            finally:
                plt.rcParams["text.usetex"] = prev_usetex
    plt.show()

def reasoning_accuracy_curve_final(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
    difficulty_path=None,
):
    writers_diff = []
    if difficulty_path is not None:
        try:
            with open(difficulty_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    writers_diff.append(item.get("writer_difficulty", "Unknown"))
        except Exception as e:
            print(f"Warning: Could not load difficulty file: {e}")
            writers_diff = [None] * len(model_results["correct"])
    else:
        writers_diff = [None] * len(model_results["correct"])

    if len(writers_diff) != len(model_results["correct"]):
        print(f"Warning: Diff len ({len(writers_diff)}) != Results len ({len(model_results['correct'])}). truncating to shorter.")
        min_len = min(len(writers_diff), len(model_results["correct"]))
        writers_diff = writers_diff[:min_len]

    rng = np.random.default_rng(rng_seed)

    accuracy_list = []
    
    for token_list, correct_list in zip(model_results["num_tokens"], model_results["correct"]):
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        
        if len(valid_tokens) == 0:
            continue
            
        accuracy_list.append(num_correct / len(valid_tokens))
    
    quantile_25 = min(max(0.2, np.quantile(accuracy_list, 0.25)), 0.8)
    quantile_75 = min(np.quantile(accuracy_list, 0.75), 0.9)

    accuracy_list_sorted = sorted(accuracy_list)
    plt.figure(figsize=(8, 4))
    plt.hist(accuracy_list_sorted, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(quantile_25, color='red', linestyle='dashed', linewidth=1, label=f'25th ({quantile_25:.2f})')
    plt.axvline(quantile_75, color='green', linestyle='dashed', linewidth=1, label=f'75th ({quantile_75:.2f})')
    plt.title("Per-Question Accuracy Distribution")
    plt.legend()
    plt.show()

    all_think_values = []
    num_used = 0

    iter_data = zip(model_results["num_tokens"], model_results["correct"])
    for token_list, correct_list in iter_data:
        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        
        if len(valid_tokens) == 0:
            continue

        acc = num_correct / len(valid_tokens)
        
        if not (acc >= quantile_25 and acc <= quantile_75):
            continue
            
        num_used += 1

        for t in token_list:
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                all_think_values.append(t[0] / t[1])

    print(f"Used {num_used} questions for analysis (out of {len(model_results['correct'])}).")

    num_bins = len(qs) - 1
    if len(all_think_values) == 0:
        nan_vec = [np.nan] * num_bins
        nan_err = [(np.nan, np.nan)] * num_bins
        if difficulty_path is not None:
            return nan_vec, nan_err, nan_vec, nan_err, [{} for _ in range(num_bins)], []
        else:
            return nan_vec, nan_err, nan_vec, nan_err, []

    all_think_values = np.asarray(all_think_values, dtype=float)
    jitter_all = all_think_values + rng.uniform(0, 1e-5, size=all_think_values.shape)
    jitter_all = np.clip(jitter_all, 0, 1)
    boundaries = np.quantile(jitter_all, qs)

    print("Boundaries (Think/Total Ratio):", boundaries)

    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]
    all_bin_difs = [[] for _ in range(num_bins)]

    iterator = zip(model_results["correct"], model_results["num_tokens"], writers_diff)

    for correct_list, token_list, q_diff in iterator:
        q_thinks = []
        q_corrs = []
        q_difs = []

        num_correct = sum(1 for c in correct_list if c is True)
        valid_tokens = [t for t in token_list if isinstance(t, list) and len(t) == 2 and t[1] < 2048]
        if len(valid_tokens) == 0:
            continue

        acc = num_correct / len(valid_tokens)
        if not (acc >= quantile_25 and acc <= quantile_75):
            continue

        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048:
                q_thinks.append(t[0] / t[1])
                q_corrs.append(c)
                q_difs.append(q_diff)
        
        if not q_thinks:
            continue

        q_thinks = np.asarray(q_thinks, dtype=float)
        q_corrs = np.asarray(q_corrs, dtype=float)
        q_difs = np.asarray(q_difs) 

        q_thinks_jittered = q_thinks + rng.uniform(0, 1e-5, size=q_thinks.shape)
        q_thinks_jittered = np.clip(q_thinks_jittered, boundaries[0], boundaries[-1])

        for b in range(num_bins):
            lo, hi = boundaries[b], boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered <= hi)
            else:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered < hi)

            if mask.any():
                all_bin_accs[b].extend(q_corrs[mask])
                all_bin_think[b].extend(q_thinks[mask])
                if difficulty_path is not None:
                    all_bin_difs[b].extend(q_difs[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []
    diff_distributions = []

    for b in range(num_bins):
        data_acc = np.array(all_bin_accs[b], dtype=float)
        if data_acc.size > 0:
            acc_means.append(np.mean(data_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(data_acc, alpha=alpha, B=B, rng=rng))
        else:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))

        data_think = np.array(all_bin_think[b], dtype=float)
        if data_think.size > 0:
            think_means.append(np.mean(data_think))
            think_errs.append(_ci_percentile_mean_bootstrap(data_think, alpha=alpha, B=B, rng=rng))
        else:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))

        if difficulty_path is not None:
            data_diff = all_bin_difs[b]
            if len(data_diff) > 0:
                sanitized_diffs = ["Unknown" if d is None else str(d) for d in data_diff]
                unique, counts = np.unique(sanitized_diffs, return_counts=True)
                diff_distributions.append(dict(zip(unique, counts)))
            else:
                diff_distributions.append({})

    if difficulty_path is not None:
        return acc_means, acc_errs, think_means, think_errs, diff_distributions, boundaries
    else:
        return acc_means, acc_errs, think_means, think_errs, all_think_values
    

def get_fig_dim(width, fraction=1, aspect_ratio=None):

    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        # If not specified, set the aspect ratio equal to the Golden ratio (https://en.wikipedia.org/wiki/Golden_ratio)
        aspect_ratio = (1 + 5**.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    font_serif: string, optional
                Set the desired font family
    mathtext_font: float, optional
        Set the desired math font family
    font_size: int, optional
        Set the large font size
    small_font_size: int, optional
        Set the small font size
    usetex: boolean, optional
        Use tex for strings
    """

    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
            
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        
        # Optionally set a smaller font size for legends and tick labels
        'legend.fontsize': small_font_size,
        'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size,
        'ytick.labelsize': small_font_size,
        
        'text.usetex': usetex,    
        'font.family' : 'serif',
        'font.serif' : font_serif,
        'mathtext.fontset' : mathtext_font
    }

    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)

ICML_WIDTH_COL_PT = 234 
ICML_WIDTH_TEXT_PT = 486

def plot_all_models_combined(
    models_data: dict,
    width_pt: float = ICML_WIDTH_TEXT_PT, 
    colors: Dict[str, str] = None,
    log_scale: bool = True,
    filename: Optional[str] = None,
):
    latexify(font_size=10, small_font_size=8)
    
    model_names = list(models_data.keys())
    num_models = len(model_names)
    
    if num_models == 0:
        print("No models to plot.")
        return

    fig_width, fig_height = get_fig_dim(width_pt, fraction=1.0)
    figsize = (fig_width, fig_height)
    fig, axes = plt.subplots(num_models, 3, figsize=figsize, sharex=True)
    
    if num_models == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, model_name in enumerate(model_names):
        (thetas, maj_mean, maj_std, rew_mean, rew_std, tok_mean, tok_std) = models_data[model_name]

        thetas = np.asarray(thetas)
        maj_mean = np.asarray(maj_mean, dtype=float)
        rew_mean = np.asarray(rew_mean, dtype=float)
        tok_mean = np.asarray(tok_mean, dtype=float)

        maj_yerr = np.asarray(maj_std, dtype=float).T
        rew_yerr = np.asarray(rew_std, dtype=float).T
        tok_yerr = np.asarray(tok_std, dtype=float).T
        
        color = colors.get(model_name, None) if colors else None

        ax = axes[i, 0]
        ax.errorbar(thetas, maj_mean, yerr=maj_yerr, fmt="-o", capsize=3, color=color)
        ax.set_ylabel(f"{model_name}\nAccuracy")
        if i == 0: ax.set_title("Majority Vote") 

        ax = axes[i, 1]
        ax.errorbar(thetas, rew_mean, yerr=rew_yerr, fmt="-o", capsize=3, color=color)
        if i == 0: ax.set_title(r"Best-of-$N$")

        ax = axes[i, 2]
        ax.errorbar(thetas, tok_mean, yerr=tok_yerr, fmt="-o", capsize=3, color=color)
        ax.set_ylabel("Tokens")
        if i == 0: ax.set_title("Tokens")

        if i == num_models - 1:
            axes[i, 0].set_xlabel("Test-time compute ($\\theta$)")
            axes[i, 1].set_xlabel("Test-time compute ($\\theta$)")
            axes[i, 2].set_xlabel("Test-time compute ($\\theta$)")

        if log_scale:
            for col in range(3):
                axes[i, col].set_xscale("log", base=2)

    fig.tight_layout()

    if filename:
        out_path = os.path.abspath(filename)
        parent_dir = os.path.dirname(out_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        try:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved figure to {out_path}")
        except Exception as e:
            logging.warning("Failed to save figure. Trying fallback.", exc_info=e)
            prev_usetex = plt.rcParams.get("text.usetex", False)
            try:
                plt.rcParams["text.usetex"] = False
                fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            finally:
                plt.rcParams["text.usetex"] = prev_usetex
            
    fig.show()


def plot_all_models_overlaid(
    models_data: dict,
    width_pt: float = ICML_WIDTH_TEXT_PT, 
    colors: Dict[str, str] = None,
    log_scale: bool = True,
    filename: Optional[str] = None,
):
    latexify(font_size=10, small_font_size=8)
    
    model_names = list(models_data.keys())
    if not model_names:
        print("No models to plot.")
        return

    base_dim = get_fig_dim(width_pt, fraction=1.0, aspect_ratio=2)
    figsize = base_dim 

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    axes[1].sharey(axes[0])
    axes[1].tick_params(labelleft=False)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
    for model_name in model_names:
        (thetas, maj_mean, maj_std, rew_mean, rew_std, tok_mean, tok_std) = models_data[model_name]

        thetas = np.asarray(thetas)
        maj_mean = np.asarray(maj_mean, dtype=float)
        rew_mean = np.asarray(rew_mean, dtype=float)
        tok_mean = np.asarray(tok_mean, dtype=float)

        maj_yerr = np.asarray(maj_std, dtype=float).T
        rew_yerr = np.asarray(rew_std, dtype=float).T
        tok_yerr = np.asarray(tok_std, dtype=float).T
        
        color = colors.get(model_name, None) if colors else None

        clean_name = model_name.replace("_", r"\_")
        label_tex = f"\\texttt{{{clean_name}}}"

        axes[0].errorbar(
            thetas, maj_mean, yerr=maj_yerr, 
            fmt="-o", capsize=3, color=color, label=label_tex,
            alpha=0.8, markersize=3
        )
        
        axes[1].errorbar(
            thetas, rew_mean, yerr=rew_yerr, 
            fmt="-o", capsize=3, color=color,
            alpha=0.8, markersize=3
        )

        axes[2].errorbar(
            thetas, tok_mean, yerr=tok_yerr, 
            fmt="-o", capsize=3, color=color,
            alpha=0.8, markersize=3
        )


    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Majority Vote")
    axes[0].set_xlabel("Test-time compute ($\\theta$)")

    axes[1].set_title(r"Best-of-$n$")
    axes[1].set_xlabel("Test-time compute ($\\theta$)")

    axes[2].set_ylabel("Tokens")
    axes[2].set_title("Tokens")
    axes[2].set_xlabel("Test-time compute ($\\theta$)")

    if log_scale:
        for ax in axes:
            ax.set_xscale("log", base=2)

    handles, labels = axes[0].get_legend_handles_labels()
    
    fig.legend(
        handles, labels, 
        loc='lower center', 
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(model_names), 4), 
        frameon=False
    )


    fig.tight_layout(rect=[0, 0.1, 1, 1])

    if filename:
        out_path = os.path.abspath(filename)
        parent_dir = os.path.dirname(out_path) or "."
        os.makedirs(parent_dir, exist_ok=True)
        try:
            fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved figure to {out_path}")
        except Exception as e:
            logging.warning("Failed to save figure. Trying fallback.", exc_info=e)
            prev_usetex = plt.rcParams.get("text.usetex", False)
            try:
                plt.rcParams["text.usetex"] = False
                fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
            finally:
                plt.rcParams["text.usetex"] = prev_usetex
            
    plt.show()


def reasoning_accuracy_curve_relative_effort(
    model_results,
    qs=(0.0, 0.10, 0.17, 0.25, 0.50, 0.75, 1.0),
    alpha=0.05,
    B=2000,
    rng_seed=0,
    difficulty_path=None,
):

    writers_diff = []
    if difficulty_path is not None:
        try:
            with open(difficulty_path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    writers_diff.append(item.get("writer_difficulty", "Unknown"))
        except Exception as e:
            print(f"Warning: Could not load difficulty file: {e}")
            writers_diff = [None] * len(model_results["correct"])
    else:
        writers_diff = [None] * len(model_results["correct"])

    if len(writers_diff) != len(model_results["correct"]):
        min_len = min(len(writers_diff), len(model_results["correct"]))
        writers_diff = writers_diff[:min_len]

    rng = np.random.default_rng(rng_seed)
    num_bins = len(qs) - 1

    all_bin_accs = [[] for _ in range(num_bins)]
    all_bin_think = [[] for _ in range(num_bins)]
    all_bin_difs = [[] for _ in range(num_bins)]
    
    all_raw_think_values = []

    iterator = zip(model_results["correct"], model_results["num_tokens"], writers_diff)

    for correct_list, token_list, q_diff in iterator:
        q_thinks = []  
        q_totals = []  
        q_corrs = []
        q_difs = []

        for c, t in zip(correct_list, token_list):
            if isinstance(t, list) and len(t) == 2 and t[1] < 2048 and t[0] > 5:
                q_thinks.append(t[0])  
                q_totals.append(t[1]) 
                q_corrs.append(c)
                q_difs.append(q_diff)
                all_raw_think_values.append(t[0]) 

        if not q_thinks:
            continue

        q_thinks = np.asarray(q_thinks, dtype=float)
        q_totals = np.asarray(q_totals, dtype=float)
        q_corrs = np.asarray(q_corrs, dtype=float)
        q_difs = np.asarray(q_difs)
        
        q_thinks_jittered = q_thinks + rng.uniform(0, 1e-5, size=q_thinks.shape)
        
        local_boundaries = np.quantile(q_thinks_jittered, qs)
        
        for i in range(1, len(local_boundaries)):
            if local_boundaries[i] <= local_boundaries[i-1]:
                local_boundaries[i] = local_boundaries[i-1] + 1e-9
        
        q_thinks_jittered = np.clip(q_thinks_jittered, local_boundaries[0], local_boundaries[-1])

        for b in range(num_bins):
            lo, hi = local_boundaries[b], local_boundaries[b + 1]
            
            if b == num_bins - 1:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered <= hi)
            else:
                mask = (q_thinks_jittered >= lo) & (q_thinks_jittered < hi)

            if mask.any():
                all_bin_accs[b].extend(q_corrs[mask])
                all_bin_think[b].extend(q_totals[mask]) 
                if difficulty_path is not None:
                    all_bin_difs[b].extend(q_difs[mask])

    acc_means = []
    acc_errs = []
    think_means = []
    think_errs = []
    diff_distributions = []

    for b in range(num_bins):
        data_acc = np.array(all_bin_accs[b], dtype=float)
        if data_acc.size > 0:
            acc_means.append(np.mean(data_acc))
            acc_errs.append(_ci_percentile_mean_bootstrap(data_acc, alpha=alpha, B=B, rng=rng))
        else:
            acc_means.append(np.nan)
            acc_errs.append((np.nan, np.nan))

        data_think = np.array(all_bin_think[b], dtype=float)
        if data_think.size > 0:
            think_means.append(np.mean(data_think))
            think_errs.append(_ci_percentile_mean_bootstrap(data_think, alpha=alpha, B=B, rng=rng))
        else:
            think_means.append(np.nan)
            think_errs.append((np.nan, np.nan))

        if difficulty_path is not None:
            data_diff = all_bin_difs[b]
            if len(data_diff) > 0:
                sanitized_diffs = ["Unknown" if d is None else str(d) for d in data_diff]
                unique, counts = np.unique(sanitized_diffs, return_counts=True)
                diff_distributions.append(dict(zip(unique, counts)))
            else:
                diff_distributions.append({})

    if difficulty_path is not None:
        return acc_means, acc_errs, think_means, think_errs, diff_distributions, None
    else:
        return acc_means, acc_errs, think_means, think_errs, all_raw_think_values
    

def plot_reasoning_curves_accuracy_and_tokens_new(
    acc_means,
    acc_errs,
    tok_means,
    tok_errs,
    title="Reasoning curves",
    num_questions=None,
    filename=None,
    color=None,
    width_pt=ICML_WIDTH_TEXT_PT,
):
    latexify(font_size=10, small_font_size=8)

    acc_means = np.asarray(acc_means, dtype=float)
    tok_means = np.asarray(tok_means, dtype=float)

    n = acc_means.size
    if tok_means.size != n:
        raise ValueError(f"tok_means length {tok_means.size} != acc_means length {n}")

    bins = np.arange(1, n + 1)

    if color is None:
        color = 'tab:blue' 

    acc_lo, acc_hi = _split_asym_err(acc_errs, n)
    tok_lo, tok_hi = _split_asym_err(tok_errs, n)


    fig_width, fig_height = get_fig_dim(width_pt, fraction=1.0)
    figsize = (fig_width, fig_height)

    fig, (ax_acc, ax_tok) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax_acc.plot(bins, acc_means, marker="o", markersize=4, linewidth=1.2, color=color)
    ax_acc.fill_between(
        bins, 
        acc_means - acc_lo, 
        acc_means + acc_hi, 
        alpha=0.15, 
        color=color, 
        edgecolor="none"
    )
    ax_acc.set_ylabel("Accuracy")

    ax_tok.plot(bins, tok_means, marker="o", markersize=4, linewidth=1.2, color=color)
    ax_tok.fill_between(
        bins, 
        tok_means - tok_lo, 
        tok_means + tok_hi, 
        alpha=0.15, 
        color=color, 
        edgecolor="none"
    )
    ax_tok.set_xlabel("Reasoning Level")
    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xticks(bins)

    if num_questions is not None:
        full_title = f"{title}\nQuestions: {num_questions}"
    else:
        full_title = title
    

    fig.suptitle(full_title, y=0.98)

    plt.tight_layout()

    if filename is not None:
        out_dir = os.path.dirname(os.path.abspath(filename)) or "."
        os.makedirs(out_dir, exist_ok=True)
        try:
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved figure to {filename}")
        except Exception as e:
            logging.warning("Failed to save reasoning curves to %s (error: %s). Trying fallback by disabling text.usetex.", filename, e)
            prev_usetex = plt.rcParams.get("text.usetex", False)
            try:
                plt.rcParams["text.usetex"] = False
                plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
            finally:
                plt.rcParams["text.usetex"] = prev_usetex
    
    plt.show()


def plot_reasoning_curves_overlaid(
    models_data: dict,
    title="Reasoning curves",
    colors: Dict[str, str] = None,
    width_pt=ICML_WIDTH_TEXT_PT,
    filename=None,
):
    latexify(font_size=10, small_font_size=8)
    
    if not models_data:
        print("No data to plot.")
        return

    fig_width, fig_height = get_fig_dim(width_pt, fraction=1.0, aspect_ratio=2)
    figsize = (fig_width, fig_height)

    fig, (ax_acc, ax_tok) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for i, (model_name, data) in enumerate(models_data.items()):
        name_mapping = {
            "reason-R1-D-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
            "reason-R1-D-Qwen-1.5B": "DeepSeek-R1-Distill-Qwen-1.5B",
            "reason-R1-D-Qwen-7B": "DeepSeek-R1-Distill-Qwen-7B"
        }

        model_name = name_mapping.get(model_name, model_name)

        acc_means, acc_errs, tok_means, tok_errs = data
        
        acc_means = np.asarray(acc_means, dtype=float)
        tok_means = np.asarray(tok_means, dtype=float)
        
        n = acc_means.size
        bins = np.arange(1, n + 1)
        
        if colors and model_name in colors:
            c = colors[model_name]
        else:
            c = default_colors[i % len(default_colors)]


        acc_lo, acc_hi = _split_asym_err(acc_errs, n)
        tok_lo, tok_hi = _split_asym_err(tok_errs, n)
        
        acc_yerr = np.vstack([acc_lo, acc_hi])
        tok_yerr = np.vstack([tok_lo, tok_hi])

        clean_name = model_name.replace("_", r"\_")
        label_tex = f"\\texttt{{{clean_name}}}"

        ax_acc.errorbar(
            bins, acc_means, yerr=acc_yerr,
            fmt="-o", linewidth=1.2, capsize=3,
            color=c, label=label_tex, alpha=0.9, markersize=3
        )

        ax_tok.errorbar(
            bins, tok_means, yerr=tok_yerr,
            fmt="-o", linewidth=1.2, capsize=3,
            color=c, alpha=0.9, markersize=3
        )

    for ax in [ax_acc, ax_tok]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    ax_acc.set_ylabel("Accuracy")
    
    ax_tok.set_ylabel("Reasoning Tokens")
    ax_tok.set_xlabel("Test-time compute ($\\theta$)")
    ax_tok.set_xticks(bins) 
    
    handles, labels = ax_acc.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01), 
        ncol=min(len(models_data), 3),
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])

    if filename is not None:
        out_dir = os.path.dirname(os.path.abspath(filename)) or "."
        os.makedirs(out_dir, exist_ok=True)
        try:
            plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
            print(f"Saved figure to {filename}")
        except Exception as e:
            logging.warning("Failed to save figure. Trying fallback.", exc_info=e)
            prev_usetex = plt.rcParams.get("text.usetex", False)
            try:
                plt.rcParams["text.usetex"] = False
                plt.savefig(filename, bbox_inches="tight", pad_inches=0.05)
            finally:
                plt.rcParams["text.usetex"] = prev_usetex
    
    plt.show()

def save_separated_plots_overlaid(
    models_data: dict,
    base_filename: str,
    width_pt: float = 487.8225, # ICML text width
    colors: Dict[str, str] = None,
    log_scale: bool = True,
    ylim_accuracy: Optional[Tuple[float, float]] = None,
    ylim_tokens: Optional[Tuple[float, float]] = None,
    samples_disp = True,
):
    # Set font sizes
    latexify(font_size=10, small_font_size=8) 
    
    model_names = list(models_data.keys())
    if not model_names:
        print("No models to plot.")
        return

    width_per_plot_pt = width_pt / 3
    

    fig_width_in = width_per_plot_pt / 72.27
    fig_height_in = fig_width_in * 0.9 
    figsize = (fig_width_in, fig_height_in)


    margins = {
        'left': 0.32,   
        'right': 0.99,
        'top': 0.98,
        'bottom': 0.23
    }

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    if ylim_accuracy is not None:
        common_ylim = ylim_accuracy
    else:
        all_acc_values = []
        for m in model_names:
            all_acc_values.extend(models_data[m][1] * 100) 
            all_acc_values.extend(models_data[m][3] * 100) 
        y_min, y_max = min(all_acc_values), max(all_acc_values)
        y_range = y_max - y_min
        common_ylim = (max(0, y_min - 0.05 * y_range), min(100, y_max + 0.05 * y_range))

        ylim_accuracy = common_ylim

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        if log_scale:
            ax.set_xscale("log", base=2)
            
            ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=4))
            

    fig_maj, ax_maj = plt.subplots(figsize=figsize)
    fig_maj.subplots_adjust(**margins)
    style_ax(ax_maj)
    
    legend_handles = []
    legend_labels = []

    for i, model_name in enumerate(model_names):
        (thetas, maj_mean, maj_std, _, _, _, _) = models_data[model_name]
        
        maj_mean_pct = maj_mean * 100
        maj_std_pct = np.asarray(maj_std).T * 100
        
        c = colors.get(model_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
        clean_name = model_name.replace("_", r"\_")
        label_tex = f"\\texttt{{{clean_name}}}"
        
        line = ax_maj.errorbar(
            thetas, maj_mean_pct, yerr=maj_std_pct,
            fmt="-o", capsize=3, color=c, label=label_tex,
            alpha=0.8, markersize=3
        )
        legend_handles.append(line)
        legend_labels.append(label_tex)

    ax_maj.set_ylabel(r"Accuracy (\%)")
    if samples_disp:
        ax_maj.set_xlabel("Test-time compute ($\\theta$)")
    else:
        ax_maj.set_xticklabels([]) 

    ax_maj.set_ylim(common_ylim)
    
    out_maj = f"{base_filename}_majority.pdf"
    fig_maj.savefig(out_maj, pad_inches=0)
    plt.close(fig_maj)

    fig_bon, ax_bon = plt.subplots(figsize=figsize)
    fig_bon.subplots_adjust(**margins) 
    style_ax(ax_bon)

    for i, model_name in enumerate(model_names):
        (thetas, _, _, rew_mean, rew_std, _, _) = models_data[model_name]
        
        rew_mean_pct = rew_mean * 100
        rew_std_pct = np.asarray(rew_std).T * 100
        c = colors.get(model_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
        
        ax_bon.errorbar(
            thetas, rew_mean_pct, yerr=rew_std_pct,
            fmt="-o", capsize=3, color=c,
            alpha=0.8, markersize=3
        )

    ax_bon.set_ylabel(r"Accuracy (\%)")
    
    if samples_disp:
        ax_bon.set_xlabel("Test-time compute ($\\theta$)")
    else:
        ax_bon.set_xticklabels([])
    ax_bon.set_ylim(common_ylim)

    out_bon = f"{base_filename}_best_of_n.pdf"
    fig_bon.savefig(out_bon, pad_inches=0)
    plt.close(fig_bon)



    fig_tok, ax_tok = plt.subplots(figsize=figsize)
    fig_tok.subplots_adjust(**margins)
    style_ax(ax_tok)

    for i, model_name in enumerate(model_names):
        (thetas, _, _, _, _, tok_mean, tok_std) = models_data[model_name]
        c = colors.get(model_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
        
        ax_tok.errorbar(
            thetas, tok_mean, yerr=np.asarray(tok_std).T,
            fmt="-o", capsize=3, color=c,
            alpha=0.8, markersize=3
        )

    ax_tok.set_ylabel("Tokens")
    if samples_disp:
        ax_tok.set_xlabel("Test-time compute ($\\theta$)")
    else:
        ax_tok.set_xticklabels([])

    if ylim_tokens is not None:
        ax_tok.set_ylim(ylim_tokens)
    else:
        all_tok_values = []
        for m in model_names:
            all_tok_values.extend(models_data[m][5]) 
        y_min, y_max = min(all_tok_values), max(all_tok_values)
        y_range = y_max - y_min
        ylim_tokens = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        ax_tok.set_ylim(ylim_tokens)

    ax_tok.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

    out_tok = f"{base_filename}_tokens.pdf"
    fig_tok.savefig(out_tok, pad_inches=0)
    plt.close(fig_tok)


    leg_width_inch = width_pt / 72.27
    
    num_models = len(model_names)
    cols = min(num_models, 4)
    rows = (num_models + cols - 1) // cols
    
    height_per_row_inch = 0.25 
    leg_height_inch = rows * height_per_row_inch
    
    fig_leg = plt.figure(figsize=(leg_width_inch, leg_height_inch)) 
    
    reordered_handles = []
    reordered_labels = []
    

    for c in range(cols):
        for r in range(rows):
            idx = r * cols + c
            if idx < num_models:
                reordered_handles.append(legend_handles[idx])
                reordered_labels.append(legend_labels[idx])

    print(reordered_handles, reordered_labels)
    print("\n")
    print(legend_handles, legend_labels)

    fig_leg.legend(
        reordered_handles, reordered_labels,
        loc='center',
        ncol=cols, 
        frameon=False,
        fontsize=8 
    )
    
    out_leg = f"{base_filename}_legend.pdf"
    fig_leg.savefig(out_leg, pad_inches=0)
    plt.close(fig_leg)
    print(f"Saved 4 files to {base_filename}_*.pdf")

    return ylim_accuracy, ylim_tokens


import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Optional, Tuple

def save_reasoning_plots_separated_f(
    models_data: dict,
    base_filename: str,
    width_pt: float = 487.8225, 
    colors: Dict[str, str] = None,
):

    
    if not models_data:
        print("No data to plot.")
        return

    model_names = list(models_data.keys())
    
    width_per_plot_pt = width_pt / 2.2  
    fig_width_in = width_per_plot_pt / 72.27
    fig_height_in = fig_width_in * 0.85 
    figsize = (fig_width_in, fig_height_in)

    margins = {
        'left': 0.28,   
        'right': 0.98,  
        'top': 0.98,    
        'bottom': 0.22  
    }

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def style_ax(ax, bins):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.set_xticks(bins) 

    legend_handles = []
    legend_labels = []

    first_key = list(models_data.keys())[0]
    n_bins_global = len(models_data[first_key][0])
    global_bins = np.arange(1, n_bins_global + 1)

    def fix_yerr_shape(errs, n_points):
        yerr = np.array(errs)

        if yerr.ndim == 2 and yerr.shape == (n_points, 2):
            return yerr.T
        return yerr


    fig_acc, ax_acc = plt.subplots(figsize=figsize)
    fig_acc.subplots_adjust(**margins)
    style_ax(ax_acc, global_bins)

    for i, (model_name, data) in enumerate(models_data.items()):
        print(model_name)
        acc_means, acc_errs, _, _ = data
        
        acc_means = np.asarray(acc_means, dtype=float) * 100
        n = acc_means.size
        bins = np.arange(1, n + 1)
        
        yerr = fix_yerr_shape(acc_errs, n)

        c = colors.get(model_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]
        
        clean_name = model_name.replace("reason-", "").replace("Distill-", "D-")
        clean_name = clean_name.replace("_", r"\_")
        label_tex = f"\\texttt{{{clean_name}}}"

        line = ax_acc.errorbar(
            bins, acc_means, yerr=yerr,
            fmt="-o", linewidth=1.2, capsize=3,
            color=c, label=label_tex, alpha=0.9, markersize=3
        )
        
        legend_handles.append(line)
        legend_labels.append(label_tex)

    ax_acc.set_ylabel("Accuracy (\%)")
    ax_acc.set_xlabel("Test-time compute ($\\theta$)")
    
    out_acc = f"{base_filename}_accuracy.pdf"
    fig_acc.savefig(out_acc, pad_inches=0)
    plt.close(fig_acc)

    fig_tok, ax_tok = plt.subplots(figsize=figsize)
    fig_tok.subplots_adjust(**margins)
    style_ax(ax_tok, global_bins)

    for i, (model_name, data) in enumerate(models_data.items()):
        _, _, tok_means, tok_errs = data
        
        tok_means = np.asarray(tok_means, dtype=float)
        n = tok_means.size
        bins = np.arange(1, n + 1)
        
        # FIX: Ensure shape is (2, N)
        yerr = fix_yerr_shape(tok_errs, n)

        c = colors.get(model_name, default_colors[i % len(default_colors)]) if colors else default_colors[i % len(default_colors)]

        ax_tok.errorbar(
            bins, tok_means, yerr=yerr,
            fmt="-o", linewidth=1.2, capsize=3,
            color=c, alpha=0.9, markersize=3
        )

    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xlabel("Test-time compute ($\\theta$)")

    out_tok = f"{base_filename}_tokens.pdf"
    fig_tok.savefig(out_tok, pad_inches=0)
    plt.close(fig_tok)

    target_font_size = 8
    latex_scale_factor = 0.8  
    adjusted_fontsize = target_font_size / latex_scale_factor 
    
    leg_width_inch = (width_pt / 72.27) / latex_scale_factor
    
    num_models = len(model_names)
    cols = min(num_models, 3) 
    rows = (num_models + cols - 1) // cols
    
    height_per_row_inch = 0.25 / latex_scale_factor
    leg_height_inch = rows * height_per_row_inch
    
    fig_leg = plt.figure(figsize=(leg_width_inch, leg_height_inch)) 
    
    fig_leg.legend(
        legend_handles, 
        legend_labels, 
        loc='center',
        ncol=cols, 
        frameon=False,
        fontsize=adjusted_fontsize 
    )
    
    out_leg = f"{base_filename}_legend.pdf"
    fig_leg.savefig(out_leg, pad_inches=0)
    plt.close(fig_leg)
    
    print(f"Saved separate files: {out_acc}, {out_tok}, {out_leg}")

ICML_WIDTH_TEXT_PT = 486

def bet_pot_analysis(providers, thetas, betas_to_test, results, base_filename="market_sim", colors=None, reasoning=False):
    latexify(font_size=10, small_font_size=8)
    
    width_per_plot = ICML_WIDTH_TEXT_PT / 3.0
    figsize = get_fig_dim(width_per_plot, fraction=1.0, aspect_ratio=0.75)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_arr = plt.cm.tab20(np.linspace(0, 1, len(providers)))
    color_map = colors if colors is not None else {
        p.name: colors_arr[i % len(colors_arr)] for i, p in enumerate(providers)
    }   
    

    num_prov = len(providers)
    offsets = np.linspace(-0.25, 0.25, num_prov)
    provider_offsets = {p.name: offsets[i] for i, p in enumerate(providers)}


    all_ineff_values = []
    for beta in betas_to_test:
        _, poa_hist, _, _ = results[beta]
        ineff = (np.array(poa_hist) - 1.0) * 100
        all_ineff_values.extend(ineff)
    
    if all_ineff_values:
        g_max = max(all_ineff_values)
        ineff_ylim = (0, g_max * 1.1 if g_max > 0 else 5.0)
    else:
        ineff_ylim = (0, 10.0)

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='both', which='major', labelsize=8)

    for beta in betas_to_test:
        history, poa_hist, pot_hist, _ = results[beta]
        total_steps = len(poa_hist)
        steps = range(total_steps)
        beta_str = str(beta).replace(".", "p")

        fig_strat, ax_strat = plt.subplots(figsize=figsize)
        
        ax_strat.spines['top'].set_visible(False)
        ax_strat.get_xaxis().tick_bottom()
        ax_strat.get_yaxis().tick_left()
        ax_strat.tick_params(axis='both', which='major', labelsize=8)

        for provider_name, trajectory in history.items():
            traj_data = np.array(trajectory[:total_steps])
            jittered_data = traj_data + provider_offsets[provider_name]
            
            ax_strat.plot(
                steps, 
                jittered_data, 
                drawstyle='steps-post', 
                linewidth=1.2,
                color=color_map[provider_name],
                alpha=0.9
            )
            
        ax_strat.set_ylabel("Test-time compute $\\theta$")
        ax_strat.set_xlabel("Iteration, $t$")
        
        if reasoning == False:
            ax_strat.set_yticks(range(len(thetas)))
            power_labels = [f"$2^{{{int(np.log2(t))}}}$" for t in thetas]
            ax_strat.set_yticklabels(power_labels, fontsize=8)
            ax_strat.set_ylim(-0.4, len(thetas) - 0.6)
        else:
            # ax_strat.set_ylim(min(thetas) * 0.9, max(thetas) * 1.1)
            ax_strat.set_yticks(range(len(thetas)))
            ax_strat.set_yticklabels([f"{t}" for t in thetas], fontsize=8)


        ax_ineff = ax_strat.twinx()
        
        ineff_data = (np.array(poa_hist) - 1.0) * 100
        
        ax_ineff.plot(steps, ineff_data, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        
        ax_ineff.set_ylabel(r"Market Inefficiency (\%)", color='black', fontsize=9)
        ax_ineff.tick_params(axis='y', labelcolor='black', labelsize=8)
        
        ax_ineff.set_ylim(ineff_ylim)
        ax_ineff.spines['top'].set_visible(False) 
        out_strat = f"{base_filename}_beta{beta_str}_strategies_combined.pdf"
        fig_strat.savefig(out_strat, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {out_strat}")
        plt.close(fig_strat)


        fig_pot, ax_pot = plt.subplots(figsize=figsize)
        style_ax(ax_pot)
        
        ax_pot.plot(steps, pot_hist, color='purple', linewidth=1.5)
        
        ax_pot.set_ylabel(r"Potential $\Phi$")
        ax_pot.set_xlabel("Iteration, $t$")

        out_pot = f"{base_filename}_beta{beta_str}_potential.pdf"
        fig_pot.savefig(out_pot, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {out_pot}")
        plt.close(fig_pot)


    leg_width, _ = get_fig_dim(ICML_WIDTH_TEXT_PT, fraction=1.0)
    
    handles = [
        plt.Line2D([0], [0], color=color_map[p.name], lw=2, label=f"\\texttt{{{p.name.replace('_', r'-')}}}") 
        for p in providers
    ]
    handles.append(plt.Line2D([0], [0], color='black', linestyle='--', lw=1.5, label=r"Inefficiency (\%)"))
    
    num_items = len(handles)
    cols = min(num_items, 4)
    rows = (num_items + cols - 1) // cols

    final_handles = []
    for c in range(cols):
        for r in range(rows):
            logical_idx = r * cols + c
            if logical_idx < num_items:
                final_handles.append(handles[logical_idx])
            else:
                final_handles.append(plt.Line2D([0], [0], color='none', label=''))

    leg_height = rows * 0.25 
    
    fig_leg = plt.figure(figsize=(leg_width, leg_height))
    fig_leg.legend(
        handles=final_handles,
        loc='center',
        ncol=cols,
        frameon=False,
        fontsize=8,
        columnspacing=1.0
    )
    
    out_leg = f"{base_filename}_legend.pdf"
    fig_leg.savefig(out_leg, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved: {out_leg}")
    plt.close(fig_leg)