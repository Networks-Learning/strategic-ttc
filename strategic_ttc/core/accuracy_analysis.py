from typing import Any, Callable, Dict, Tuple
import numpy as np
from tqdm import tqdm
import json

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

    for theta in thetas:
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

def unreasoning_models_data(unreasoning_models, results, pred_fn, B=50):

    all_models_plot_data = {}

    for model_name in unreasoning_models:
        print(f"Processing {model_name}...")
        data = results[model_name]
        
        all_models_plot_data[model_name] = compute_curves_for_model_fast(
            data,
            parse_pred_fn=pred_fn,
            sample_size=B,
        )

    return all_models_plot_data

def reasoning_models_data(reasoning_models, results, B=2000):
    reasoning_plot_data = {}
    for model_name in reasoning_models:
        print(f"Processing {model_name}...")
        reasoning_plot_data[model_name] = reasoning_accuracy_curve_relative_effort(
            results[model_name], qs=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), B=B
        )
    return reasoning_plot_data