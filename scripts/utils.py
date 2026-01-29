import numpy as np
import os
from typing import Any, List, Optional, Tuple, Dict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Image

TOL_12 = [
    "#004183", "#4192E3", "#84CDE6", "#16A990", "#786B06", "#E1C84C",
    "#F7949F", "#AA3377", "#56002B", "#C5C3C3", "#6E6E6E", "#000000"
]
ICML_WIDTH_TEXT_PT = 486

def get_fig_dim(width, fraction=1, aspect_ratio=None):
    if aspect_ratio is None:
        aspect_ratio = (1 + 5**.5) / 2
    fig_width_in = (width * fraction) / 72.27
    return (fig_width_in, fig_width_in / aspect_ratio)

def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    if small_font_size is None: small_font_size = font_size
    params = {
        'backend': 'ps', 'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
        'axes.labelsize': font_size, 'axes.titlesize': font_size, 'font.size': font_size,
        'legend.fontsize': small_font_size, 'legend.title_fontsize': small_font_size,
        'xtick.labelsize': small_font_size, 'ytick.labelsize': small_font_size,
        'text.usetex': usetex, 'font.family': 'serif', 'font.serif': font_serif,
        'mathtext.fontset': mathtext_font
    }
    matplotlib.rcParams.update(params)
    plt.rcParams.update(params)

def clean_label(name: str) -> str:
    clean = name.replace("reason-", "").replace("Distill-", "D-").replace("_", r"\_")
    return f"\\texttt{{{clean}}}"

def save_fig(fig, path, dpi=None, close=True):
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02, dpi=dpi)
    if close:
        plt.close(fig)

def save_sanity_check(fig, base_filename, suffix="check_one"):
    path = f"{base_filename}{suffix}.png"
    save_fig(fig, path, dpi=150, close=True)
    display(Image(filename=path))

def style_axis(ax, log_x=False, log_base=2, hide_spines=True, show_xlabel=True):
    if hide_spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    if log_x:
        ax.set_xscale("log", base=log_base)
        if log_base == 2:
            ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=4))
    
    if not show_xlabel:
        ax.set_xticklabels([])
        ax.set_xlabel("")

def save_grid_legend(base_filename, handles, labels, width_pt, scaling=1.0, font_size=8):
    leg_width = (width_pt / 72.27) / scaling
    num = len(labels)
    cols = min(num, 4)
    rows = (num + cols - 1) // cols
    leg_height = rows * (0.25 / scaling)
    
    fig = plt.figure(figsize=(leg_width, leg_height))
    final_handles, final_labels = [], []
    for c in range(cols):
        for r in range(rows):
            idx = r * cols + c
            if idx < num:
                final_handles.append(handles[idx])
                final_labels.append(labels[idx])
            else:
                final_handles.append(plt.Line2D([0], [0], color='none', label=''))
                final_labels.append('')

    fig.legend(final_handles, final_labels, loc='center', ncol=cols, frameon=False, fontsize=font_size/scaling)
    save_fig(fig, f"{base_filename}legend.pdf")

def assign_colors(model_names):
    return {m: TOL_12[i] for i, m in enumerate(model_names)}

def categorize_models(model_names):
    r = sorted([m for m in model_names if "reason" in m])
    u = sorted([m for m in model_names if m not in r])
    return r, u

def categorize_families(all_models_plot_data):
    llama = {x: d for x, d in all_models_plot_data.items() if "Llama" in x}
    qwen = {x: d for x, d in all_models_plot_data.items() if x not in llama}
    return llama, qwen

def calculate_plot_limits(models_data: dict) -> Tuple[Tuple, Tuple]:
    acc, tok = [], []
    for m in models_data:
        acc.extend(models_data[m][1] * 100); acc.extend(models_data[m][3] * 100)
        tok.extend(models_data[m][5])
    
    def get_lim(vals, is_acc=False):
        if not vals: return (0, 100) if is_acc else (0, 1000)
        mn, mx = min(vals), max(vals)
        rng = mx - mn
        pad = 0.05 * rng
        return (max(0, mn - pad), min(100, mx + pad)) if is_acc else (mn - pad, mx + pad)
    return get_lim(acc, True), get_lim(tok, False)

def compute_shared_limits(d1, d2):
    l1, t1 = calculate_plot_limits(d1)
    l2, t2 = calculate_plot_limits(d2)
    return (min(l1[0], l2[0]), max(l1[1], l2[1])), (min(t1[0], t2[0]), max(t1[1], t2[1]))

def _plot_metric_on_ax(ax, models_data, model_names, colors, idx_mean, idx_std, is_pct):
    handles, labels = [], []
    for m in model_names:
        data = models_data[m]
        mean, std = (data[idx_mean] * 100, np.array(data[idx_std]).T * 100) if is_pct else (data[idx_mean], np.array(data[idx_std]).T)
        c = colors[m]
        lbl = clean_label(m)
        l = ax.errorbar(data[0], mean, yerr=std, fmt="-o", capsize=3, color=c, label=lbl, alpha=0.8, markersize=3)
        handles.append(l); labels.append(lbl)
    return handles, labels

def save_separated_plots_overlaid(
    models_data, 
    base_filename, 
    width_pt=486, 
    colors=None, 
    log_scale=True, 
    ylim_accuracy=(0,100), 
    ylim_tokens=(0,1000), 
    samples_disp=True
):
    latexify(font_size=10, small_font_size=8)
    if not models_data: return
    
    names = list(models_data.keys())
    figsize = (width_pt/3.0/72.27, (width_pt/3.0/72.27)*0.9)
    margins = {'left': 0.32, 'right': 0.99, 'top': 0.98, 'bottom': 0.23}

    def _save_component(suffix, idx_m, idx_s, is_pct, ylabel, ylim, log, show_x):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(**margins)
        
        style_axis(ax, log_x=log, show_xlabel=show_x)
        
        h, l = _plot_metric_on_ax(ax, models_data, names, colors, idx_m, idx_s, is_pct)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)
        
        if show_x:
            ax.set_xlabel(r"Test-time compute, $\theta$")
            
        if "Tokens" in ylabel: 
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            
        save_fig(fig, f"{base_filename}{suffix}.pdf")
        return h, l

    h, l = _save_component("majority", 1, 2, True, r"Accuracy (\%)", ylim_accuracy, log_scale, samples_disp)
    _save_component("best_of_n", 3, 4, True, r"Accuracy (\%)", ylim_accuracy, log_scale, samples_disp)
    _save_component("tokens", 5, 6, False, "Tokens", ylim_tokens, log_scale, samples_disp)
    
    save_grid_legend(base_filename, h, l, width_pt)
    print(f"Saved files to {base_filename}_*.pdf")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    params = [(1,2,True,ylim_accuracy,"Majority Voting"), (3,4,True,ylim_accuracy,"Best-of-N"), (5,6,False,ylim_tokens,"Avg Tokens")]
    for ax, p in zip(axes, params):
        style_axis(ax, log_x=log_scale, show_xlabel=True)
        ax.set_xlabel(r"Test-time compute, $\theta$")
        _plot_metric_on_ax(ax, models_data, names, colors, p[0], p[1], p[2])
        ax.set_title(p[4]); ax.set_ylim(p[3])
    save_sanity_check(fig, base_filename)

def _plot_reasoning_lines(ax, models_data, colors, idx_m, idx_e, is_acc):
    handles, labels = [], []
    def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (m, data) in enumerate(models_data.items()):
        mean = np.array(data[idx_m], float) * (100 if is_acc else 1)
        err = np.array(data[idx_e]); 
        if err.ndim == 2 and err.shape == (mean.size, 2): err = err.T
        c = colors.get(m, def_cols[i % len(def_cols)]) if colors else def_cols[i % len(def_cols)]
        lbl = clean_label(m)
        l = ax.errorbar(np.arange(1, mean.size+1), mean, yerr=err, fmt="-o", linewidth=1.2, capsize=3, color=c, label=lbl, alpha=0.9, markersize=3)
        handles.append(l); labels.append(lbl)
    return handles, labels

def save_reasoning_plots_separated(models_data, base_filename, width_pt=486, colors=None):
    if not models_data: return
    
    w_per = width_pt / 2.2
    figsize = (w_per/72.27, (w_per/72.27)*0.85)
    margins = {'left': 0.28, 'right': 0.98, 'top': 0.98, 'bottom': 0.22}
    bins = np.arange(1, len(list(models_data.values())[0][0]) + 1)

    def _do_plot(idx_m, idx_e, is_acc, ylab, out_name):
        fig, ax = plt.subplots(figsize=figsize)
        fig.subplots_adjust(**margins)
        style_axis(ax); ax.set_xticks(bins)
        h, l = _plot_reasoning_lines(ax, models_data, colors, idx_m, idx_e, is_acc)
        ax.set_ylabel(ylab); ax.set_xlabel(r"Test-time compute, $\theta$")
        save_fig(fig, out_name)
        return h, l

    h, l = _do_plot(0, 1, True, r"Accuracy (\%)", f"{base_filename}accuracy.pdf")
    _do_plot(2, 3, False, "Tokens", f"{base_filename}tokens.pdf")
    save_grid_legend(base_filename, h, l, width_pt, scaling=0.8)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, p in zip(axes, [(0,1,True,r"Accuracy (\%)"), (2,3,False,"Tokens")]):
        style_axis(ax); ax.set_xticks(bins)
        _plot_reasoning_lines(ax, models_data, colors, p[0], p[1], p[2])
        ax.set_title(p[3] if "Acc" in p[3] else "Tokens")
    save_sanity_check(fig, base_filename)

def plot_v_curves(V_curves, thetas, config, base_filename, width_pt=ICML_WIDTH_TEXT_PT/3.0, colors=None, reasoning=False):
    latexify(font_size=9, small_font_size=6)
    figsize = get_fig_dim(width_pt, 1.0, 1.6)
    SCALE = 1000.0
    
    def _draw(ax_target):
        style_axis(ax_target, log_x=(not reasoning))
        if not reasoning:
            ax_target.set_xticks(thetas)
            ax_target.set_xticklabels([f"$2^{{{int(np.log2(t))}}}$" for t in thetas], fontsize=6)
        else:
            ax_target.set_xticks(thetas); ax_target.set_xticklabels([str(t) for t in thetas], fontsize=6)
        ax_target.axhline(0, color='black', lw=0.8, ls='--', alpha=0.5)
        
        handles, labels, vals = [], [], []
        def_cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (name, v) in enumerate(V_curves.items()):
            c = colors.get(name, def_cols[i % len(def_cols)]) if colors else def_cols[i % len(def_cols)]
            lbl = clean_label(name)
            v_scaled = v * SCALE
            vals.extend(v_scaled)
            l, = ax_target.plot(thetas, v_scaled, marker="o", lw=1.2, ms=3, label=lbl, color=c, alpha=0.9)
            handles.append(l); labels.append(lbl)
        return handles, labels, vals

    fig, ax = plt.subplots(figsize=figsize)
    h, l, v_vals = _draw(ax)
    if v_vals:
        rng = max(v_vals) - min(v_vals)
        ax.set_ylim(min(v_vals) - 0.1*rng, max(v_vals) + 0.1*rng)
    ax.set_xlabel(r"Test-time compute, $\theta$", fontsize=7)
    ax.set_ylabel(r"Value ($\$ \times 10^{-3}$)", fontsize=7)
    ax.tick_params(labelsize=6)
    
    save_fig(fig, f"{base_filename}v_curves.pdf")
    save_grid_legend(base_filename, h, l, 486)
    print(f"Saved: {base_filename}v_curves.pdf")
    
    fig_s, ax_s = plt.subplots(figsize=(4, 2.5), dpi=120)
    _draw(ax_s)
    save_sanity_check(fig_s, base_filename)


def _plot_strategies_component(ax, history, steps, color_map, offsets, thetas, reasoning):
    for p_name, traj in history.items():
        data = np.array(traj[:len(steps)], float) + offsets[p_name]
        ax.plot(steps, data, drawstyle='steps-post', lw=1.2, color=color_map[p_name], alpha=0.9)
    if not reasoning:
        ax.set_yticks(range(len(thetas)))
        ax.set_yticklabels([f"$2^{{{int(np.log2(t))}}}$" for t in thetas])
        ax.set_ylim(-0.4, len(thetas) - 0.6)
    else:
        ax.set_yticks(range(len(thetas))); ax.set_yticklabels([str(t) for t in thetas])

def _plot_ineff_twin(ax_main, steps, poa_hist, ylim):
    ax_ineff = ax_main.twinx()
    ax_ineff.plot(steps, (np.array(poa_hist)-1.0)*100, 'k--', lw=1.2, alpha=0.6)
    ax_ineff.set_ylim(ylim)
    ax_ineff.spines['top'].set_visible(False)
    return ax_ineff

def _plot_shares_component(ax, share_hist, steps, providers, color_map):
    bottom = np.zeros(len(steps))
    for i, p in enumerate(providers):
        shares = np.array(share_hist)[:, i]
        ax.bar(steps, shares, bottom=bottom, color=color_map[p.name], width=1.0, edgecolor='none', label=p.name, alpha=0.9)
        bottom += shares
    ax.set_ylim(0, 1.0)
    ax.set_xlim(-0.5, len(steps) - 0.5)

def _get_common_dyn_args(providers, betas_to_test, results, colors):
    latexify(font_size=10, small_font_size=8) if colors is None else latexify(font_size=9, small_font_size=6)
    c_arr = plt.cm.tab20(np.linspace(0, 1, len(providers)))
    c_map = colors if colors else {p.name: c_arr[i % len(c_arr)] for i, p in enumerate(providers)}
    offs = {p.name: v for p, v in zip(providers, np.linspace(-0.25, 0.25, len(providers)))}
    
    all_ineff = [v for b in betas_to_test for v in (np.array(results[b][1]) - 1.0) * 100]
    mx = max(all_ineff) if all_ineff else 0
    ylim_ineff = (0, max(5.0, mx * 1.1))
    return c_map, offs, ylim_ineff

def _save_common_legend(base_filename, providers, c_map):
    handles = [plt.Line2D([0], [0], color=c_map[p.name], lw=2, label=clean_label(p.name)) for p in providers]
    handles.append(plt.Line2D([0], [0], color='black', ls='--', lw=1.5, label=r"Inefficiency (\%)"))
    save_grid_legend(f"{base_filename}/", handles, [h.get_label() for h in handles], ICML_WIDTH_TEXT_PT)

def bet_pot_analysis(providers, thetas, betas_to_test, results, base_filename=None, colors=None, reasoning=False):
    c_map, offsets, ylim_ineff = _get_common_dyn_args(providers, betas_to_test, results, colors)
    figsize = get_fig_dim(ICML_WIDTH_TEXT_PT/3.0, 1.0, 0.75)

    for beta in betas_to_test:
        hist, poa, pot, _, _ = results[beta]
        steps = range(len(poa))
        b_str = f"beta{str(beta).replace('.', 'p')}"
        
        fig, ax = plt.subplots(figsize=figsize); style_axis(ax)
        _plot_strategies_component(ax, hist, steps, c_map, offsets, thetas, reasoning)
        ax.set_ylabel(r"Test-time compute, $\theta$"); ax.set_xlabel("Iteration, $t$")
        ax2 = _plot_ineff_twin(ax, steps, poa, ylim_ineff)
        ax2.set_ylabel(r"Inefficiency (\%)", color='black')
        save_fig(fig, f"{base_filename}/{b_str}/strategies_combined.pdf")

        fig, ax = plt.subplots(figsize=figsize); style_axis(ax)
        ax.plot(steps, pot, color='purple', lw=1.5)
        ax.set_ylabel(r"Potential, $\Phi$"); ax.set_xlabel("Iteration, $t$")
        save_fig(fig, f"{base_filename}/{b_str}/potential.pdf")

    _save_common_legend(base_filename, providers, c_map)

def plot_market_shares(providers, betas_to_test, results, base_filename=None, colors=None):
    c_map, _, _ = _get_common_dyn_args(providers, betas_to_test, results, colors)
    figsize = get_fig_dim(ICML_WIDTH_TEXT_PT/3.0, 1.0, 0.75)
    
    for beta in betas_to_test:
        share_hist = results[beta][3]
        steps = range(len(share_hist))
        fig, ax = plt.subplots(figsize=figsize); style_axis(ax)
        _plot_shares_component(ax, share_hist, steps, providers, c_map)
        ax.set_ylabel("Market share"); ax.set_xlabel("Iteration, $t$")
        save_fig(fig, f"{base_filename}/beta{str(beta).replace('.', 'p')}/shares.pdf")

def plot_combined_dynamics(providers, thetas, betas_to_test, results, base_filename=None, colors=None, reasoning=False):
    c_map, offsets, ylim_ineff = _get_common_dyn_args(providers, betas_to_test, results, colors)
    latexify(font_size=9, small_font_size=6)
    
    figsize_stacked = get_fig_dim(ICML_WIDTH_TEXT_PT/3.0, 1.0, 0.9)
    figsize_pot = get_fig_dim(ICML_WIDTH_TEXT_PT/3.0, 1.0, 0.75)

    for beta in betas_to_test:
        hist, poa, pot, shares, _ = results[beta]
        steps = range(len(poa))
        b_str = f"beta{str(beta).replace('.', 'p')}"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize_stacked, sharex=True, gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.08})
        style_axis(ax1); style_axis(ax2, hide_spines=False)
        ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
        
        _plot_strategies_component(ax1, hist, steps, c_map, offsets, thetas, reasoning)
        ax1.set_ylabel(r"Test-time compute, $\theta$", fontsize=7)
        ax_tw = _plot_ineff_twin(ax1, steps, poa, ylim_ineff)
        ax_tw.set_ylabel(r"Inefficiency (\%)", color='black', fontsize=7)

        _plot_shares_component(ax2, shares, steps, providers, c_map)
        ax2.set_ylabel("Market share", fontsize=7); ax2.set_xlabel(r"Iteration, $t$", fontsize=7)
        ax2.set_yticks([0, 0.5, 1.0]); ax2.set_yticklabels(["0", ".5", "1"], fontsize=6)
        
        save_fig(fig, f"{base_filename}/{b_str}/combined.pdf")

        fig, ax = plt.subplots(figsize=figsize_pot); style_axis(ax)
        ax.plot(steps, pot, color='purple', lw=1.5)
        ax.set_ylabel(r"Potential, $\Phi$"); ax.set_xlabel(r"Iteration, $t$")
        save_fig(fig, f"{base_filename}/{b_str}/potential.pdf")

    _save_common_legend(base_filename, providers, c_map)
    _sanity_check_combined(results, betas_to_test, providers, thetas, c_map, offsets, ylim_ineff, reasoning, base_filename)

def _sanity_check_combined(results, betas, providers, thetas, c_map, offsets, ylim_ineff, reasoning, base_filename):
    n = len(betas)
    fig, axes = plt.subplots(n, 3, figsize=(18, 4*n), constrained_layout=True)
    if n == 1: axes = np.array([axes])
    
    for i, beta in enumerate(betas):
        hist, poa, pot, shares, _ = results[beta]
        steps = range(len(poa))
        
        ax = axes[i, 0]; style_axis(ax); ax.set_title(f"Beta={beta}: Strategies")
        _plot_strategies_component(ax, hist, steps, c_map, offsets, thetas, reasoning)
        _plot_ineff_twin(ax, steps, poa, ylim_ineff)
        
        ax = axes[i, 1]; style_axis(ax); ax.set_title("Shares")
        _plot_shares_component(ax, shares, steps, providers, c_map)
        
        ax = axes[i, 2]; style_axis(ax); ax.set_title("Potential")
        ax.plot(steps, pot, color='purple'); ax.set_ylabel(r"$\Phi$")

    save_sanity_check(fig, base_filename)

def plot_beta_sweep(beta_values, final_poas, base_filename="beta_sweep"):
    latexify(font_size=10, small_font_size=8)
    fig, ax = plt.subplots(figsize=get_fig_dim(ICML_WIDTH_TEXT_PT, 1.0))
    style_axis(ax)
    ineff = (np.array(final_poas) - 1.0) * 100
    ax.plot(beta_values, ineff, 'k-', lw=1.5)
    ax.set_xscale('log'); ax.set_xlabel(r"User rationality, $\beta$"); ax.set_ylabel(r"Inefficiency (\%)")
    if len(ineff) > 0: ax.set_ylim(0, max(5.0, max(ineff) * 1.1))
    
    save_fig(fig, f"{base_filename}inefficiency.pdf", close=False)
    save_sanity_check(fig, base_filename, suffix="inefficiency")