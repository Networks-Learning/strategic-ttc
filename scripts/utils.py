import numpy as np
import os
from typing import Any, List, Optional, Tuple, Dict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from IPython.display import display, Image

TOL_12 = [
    "#004183",  # blue
    "#4192E3",  # light blue,
    "#84CDE6",  # cyan,
    "#16A990",  # teal
    
    "#786B06",  # yellow
    "#E1C84C",  # sand
    "#F7949F",  # red
    "#AA3377",  # purple
    "#56002B",  # wine
    
    "#C5C3C3",  # grey
    "#6E6E6E",  # green
    "#000000",  # black
    
]

def get_fig_dim(width, fraction=1, aspect_ratio=None):

    fig_width_pt = width * fraction

    inches_per_pt = 1 / 72.27

    if aspect_ratio is None:
        aspect_ratio = (1 + 5**.5) / 2

    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in / aspect_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def latexify(font_serif='Computer Modern', mathtext_font='cm', font_size=10, small_font_size=None, usetex=True):
    if small_font_size is None:
        small_font_size = font_size

    params = {
        'backend': 'ps',
        'text.latex.preamble': '\\usepackage{gensymb} \\usepackage{bm}',
            
        'axes.labelsize': font_size,
        'axes.titlesize': font_size,
        'font.size': font_size,
        
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

def assign_colors(model_names):
    idx = 0
    model_color = {}
    for model in model_names:
        model_color[model] = TOL_12[idx]
        idx += 1
    return model_color

def categorize_models(model_names):
    reasoning_models = sorted([model for model in model_names if "reason" in model])
    unreasoning_models = sorted([model for model in model_names if model not in reasoning_models])
    return reasoning_models, unreasoning_models

def categorize_families(all_models_plot_data):
    models_data_llama = {x: all_models_plot_data[x] for x in all_models_plot_data.keys() if "Llama" in x}
    models_data_qwen = {x: all_models_plot_data[x] for x in all_models_plot_data.keys() if x not in models_data_llama.keys()}
    return models_data_llama, models_data_qwen


def calculate_plot_limits(
    models_data: dict,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    
    model_names = list(models_data.keys())
    
    all_acc_values = []
    for m in model_names:
        all_acc_values.extend(models_data[m][1] * 100) 
        all_acc_values.extend(models_data[m][3] * 100)
        
    if all_acc_values:
        y_min, y_max = min(all_acc_values), max(all_acc_values)
        y_range = y_max - y_min
        ylim_accuracy = (max(0, y_min - 0.05 * y_range), min(100, y_max + 0.05 * y_range))

    all_tok_values = []
    for m in model_names:
        all_tok_values.extend(models_data[m][5])
        
    if all_tok_values:
        y_min, y_max = min(all_tok_values), max(all_tok_values)
        y_range = y_max - y_min
        ylim_tokens = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)


    return ylim_accuracy, ylim_tokens

def compute_shared_limits(models_data_llama, models_data_qwen):

    ylim_accuracy1, ylim_tokens1 = calculate_plot_limits(
        models_data=models_data_llama,
    )


    ylim_accuracy2, ylim_tokens2 = calculate_plot_limits(
        models_data=models_data_qwen,
    )

    return (min(ylim_accuracy1[0], ylim_accuracy2[0]), max(ylim_accuracy1[1], ylim_accuracy2[1])), \
           (min(ylim_tokens1[0], ylim_tokens2[0]), max(ylim_tokens1[1], ylim_tokens2[1]))


def setup_plot_dimensions(width_pt: float) -> Tuple[Tuple[float, float], Dict[str, float]]:
    width_per_plot_pt = width_pt / 3.0
    fig_width_in = width_per_plot_pt / 72.27

    fig_height_in = fig_width_in * 0.9 
    figsize = (fig_width_in, fig_height_in)
    
    margins = {
        'left': 0.32,   
        'right': 0.99,
        'top': 0.98,
        'bottom': 0.23
    }
    return figsize, margins

def apply_axis_style(ax: plt.Axes, log_scale: bool, samples_disp: bool) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    if log_scale:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(ticker.LogLocator(base=2.0, subs=[1.0], numticks=4))

    if samples_disp:
        ax.set_xlabel("Test-time compute ($\\theta$)")
    else:
        ax.set_xticklabels([])

def plot_single_metric(
    ax: plt.Axes,
    models_data: dict,
    model_names: List[str],
    colors: Dict[str, str],
    metric_idx_mean: int,
    metric_idx_std: int,
    is_percentage: bool,
) -> Tuple[List, List]:
    handles = []
    labels = []

    for i, model_name in enumerate(model_names):
        data_tuple = models_data[model_name]
        thetas = data_tuple[0]
        mean_val = data_tuple[metric_idx_mean]
        std_val = data_tuple[metric_idx_std]

        if is_percentage:
            y_mean = mean_val * 100
            y_err = np.asarray(std_val).T * 100
        else:
            y_mean = mean_val
            y_err = np.asarray(std_val).T

        if colors:
            c = colors.get(model_name)
        else:
            raise ValueError("Colors dictionary must be provided")

        clean_name = model_name.replace("_", r"-")
        label_tex = f"\\texttt{{{clean_name}}}"

        line = ax.errorbar(
            thetas, y_mean, yerr=y_err,
            fmt="-o", capsize=3, color=c, label=label_tex,
            alpha=0.8, markersize=3
        )
        handles.append(line)
        labels.append(label_tex)
        
    return handles, labels

def save_standalone_legend(
    base_filename: str,
    handles: List,
    labels: List,
    width_pt: float
) -> None:

    leg_width_inch = width_pt / 72.27
    
    num_models = len(labels)
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
                reordered_handles.append(handles[idx])
                reordered_labels.append(labels[idx])

    fig_leg.legend(
        reordered_handles, reordered_labels,
        loc='center',
        ncol=cols, 
        frameon=False,
        fontsize=8 
    )
    
    out_leg = f"{base_filename}legend.pdf"
    fig_leg.savefig(out_leg, pad_inches=0)
    plt.close(fig_leg)

def plot_accuracy_one_figure_unreasoning(
    base_filename: str,
    models_data: dict,
    model_names: List[str],
    colors: Dict[str, str],
    log_scale: bool,
    ylim_accuracy: Tuple[float, float],
    ylim_tokens: Tuple[float, float]
):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    
    apply_axis_style(axes[0], log_scale, samples_disp=True)
    plot_single_metric(axes[0], models_data, model_names, colors, 1, 2, True)
    axes[0].set_title("Majority Voting")
    axes[0].set_ylabel("Accuracy (\\%)")
    axes[0].set_ylim(ylim_accuracy)

    apply_axis_style(axes[1], log_scale, samples_disp=True)
    plot_single_metric(axes[1], models_data, model_names, colors, 3, 4, True)
    axes[1].set_title("Best-of-N")
    axes[1].set_ylabel("Accuracy (\\%)")
    axes[1].set_ylim(ylim_accuracy)

    apply_axis_style(axes[2], log_scale, samples_disp=True)
    handles, labels = plot_single_metric(axes[2], models_data, model_names, colors, 5, 6, False)
    axes[2].set_title("Avg Tokens")
    axes[2].set_ylabel("Tokens")
    axes[2].set_ylim(ylim_tokens)

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    sanity_filename = f"{base_filename}check_one.png"
    fig.savefig(sanity_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    display(Image(filename=sanity_filename))

def save_separated_plots_overlaid(
    models_data: dict,
    base_filename: str,
    width_pt: float = 486, 
    colors: Dict[str, str] = None,
    log_scale: bool = True,
    ylim_accuracy: Tuple[float, float] = (0, 100),
    ylim_tokens: Tuple[float, float] = (0, 1000),
    samples_disp: bool = True,
):
    latexify(font_size=10, small_font_size=8)
    model_names = list(models_data.keys())
    if not model_names:
        print("No models to plot.")
        return

    figsize, margins = setup_plot_dimensions(width_pt)
    fig_maj, ax_maj = plt.subplots(figsize=figsize)
    fig_maj.subplots_adjust(**margins)
    apply_axis_style(ax_maj, log_scale, samples_disp)
    
    handles, labels = plot_single_metric(
        ax_maj, models_data, model_names, colors, 
        metric_idx_mean=1, metric_idx_std=2, is_percentage=True, 
    )
    
    ax_maj.set_ylabel(r"Accuracy (\\%)")
    ax_maj.set_ylim(ylim_accuracy)

    out_maj = f"{base_filename}majority.pdf"

    os.makedirs(os.path.dirname(out_maj), exist_ok=True)
    fig_maj.savefig(out_maj, pad_inches=0)
    plt.close(fig_maj)

    fig_bon, ax_bon = plt.subplots(figsize=figsize)
    fig_bon.subplots_adjust(**margins) 
    apply_axis_style(ax_bon, log_scale, samples_disp)

    plot_single_metric(
        ax_bon, models_data, model_names, colors, 
        metric_idx_mean=3, metric_idx_std=4, is_percentage=True, 
    )

    ax_bon.set_ylabel(r"Accuracy (\\%)")
    ax_bon.set_ylim(ylim_accuracy)
    
    out_bon = f"{base_filename}best_of_n.pdf"
    fig_bon.savefig(out_bon, pad_inches=0)
    plt.close(fig_bon)

    fig_tok, ax_tok = plt.subplots(figsize=figsize)
    fig_tok.subplots_adjust(**margins)
    apply_axis_style(ax_tok, log_scale, samples_disp)

    plot_single_metric(
        ax_tok, models_data, model_names, colors, 
        metric_idx_mean=5, metric_idx_std=6, is_percentage=False, 
    )

    ax_tok.set_ylabel("Tokens")
    ax_tok.set_ylim(ylim_tokens)
    ax_tok.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
    
    out_tok = f"{base_filename}tokens.pdf"
    fig_tok.savefig(out_tok, pad_inches=0)
    plt.close(fig_tok)

    save_standalone_legend(base_filename, handles, labels, width_pt)
    
    print(f"Saved 4 files to {base_filename}_*.pdf")

    plot_accuracy_one_figure_unreasoning(
        base_filename, models_data, model_names, colors, 
        log_scale, ylim_accuracy, ylim_tokens
    )

def setup_reasoning_layout(width_pt: float) -> Tuple[Tuple[float, float], Dict[str, float]]:
    """
    Calculates figure dimensions and margins specific to reasoning plots.
    Note: Uses a narrower width divisor (2.2) and aspect ratio (0.85) than standard plots.
    """
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
    return figsize, margins

def format_reasoning_label(model_name: str) -> str:
    clean_name = model_name.replace("reason-", "").replace("Distill-", "D-")
    clean_name = clean_name.replace("_", r"\_")
    return f"\\texttt{{{clean_name}}}"

def fix_yerr_shape(errs: Any, n_points: int) -> np.ndarray:
    yerr = np.array(errs)
    if yerr.ndim == 2 and yerr.shape == (n_points, 2):
        return yerr.T
    return yerr

def style_reasoning_ax(ax: plt.Axes, bins: np.ndarray) -> None:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.set_xticks(bins) 

def plot_reasoning_lines(
    ax: plt.Axes,
    models_data: dict,
    colors: Dict[str, str],
    metric_idx_mean: int,
    metric_idx_err: int,
    is_accuracy: bool,
    default_colors: List[str] = None
) -> Tuple[List, List]:
    if default_colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    handles = []
    labels = []

    for i, (model_name, data) in enumerate(models_data.items()):
        means = data[metric_idx_mean]
        errs = data[metric_idx_err]
        
        means = np.asarray(means, dtype=float)
        if is_accuracy:
            means = means * 100
            
        n = means.size
        bins = np.arange(1, n + 1)
        yerr = fix_yerr_shape(errs, n)

        if colors:
            c = colors.get(model_name, default_colors[i % len(default_colors)])
        else:
            c = default_colors[i % len(default_colors)]

        label_tex = format_reasoning_label(model_name)

        line = ax.errorbar(
            bins, means, yerr=yerr,
            fmt="-o", linewidth=1.2, capsize=3,
            color=c, label=label_tex, alpha=0.9, markersize=3
        )
        
        handles.append(line)
        labels.append(label_tex)
        
    return handles, labels

def save_reasoning_legend(
    base_filename: str,
    handles: List,
    labels: List,
    width_pt: float
) -> None:

    target_font_size = 8
    latex_scale_factor = 0.8  
    adjusted_fontsize = target_font_size / latex_scale_factor 
    
    leg_width_inch = (width_pt / 72.27) / latex_scale_factor
    
    num_models = len(labels)
    cols = min(num_models, 3) 
    rows = (num_models + cols - 1) // cols
    
    height_per_row_inch = 0.25 / latex_scale_factor
    leg_height_inch = rows * height_per_row_inch
    
    fig_leg = plt.figure(figsize=(leg_width_inch, leg_height_inch)) 
    
    fig_leg.legend(
        handles, 
        labels, 
        loc='center',
        ncol=cols, 
        frameon=False,
        fontsize=adjusted_fontsize 
    )
    
    out_leg = f"{base_filename}legend.pdf"
    fig_leg.savefig(out_leg, pad_inches=0)
    plt.close(fig_leg)

def plot_accuracy_one_figure_reasoning(
    models_data: dict,
    colors: Dict[str, str],
    base_filename: str
):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    
    first_key = list(models_data.keys())[0]
    n_bins_global = len(models_data[first_key][0])
    global_bins = np.arange(1, n_bins_global + 1)

    style_reasoning_ax(axes[0], global_bins)
    handles, labels = plot_reasoning_lines(
        axes[0], models_data, colors, 
        metric_idx_mean=0, metric_idx_err=1, is_accuracy=True
    )
    axes[0].set_title("Accuracy")
    axes[0].set_ylabel("Accuracy (\\%)")
    axes[0].set_xlabel(r"Test-time compute ($\theta$)")

    style_reasoning_ax(axes[1], global_bins)
    plot_reasoning_lines(
        axes[1], models_data, colors, 
        metric_idx_mean=2, metric_idx_err=3, is_accuracy=False
    )
    axes[1].set_title("Tokens")
    axes[1].set_ylabel("Tokens")
    axes[1].set_xlabel(r"Test-time compute ($\theta$)")

    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(1.15, 0.5))
    
    sanity_filename = f"{base_filename}check_one.png"
    fig.savefig(sanity_filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    display(Image(filename=sanity_filename))

def save_reasoning_plots_separated(
    models_data: dict,
    base_filename: str,
    width_pt: float = 486, 
    colors: Dict[str, str] = None,
):
    if not models_data:
        print("No data to plot.")
        return

    figsize, margins = setup_reasoning_layout(width_pt)
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    first_key = list(models_data.keys())[0]
    n_bins_global = len(models_data[first_key][0])
    global_bins = np.arange(1, n_bins_global + 1)

    fig_acc, ax_acc = plt.subplots(figsize=figsize)
    fig_acc.subplots_adjust(**margins)
    style_reasoning_ax(ax_acc, global_bins)

    handles, labels = plot_reasoning_lines(
        ax_acc, models_data, colors, 
        metric_idx_mean=0, metric_idx_err=1, is_accuracy=True,
        default_colors=default_colors
    )

    ax_acc.set_ylabel(r"Accuracy (\\%)")
    ax_acc.set_xlabel(r"Test-time compute ($\theta$)")

    out_acc = f"{base_filename}accuracy.pdf"
    os.makedirs(os.path.dirname(out_acc), exist_ok=True)
    fig_acc.savefig(out_acc, pad_inches=0)
    plt.close(fig_acc)

    fig_tok, ax_tok = plt.subplots(figsize=figsize)
    fig_tok.subplots_adjust(**margins)
    style_reasoning_ax(ax_tok, global_bins)

    plot_reasoning_lines(
        ax_tok, models_data, colors, 
        metric_idx_mean=2, metric_idx_err=3, is_accuracy=False,
        default_colors=default_colors
    )

    ax_tok.set_ylabel("Tokens")
    ax_tok.set_xlabel(r"Test-time compute ($\theta$)")

    out_tok = f"{base_filename}tokens.pdf"
    fig_tok.savefig(out_tok, pad_inches=0)
    plt.close(fig_tok)

    save_reasoning_legend(base_filename, handles, labels, width_pt)

    print(f"Saved separate files: {out_acc}, {out_tok}, {base_filename}legend.pdf")

    plot_accuracy_one_figure_reasoning(models_data, colors, base_filename)

def style_ax(ax: plt.Axes, thetas: List[int]):
    """Applies clean spine styling and log-scale x-axis with 2^k labels."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    
    # Log scale base 2
    ax.set_xscale("log", base=2)
    ax.set_xticks(thetas)
    
    # --- CHANGED: Format labels as 2^k ---
    # np.log2(t) gives the exponent. We cast to int.
    power_labels = [f"$2^{{{int(np.log2(t))}}}$" for t in thetas]
    ax.set_xticklabels(power_labels)
    
    # Horizontal line at 0 for reference
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

def plot_v_curves(
    V_curves: Dict[str, np.ndarray],
    thetas: List[int],
    config: object,
    base_filename: str,
    width_pt: float = 486,
    colors: Dict[str, str] = None,
):
    latexify(font_size=10, small_font_size=8)
    
    figsize = get_fig_dim(width_pt, fraction=1.0, aspect_ratio=0.75)
    
    fig, ax = plt.subplots(figsize=figsize)
    style_ax(ax, thetas)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    handles, labels = [], []

    for i, (name, v) in enumerate(V_curves.items()):
        if colors and name in colors:
            c = colors[name]
        else:
            c = default_colors[i % len(default_colors)]
            
        clean_name = name.replace("reason-", "").replace("Distill-", "D-").replace("_", r"\_")
        label_tex = f"\\texttt{{{clean_name}}}"
        
        line, = ax.plot(
            thetas, v, 
            marker="o", 
            linewidth=1.4, 
            markersize=4,
            label=label_tex, 
            color=c,
            alpha=0.9
        )
        
        handles.append(line)
        labels.append(label_tex)

    ax.set_xlabel(r"Test-time compute $\theta$")
    ax.set_ylabel(r"Value $V(\theta)$")
    

    out_main = f"{base_filename}v_curves.pdf"
    os.makedirs(os.path.dirname(out_main), exist_ok=True)
    fig.savefig(out_main, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    
    save_standalone_legend(base_filename, handles, labels, width_pt)
    
    print(f"Saved: {out_main}")
    print(f"Saved: {base_filename}_legend.pdf")

    _sanity_check_v_curves(V_curves, thetas, colors, handles, labels, base_filename)

def _sanity_check_v_curves(V_curves, thetas, colors, handles, labels, base_filename):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_ax(ax, thetas)
    
    for i, (name, v) in enumerate(V_curves.items()):
        c = colors.get(name, 'black') if colors else 'black'
        ax.plot(thetas, v, marker="o", linewidth=1.4, color=c, label=name)

    ax.set_xlabel("Test-time compute ($\\theta$)")
    ax.set_ylabel("V")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    sanity_file = f"{base_filename}check_one.png"
    fig.savefig(sanity_file, dpi=100, bbox_inches='tight')
    plt.close(fig)
    display(Image(filename=sanity_file))

ICML_WIDTH_TEXT_PT = 486

def bet_pot_analysis(providers, thetas, betas_to_test, results, base_filename=None, colors=None, reasoning=False):
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
            
        ax_strat.set_ylabel("Test-time compute ($\\theta$)")
        ax_strat.set_xlabel("Iteration, $t$")
        
        if reasoning == False:
            ax_strat.set_yticks(range(len(thetas)))
            power_labels = [f"$2^{{{int(np.log2(t))}}}$" for t in thetas]
            ax_strat.set_yticklabels(power_labels, fontsize=8)
            ax_strat.set_ylim(-0.4, len(thetas) - 0.6)
        else:
            ax_strat.set_yticks(range(len(thetas)))
            ax_strat.set_yticklabels([f"{t}" for t in thetas], fontsize=8)


        ax_ineff = ax_strat.twinx()
        
        ineff_data = (np.array(poa_hist) - 1.0) * 100
        
        ax_ineff.plot(steps, ineff_data, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        
        ax_ineff.set_ylabel(r"Market Inefficiency (\\%)", color='black', fontsize=9)
        ax_ineff.tick_params(axis='y', labelcolor='black', labelsize=8)
        
        ax_ineff.set_ylim(ineff_ylim)
        ax_ineff.spines['top'].set_visible(False) 
        out_strat = f"{base_filename}/beta{beta_str}/strategies_combined.pdf"
        os.makedirs(os.path.dirname(out_strat), exist_ok=True)
        fig_strat.savefig(out_strat, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig_strat)


        fig_pot, ax_pot = plt.subplots(figsize=figsize)
        style_ax(ax_pot)
        
        ax_pot.plot(steps, pot_hist, color='purple', linewidth=1.5)
        
        ax_pot.set_ylabel(r"Potential $\Phi$")
        ax_pot.set_xlabel("Iteration, $t$")

        out_pot = f"{base_filename}/beta{beta_str}/potential.pdf"
        os.makedirs(os.path.dirname(out_pot), exist_ok=True)
        fig_pot.savefig(out_pot, bbox_inches='tight', pad_inches=0.02)
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

    out_leg = f"{base_filename}/legend.pdf"
    os.makedirs(os.path.dirname(out_leg), exist_ok=True)
    fig_leg.savefig(out_leg, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_leg)

    # _sanity_check_all(
    #     results, betas_to_test, providers, thetas, color_map, provider_offsets, ineff_ylim, reasoning, base_filename
    # )

def plot_market_shares(providers, betas_to_test, results, base_filename=None, colors=None):

    latexify(font_size=10, small_font_size=8)
    
    width_per_plot = ICML_WIDTH_TEXT_PT / 3.0
    figsize = get_fig_dim(width_per_plot, fraction=1.0, aspect_ratio=0.75)
    
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors_arr = plt.cm.tab20(np.linspace(0, 1, len(providers)))
    color_map = colors if colors is not None else {
        p.name: colors_arr[i % len(colors_arr)] for i, p in enumerate(providers)
    } 

    def style_ax(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        ax.tick_params(axis='both', which='major', labelsize=8)

    for beta in betas_to_test:
        history, poa_hist, pot_hist, share_hist = results[beta]
        

        share_data = np.array(share_hist)
        total_steps = len(share_data)
        steps = range(total_steps)
        
        beta_str = str(beta).replace(".", "p")


        fig, ax = plt.subplots(figsize=figsize)
        style_ax(ax)
        
        bottom = np.zeros(total_steps)
        
        for i, p in enumerate(providers):
            provider_shares = share_data[:, i]
            
            ax.bar(
                steps, 
                provider_shares, 
                bottom=bottom, 
                color=color_map[p.name], 
                width=1.0, 
                edgecolor='white', 
                linewidth=0.3,
                label=p.name,
                alpha=0.9
            )
            
            bottom += provider_shares
            
        ax.set_ylabel("Market Share")
        ax.set_xlabel("Iteration")
        ax.set_ylim(0, 1.0) 
        ax.set_xlim(-0.5, total_steps - 0.5)

        out_file = f"{base_filename}/beta{beta_str}/shares.pdf"
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

def _sanity_check_combined(results, betas_to_test, providers, thetas, color_map, provider_offsets, ineff_ylim, reasoning, base_filename):
    
    num_betas = len(betas_to_test)
    fig, axes = plt.subplots(nrows=num_betas, ncols=3, figsize=(18, 4 * num_betas), constrained_layout=True)
    
    if num_betas == 1:
        axes = np.array([axes])

    for i, beta in enumerate(betas_to_test):
        history, poa_hist, pot_hist, share_hist = results[beta]
        total_steps = len(poa_hist)
        steps = range(total_steps)
        
        ax_strat = axes[i, 0]
        ax_strat.set_title(f"Beta={beta}: Strategies")
        
        for provider_name, trajectory in history.items():
            traj_data = np.array(trajectory[:total_steps], dtype=float)
            traj_data += provider_offsets[provider_name]
            
            ax_strat.plot(
                steps, 
                traj_data, 
                drawstyle='steps-post', 
                linewidth=1.2,
                color=color_map[provider_name],
                alpha=0.9
            )
        
        ax_strat.set_ylabel(r"Compute $\theta$")
        ax_strat.set_xlabel("Iteration")
        
        if not reasoning:
            ax_strat.set_yticks(range(len(thetas)))
            power_labels = [f"$2^{{{int(np.log2(t))}}}$" for t in thetas]
            ax_strat.set_yticklabels(power_labels, fontsize=8)
            ax_strat.set_ylim(-0.4, len(thetas) - 0.6)
        else:
            ax_strat.set_yticks(range(len(thetas)))
            ax_strat.set_yticklabels([f"{t}" for t in thetas], fontsize=8)

        ax_ineff = ax_strat.twinx()
        ineff_data = (np.array(poa_hist) - 1.0) * 100
        ax_ineff.plot(steps, ineff_data, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        ax_ineff.set_ylabel("Ineff (%)", color='black')
        ax_ineff.set_ylim(ineff_ylim)

        ax_share = axes[i, 1]
        ax_share.set_title(f"Beta={beta}: Shares")
        
        share_data = np.array(share_hist)
        bottom = np.zeros(total_steps)
        
        for p in providers:
            p_shares = share_data[:, providers.index(p)]
            ax_share.bar(
                steps, 
                p_shares, 
                bottom=bottom, 
                color=color_map[p.name], 
                width=1.0, 
                edgecolor='none'
            )
            bottom += p_shares
            
        ax_share.set_ylim(0, 1.0)
        ax_share.set_ylabel("Share")
        ax_share.set_xlabel("Iteration")

        ax_pot = axes[i, 2]
        ax_pot.set_title(f"Beta={beta}: Potential")
        ax_pot.plot(steps, pot_hist, color='purple', linewidth=1.5)
        ax_pot.set_ylabel(r"Potential $\Phi$")
        ax_pot.set_xlabel("Iteration")

    sanity_path = f"{base_filename}/check_one.png"
    if os.path.dirname(sanity_path):
        os.makedirs(os.path.dirname(sanity_path), exist_ok=True)
        
    fig.savefig(sanity_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    display(Image(filename=sanity_path))

def plot_combined_dynamics(providers, thetas, betas_to_test, results, base_filename=None, colors=None, reasoning=False):
    latexify(font_size=10, small_font_size=8)
    

    width_per_plot = ICML_WIDTH_TEXT_PT / 3.0
    

    figsize = get_fig_dim(width_per_plot, fraction=1.0, aspect_ratio=0.9)
    
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
        ineff_ylim = (0, max(5.0, g_max * 1.1))
    else:
        ineff_ylim = (0, 10.0)

    for beta in betas_to_test:
        history, poa_hist, pot_hist, share_hist = results[beta]
        total_steps = len(poa_hist)
        steps = range(total_steps)
        beta_str = str(beta).replace(".", "p")
        
        fig, (ax_strat, ax_share) = plt.subplots(
            nrows=2, 
            ncols=1, 
            figsize=figsize, 
            sharex=True,
            gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.08} 
        )


        ax_strat.spines['top'].set_visible(False)
        ax_strat.get_yaxis().tick_left()
        ax_strat.tick_params(axis='both', which='major', labelsize=8)

        for provider_name, trajectory in history.items():
            traj_data = np.array(trajectory[:total_steps], dtype=float)

            traj_data += provider_offsets[provider_name]
            
            ax_strat.plot(
                steps, 
                traj_data, 
                drawstyle='steps-post', 
                linewidth=1.2,
                color=color_map[provider_name],
                alpha=0.9
            )
            
        ax_strat.set_ylabel(r"Compute $\theta$") 
        
        if not reasoning:
            ax_strat.set_yticks(range(len(thetas)))
            power_labels = [f"$2^{{{int(np.log2(t))}}}$" for t in thetas]
            ax_strat.set_yticklabels(power_labels, fontsize=8)
            ax_strat.set_ylim(-0.4, len(thetas) - 0.6)
        else:
            ax_strat.set_yticks(range(len(thetas)))
            ax_strat.set_yticklabels([f"{t}" for t in thetas], fontsize=8)

        ax_ineff = ax_strat.twinx()
        ineff_data = (np.array(poa_hist) - 1.0) * 100
        ax_ineff.plot(steps, ineff_data, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
        
        ax_ineff.set_ylabel(r"Ineff. (\%)", color='black', fontsize=9) 
        ax_ineff.tick_params(axis='y', labelcolor='black', labelsize=8)
        ax_ineff.set_ylim(ineff_ylim)
        ax_ineff.spines['top'].set_visible(False)


        ax_share.spines['top'].set_visible(False) 
        ax_share.spines['right'].set_visible(False)
        ax_share.get_xaxis().tick_bottom()
        ax_share.get_yaxis().tick_left()
        ax_share.tick_params(axis='both', which='major', labelsize=8)

        share_data = np.array(share_hist)
        bottom = np.zeros(total_steps)
        
        for i, p in enumerate(providers):
            provider_shares = share_data[:, i]
            ax_share.bar(
                steps, 
                provider_shares, 
                bottom=bottom, 
                color=color_map[p.name], 
                width=1.0, 
                edgecolor='none', 
                label=p.name,
                alpha=0.9
            )
            bottom += provider_shares
            
        ax_share.set_ylabel("Share")
        ax_share.set_xlabel(r"Iteration, $t$")
        ax_share.set_ylim(0, 1.0)
        ax_share.set_yticks([0, 0.5, 1.0]) 
        ax_share.set_yticklabels(["0", ".5", "1"], fontsize=8)
        ax_share.set_xlim(-0.5, total_steps - 0.5)

        out_file = f"{base_filename}/beta{beta_str}/combined.pdf"
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)

        fig_pot, ax_pot = plt.subplots(figsize=get_fig_dim(width_per_plot, fraction=1.0, aspect_ratio=0.75))
        ax_pot.spines['top'].set_visible(False)
        ax_pot.spines['right'].set_visible(False)
        ax_pot.get_xaxis().tick_bottom()
        ax_pot.get_yaxis().tick_left()
        ax_pot.tick_params(axis='both', which='major', labelsize=8)
        
        ax_pot.plot(steps, pot_hist, color='purple', linewidth=1.5)
        ax_pot.set_ylabel(r"Potential $\Phi$")
        ax_pot.set_xlabel(r"Iteration, $t$")

        out_pot = f"{base_filename}/beta{beta_str}/potential.pdf"
        fig_pot.savefig(out_pot, bbox_inches='tight', pad_inches=0.02)
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

    out_leg = f"{base_filename}/legend.pdf"
    fig_leg.savefig(out_leg, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig_leg)

    _sanity_check_combined(
        results, betas_to_test, providers, thetas, color_map, provider_offsets, ineff_ylim, reasoning, base_filename
    )

def plot_beta_sweep(beta_values, final_poas, base_filename="beta_sweep"):
    latexify(font_size=10, small_font_size=8)
    figsize = get_fig_dim(ICML_WIDTH_TEXT_PT, fraction=1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', which='major', labelsize=8)

    ineff_values = (np.array(final_poas) - 1.0) * 100
    
    ax.plot(beta_values, ineff_values, linestyle='-', linewidth=1.5, color='black')
    
    ax.set_xscale('log') 
    ax.set_xlabel(r"User Rationality $\beta$")
    ax.set_ylabel(r"Market Inefficiency (\%)")
    
    if len(ineff_values) > 0:
        y_max = max(ineff_values)
        ax.set_ylim(0, max(5.0, y_max * 1.1)) 
    
    output_pdf = f"{base_filename}inefficiency.pdf"
    if os.path.dirname(output_pdf):
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)
    fig.savefig(output_pdf, bbox_inches='tight', pad_inches=0.02)

    output_png = f"{base_filename}inefficiency.png"
    fig.savefig(output_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    display(Image(filename=output_png))