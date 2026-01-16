from dataclasses import dataclass
import copy
import random
from typing import Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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



@dataclass
class MarketConfig:
    alpha: float = 10.0
    beta: float = 1.0
    default_margin: float = 0.25

    value_curve: str = "exp" # "linear", "power", "exp"
    baseline_frac: float = 0.1 

    kappa: float = 6.0 # for "exp"
    gamma: float = 3.5 # for "power"


MODEL_PRICES_PER_1M = {
    "Llama-3-8B": 0.1455,
    "Llama-3.1-8B": 0.1245,
    "Llama-3.2-1B": 0.10,
    "Llama-3.2-3B": 0.08,
    "Qwen2-0.5B": 0.10,
    "Qwen2-1.5B": 0.10,
    "Qwen2-7B": 0.20,
    "Qwen2.5-3B": 0.065,
    "Qwen2.5-7B": 0.1465,
    "reason-R1-D-Llama-8B": 0.1250,
    "reason-R1-D-Qwen-1.5B": 0.1000,
    "reason-R1-D-Qwen-7B": 0.1750
}

class Provider:
    def __init__(
        self, 
        name: str, 
        base_model_key: str, 
        accuracy_curve: np.ndarray, 
        token_curve: np.ndarray, 
        margin: Optional[float] = None
    ):
        self.name = name
        self.accuracy_curve = accuracy_curve
        self.token_curve = token_curve
        

        price_per_1m = 0.0
        for key, price in MODEL_PRICES_PER_1M.items():
            if key in base_model_key:
                price_per_1m = price
                break
        
        if price_per_1m == 0.0:
            print(f"Warning: No price found for {base_model_key}, defaulting to $0.10")
            price_per_1m = 0.10
            
        self.price_per_token = price_per_1m / 1_000_000
        self.margin = margin  
    
    def get_economics(self, theta_idx: int, config: MarketConfig):
        raw_accuracy = float(self.accuracy_curve[theta_idx])
        raw_tokens = float(self.token_curve[theta_idx])

        p_theta = raw_tokens * self.price_per_token
        c_theta = p_theta / (1 + self.margin)

        a = float(raw_accuracy)
        b = float(config.baseline_frac)

        if config.value_curve == "linear":
            f = a
        elif config.value_curve == "power":
            f = a ** config.gamma
        elif config.value_curve == "exp":
            denom = np.expm1(config.kappa)
            f = (np.expm1(config.kappa * a) / denom) if denom != 0 else a
        else:
            raise ValueError(f"Unknown value_curve: {config.value_curve}")

        q_theta = config.alpha * (b + (1 - b) * f)

        v_theta = q_theta - p_theta

        return {
            "q": q_theta,
            "p": p_theta,
            "c": c_theta,
            "V": v_theta,
            "raw_acc": raw_accuracy,
            "profit_margin": p_theta - c_theta
        }
    
def get_market_shares(values: np.ndarray, beta: float, v0: float = 0.0) -> np.ndarray:
    max_val = np.max(np.append(values, v0))
    
    exp_values = np.exp(beta * (values - max_val))
    exp_v0 = np.exp(beta * (v0 - max_val))
    
    denominator = exp_v0 + np.sum(exp_values)
    return exp_values / denominator

def get_utility(share: float, p: float, c: float) -> float:
    return share * (p - c)


import numpy as np
from typing import List

def get_global_max_welfare_brute_force_fast(
    providers: List[Provider], 
    config: MarketConfig, 
    v0: float = 0.0,
    batch_size: int = 1_000_000 
) -> float:

    num_providers = len(providers)
    num_thetas = len(providers[0].accuracy_curve)
    

    V_matrix = np.zeros((num_providers, num_thetas), dtype=np.float32)
    SW_matrix = np.zeros((num_providers, num_thetas), dtype=np.float32)
    
    for i, p in enumerate(providers):
        for t_idx in range(num_thetas):
            econ = p.get_economics(t_idx, config)
            V_matrix[i, t_idx] = econ['V']
            SW_matrix[i, t_idx] = econ['q'] - econ['c']

    total_combinations = num_thetas ** num_providers
    
    max_welfare = -np.inf
    

    for start_idx in range(0, total_combinations, batch_size):
        end_idx = min(start_idx + batch_size, total_combinations)
        current_batch_size = end_idx - start_idx
        
        indices = np.zeros((current_batch_size, num_providers), dtype=int)
        temp_indices = np.arange(start_idx, end_idx)
        
        for p in range(num_providers - 1, -1, -1):
            indices[:, p] = temp_indices % num_thetas
            temp_indices //= num_thetas
            
        batch_Vs = V_matrix[np.arange(num_providers), indices]
        batch_SWs = SW_matrix[np.arange(num_providers), indices]
        
        max_vals = np.maximum(np.max(batch_Vs, axis=1), v0)[:, np.newaxis]
        
        exp_vals = np.exp(config.beta * (batch_Vs - max_vals))
        exp_v0 = np.exp(config.beta * (v0 - max_vals)).flatten()
        
        sum_exp = np.sum(exp_vals, axis=1) + exp_v0
        
        shares = exp_vals / sum_exp[:, np.newaxis]
        

        batch_welfares = np.sum(shares * batch_SWs, axis=1)
        
        current_batch_max = np.max(batch_welfares)
        if current_batch_max > max_welfare:
            max_welfare = current_batch_max

    print(f"Fast Global Optimization Complete. Max Welfare: {max_welfare:.4f}")
    return float(max_welfare)

def get_global_max_welfare(providers: List[Provider], config: MarketConfig) -> float:
    max_welfare = -np.inf
    
    for p in providers:
        for theta_idx in range(len(p.accuracy_curve)):
            econ = p.get_economics(theta_idx, config)
            welfare = econ['q'] - econ['c']
            if welfare > max_welfare:
                max_welfare = welfare
                
    return max_welfare


def calculate_current_welfare(
    providers: List[Provider], 
    strategies: np.ndarray, 
    shares: np.ndarray, 
    config: MarketConfig
) -> float:
    total_welfare = 0.0
    for i, p in enumerate(providers):
        theta_idx = strategies[i]
        econ = p.get_economics(theta_idx, config)
        sw_i = econ['q'] - econ['c']
        total_welfare += shares[i] * sw_i
        
    return total_welfare

def calculate_potential(
    providers: List[Provider], 
    strategies: np.ndarray, 
    config: MarketConfig,
    v0: float = 0.0
) -> float:

    beta = config.beta
    log_profit_term = 0.0
    sum_V = 0.0
    

    exponents = [beta * v0]

    for i, p in enumerate(providers):
        theta_idx = strategies[i]
        econ = p.get_economics(theta_idx, config)
        
        profit = econ['profit_margin']
        if profit <= 1e-12: 
            return -np.inf 
        log_profit_term += np.log(profit)
        
        v_i = econ['V']
        sum_V += v_i
        
        exponents.append(beta * v_i)

    exponents = np.array(exponents)
    max_exp = np.max(exponents)
    

    term_3 = max_exp + np.log(np.sum(np.exp(exponents - max_exp)))

    term_1 = log_profit_term
    term_2 = beta * sum_V
    
    return term_1 + term_2 - term_3

def run_simulation_sequential_robust(
    providers: List[Provider], 
    config: MarketConfig, 
    max_iter: int = 50,
    v0: float = 0.0,
    tolerance: float = 0.01
):
    num_providers = len(providers)
    num_strategies = len(providers[0].accuracy_curve)
    
    max_possible_sw = get_global_max_welfare_brute_force_fast(providers, config)
    current_strategies = np.zeros(num_providers, dtype=int)
    
    history = {p.name: [0] for p in providers} 
    poa_history = []
    potential_history = [] 
    share_history = []  
    
    init_economics = [prov.get_economics(0, config) for prov in providers]
    init_Vs = np.array([e['V'] for e in init_economics])
    init_shares = get_market_shares(init_Vs, config.beta, v0)
    init_sw = calculate_current_welfare(providers, current_strategies, init_shares, config)
    
    poa_history.append(1.0 if init_sw <= 1e-9 else max_possible_sw / init_sw)
    potential_history.append(calculate_potential(providers, current_strategies, config, v0))
    share_history.append(init_shares) 

    provider_indices = list(range(num_providers))

    for t in range(max_iter):
        strategies_changed_this_round = False
        random.shuffle(provider_indices)
        
        for i in provider_indices:
            p = providers[i]
            
            current_economics = [prov.get_economics(current_strategies[k], config) 
                                 for k, prov in enumerate(providers)]
            current_Vs = np.array([e['V'] for e in current_economics])
            current_shares = get_market_shares(current_Vs, config.beta, v0)
            
            current_econ = current_economics[i]
            current_util = get_utility(current_shares[i], current_econ['p'], current_econ['c'])
            
            best_util = current_util
            target_strat_idx = current_strategies[i] 
            other_Vs = current_Vs.copy()

            for theta_candidate in range(num_strategies):
                if theta_candidate == current_strategies[i]:
                    continue 
                
                cand_econ = p.get_economics(theta_candidate, config)
                hypothetical_Vs = other_Vs.copy()
                hypothetical_Vs[i] = cand_econ['V']
                
                shares = get_market_shares(hypothetical_Vs, config.beta, v0)
                util = get_utility(shares[i], cand_econ['p'], cand_econ['c'])
                
                margin = abs(best_util) * tolerance
                if margin == 0: margin = 1e-9
                improvement_threshold = best_util + margin
                
                if util > improvement_threshold:
                    best_util = util
                    target_strat_idx = theta_candidate
            
            current_idx = current_strategies[i]
            new_idx = current_idx
            if target_strat_idx > current_idx: new_idx = current_idx + 1
            elif target_strat_idx < current_idx: new_idx = current_idx - 1
            
            if new_idx != current_idx:
                current_strategies[i] = new_idx
                strategies_changed_this_round = True
            
            for k, prov in enumerate(providers):
                history[prov.name].append(int(current_strategies[k]))
                
            micro_economics = [prov.get_economics(current_strategies[k], config) for k, prov in enumerate(providers)]
            micro_Vs = np.array([e['V'] for e in micro_economics])
            micro_shares = get_market_shares(micro_Vs, config.beta, v0)
            micro_sw = calculate_current_welfare(providers, current_strategies, micro_shares, config)
            
            poa = 1.0 if micro_sw <= 1e-9 else max_possible_sw / micro_sw
            poa_history.append(poa)
            potential_history.append(calculate_potential(providers, current_strategies, config, v0))
            share_history.append(micro_shares)

        if not strategies_changed_this_round:
            break

    return history, poa_history, potential_history, share_history

def plot_market_shares(providers, betas_to_test, results, base_filename="market_sim", colors=None):

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
        
        out_file = f"{base_filename}_beta{beta_str}_shares.pdf"
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {out_file}")
        plt.close(fig)

def sweep_beta_vs_poa(
    providers: List[Provider], 
    base_config: MarketConfig, 
    beta_values: np.ndarray
):
    final_poas = []
    final_potentials = []
        
    for i, b in enumerate(beta_values):
        sim_config = copy.copy(base_config)
        sim_config.beta = b
        
        _, poa_hist, potential_history, _ = run_simulation_sequential_robust(
            providers, 
            sim_config, 
            max_iter=1000,      
            v0=0.0, 
            tolerance=0.00000001

        )
        
        final_poa = poa_hist[-1]
        final_poas.append(final_poa)
        final_potentials.append(potential_history[-1])
            
    return final_poas, final_potentials


def compute_V_curves(thetas, providers, config):
    V = {}
    for pr in providers:
        V[pr.name] = np.array([pr.get_economics(i, config)["V"] for i in range(len(thetas))], dtype=float)
    return V

def style_ax(ax):
    """Common styling to remove top/right spines."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', which='major', labelsize=8)


def plot_beta_sweep(beta_values, final_poas, base_filename="beta_sweep"):
    latexify(font_size=10, small_font_size=8)
    
    figsize = get_fig_dim(ICML_WIDTH_COL_PT, fraction=1.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    style_ax(ax)
    

    ineff_values = (np.array(final_poas) - 1.0) * 100
    
    ax.plot(beta_values, ineff_values, linestyle='-', linewidth=1.5, color='black')
    

    ax.set_xscale('log') 
    ax.set_xlabel(r"User Rationality $\beta$")
    ax.set_ylabel(r"Market Inefficiency (\%)")
    
    if len(ineff_values) > 0:
        y_max = max(ineff_values)
        ax.set_ylim(0, max(5.0, y_max * 1.1)) 
    
    output_filename = f"{base_filename}_inefficiency.pdf"
    fig.savefig(output_filename, bbox_inches='tight', pad_inches=0.02)
    print(f"Saved plot to {output_filename}")
    plt.close(fig)


def plot_combined_dynamics(providers, thetas, betas_to_test, results, base_filename="market_sim", colors=None, reasoning=False):
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

        out_file = f"{base_filename}_beta{beta_str}_combined.pdf"
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {out_file}")
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