import numpy as np
from dataclasses import dataclass
from typing import Optional, List
import random
import copy
from tqdm import tqdm

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


def init_providers(models_data, ttc_type="majority", margin=0.25):
    providers = []

    for model_name in models_data.keys():
        print(f"Initializing provider for {model_name}...")
        if ttc_type == "majority":
            accuracy_data = models_data[model_name][1]
            token_data = models_data[model_name][5]
        elif ttc_type == "best-of-n":
            accuracy_data = models_data[model_name][3]
            token_data = models_data[model_name][5]
        elif ttc_type == "relative_efforts":
            accuracy_data = models_data[model_name][0]
            token_data = models_data[model_name][2]
        else:
            raise ValueError(f"Unknown ttc_type: {ttc_type}; must be 'majority', 'best-of-n', or 'relative_efforts'.")

        key = model_name
        provider = Provider(
            name=model_name,
            base_model_key=key,
            accuracy_curve=accuracy_data,
            token_curve=token_data,
            margin=margin
        )
        providers.append(provider)

    return providers

def compute_V_curves(thetas, providers, config):
    V = {}
    for pr in providers:
        V[pr.name] = np.array([pr.get_economics(i, config)["V"] for i in range(len(thetas))], dtype=float)
    return V

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

    return float(max_welfare)

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


def compute_market_dynamics(betas_to_test, profit_margin, unreasoning_providers, config):
    market_dynamics = {}

    for b in tqdm(betas_to_test):
        sim_config = copy.copy(config) 
        sim_config.beta = b
        sim_config.default_margin = profit_margin
        
        hist, poa, pot, shares = run_simulation_sequential_robust(
            unreasoning_providers, 
            sim_config, 
            max_iter=1000, 
            v0=0.0,     
            tolerance=0.00000001
        )
        market_dynamics[b] = (hist, poa, pot, shares)

    return market_dynamics

def sweep_beta_vs_poa(
    providers: List[Provider], 
    base_config: MarketConfig, 
    beta_values: np.ndarray
):
    final_poas = []
    final_potentials = []

    for i, b in tqdm(enumerate(beta_values), total=len(beta_values)):
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