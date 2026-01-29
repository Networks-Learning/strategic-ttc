import numpy as np
import copy

def calculate_auction_outcome(providers, value_curves, base_config):
    auction_candidates = []

    for prov in providers:
        best_w = -np.inf
        best_stats = None

        possible_moves = range(len(value_curves[prov.name])) 
        
        for move in possible_moves:
            stats = prov.get_economics(move, base_config) 
        
            val = value_curves[prov.name][move] 
            cost = stats['c'] 
            quality = stats['q'] 
            
            w_contribution = val + stats['profit_margin']
            
            if w_contribution > best_w:
                best_w = w_contribution
                best_stats = {
                    'name': prov.name,
                    'q': quality,
                    'c': cost,
                    'W': w_contribution
                }
                
        auction_candidates.append(best_stats)

    auction_candidates.sort(key=lambda x: x['W'], reverse=True)
    winner = auction_candidates[0]
    runner_up = auction_candidates[1] 

    return {
        'winner': winner,
        'runner_up': runner_up,
        'sw': winner['W'],
        'user_value': runner_up['W'],
        'price': winner['q'] - runner_up['W'],
        'utility': winner['W'] - runner_up['W']
    }

def calculate_market_stats(beta, market_dynamics, value_curves, providers, base_config, profit_margin):
    final_moves = {model: moves[-1] for model, moves in market_dynamics[beta][0].items()}
    
    shares = market_dynamics[beta][3][-1]
    
    final_values = []
    for model_name, move in final_moves.items():
        for v_key in value_curves.keys():
            if model_name in v_key:
                final_values.append(value_curves[v_key][move])
                break
    
    sim_config = copy.copy(base_config) 
    sim_config.beta = beta
    sim_config.default_margin = profit_margin
    
    micro_economics = [
        prov.get_economics(final_moves[prov.name], sim_config) 
        for prov in providers
    ]

    avg_w_value = np.sum(np.array(final_values) * np.array(shares))
    avg_value = np.mean(final_values)
    
    avg_w_price = np.sum([micro_economics[i]['p'] * shares[i] for i in range(len(providers))])
    avg_price = np.mean([micro_economics[i]['p'] for i in range(len(providers))])
    
    avg_w_utility = np.sum([micro_economics[i]['profit_margin'] * shares[i] for i in range(len(providers))])
    avg_utility = np.mean([micro_economics[i]['profit_margin'] for i in range(len(providers))])
    
    opt_sw, eq_sw = market_dynamics[beta][4][-1]

    return {
        'avg_w_value': avg_w_value,
        'avg_value': avg_value,
        'opt_sw': opt_sw,
        'eq_sw': eq_sw,
        'avg_w_price': avg_w_price,
        'avg_price': avg_price,
        'avg_w_utility': avg_w_utility,
        'avg_utility': avg_utility
    }

def compare_market_to_auction(betas, market_dynamics, value_curves, providers, base_config, profit_margin):
    auction = calculate_auction_outcome(providers, value_curves, base_config)

    for beta in betas:
        m_stats = calculate_market_stats(
            beta, market_dynamics, value_curves, providers, base_config, profit_margin
        )
        
        print(f"BETA: {beta}")
        print(f"\tAverage Weighted Value: {m_stats['avg_w_value']:.5f}")
        print(f"\tAverage Value: {m_stats['avg_value']:.5f}")
        print(f"\tOptimal SW: {m_stats['opt_sw']:.5f}, Equilibrium SW: {m_stats['eq_sw']:.5f}")
        print(f"\tAverage Weighted Price: {m_stats['avg_w_price']:.5f}")
        print(f"\tAverage Price: {m_stats['avg_price']:.5f}")
        print(f"\tAverage Weighted Utility: {m_stats['avg_w_utility']:.5f}")
        print(f"\tAverage Utility: {m_stats['avg_utility']:.5f}\n")

    print(f"AUCTION RESULTS")
    print(f"Winner: {auction['winner']['name']} (W={auction['winner']['W']:.5f})")
    print(f"Runner-up: {auction['runner_up']['name']} (W={auction['runner_up']['W']:.5f})")
    print(f"")
    print(f"\tRealized User Value: {auction['user_value']:.5f}")
    print(f"\tRealized Price: {auction['price']:.5f}")
    print(f"\tRealized Utility: {auction['utility']:.5f}")
    print(f"\tTotal SW: {auction['sw']:.5f}")
    print("-" * 100)