import numpy as np

def facility_location_score(S_indices, sim_matrix):
    """Calculates the value of the Facility Location submodular function."""
    if not S_indices:
        return 0.0
    return np.sum(np.max(sim_matrix[:, S_indices], axis=1))

def greedy_submodular_maximization(V_size, k, sim_matrix):
    """
    Classical greedy algorithm for submodular function maximization.
    """
    S_indices = []
    
    print(f"Starting selection of {k} elements out of {V_size}...")
    
    for step in range(k):
        best_gain = -1
        best_element = None
        
        # Current score of our selected subset
        current_score = facility_location_score(S_indices, sim_matrix)
        
        # Find the element that provides the highest marginal gain
        for v in range(V_size):
            if v in S_indices:
                continue
                
            # Calculate the score for subset S expanded by element v
            gain = facility_location_score(S_indices + [v], sim_matrix) - current_score
            
            if gain > best_gain:
                best_gain = gain
                best_element = v
                
        S_indices.append(best_element)
        print(f"Step {step+1}: Selected element {best_element}, Marginal Gain: {best_gain:.4f}")
        
    return S_indices

def log_determinant_score(S_indices, sim_matrix, eps=1e-4):
    """Calculates the value of the Log-Determinant submodular function."""
    if not S_indices:
        return 0.0
    # Wycinamy podmacierz korelacji dla wybranych indeksów
    sub_matrix = sim_matrix[np.ix_(S_indices, S_indices)]
    reg_matrix = sub_matrix + eps * np.eye(len(S_indices))
    sign, logdet = np.linalg.slogdet(reg_matrix)
    return logdet

def saturated_coverage_score(S_indices, sim_matrix, alpha=1.0):
    """Calculates the value of the Saturated Coverage submodular function."""
    if not S_indices:
        return 0.0
    total_sim_per_row = np.sum(sim_matrix[:, S_indices], axis=1)
    saturated_sim = np.minimum(total_sim_per_row, alpha)
    return np.sum(saturated_sim)

def greedy_submodular_maximization_2(V_size, k, score_type, sim_matrix, alpha=1.0):
    """
    Generalized greedy algorithm for submodular function maximization.
    Returns BOTH the selected feature indices AND the history of marginal gains.
    """
    S_indices = []
    marginal_gains = []
    
    if score_type == 'facility_location':
        score_fn = lambda S: facility_location_score(S, sim_matrix)
    elif score_type == 'log_determinant':
        score_fn = lambda S: log_determinant_score(S, sim_matrix)
    elif score_type == 'saturated_coverage':
        score_fn = lambda S: saturated_coverage_score(S, sim_matrix, alpha=alpha)
    else:
        raise ValueError("Invalid score_type. Choose 'facility_location', 'log_determinant' or 'saturated_coverage'.")
        
    print(f"Starting greedy selection using {score_type} (k={k})...")
    
    for step in range(k):
        best_gain = -np.inf
        best_element = None
        current_score = score_fn(S_indices)
        
        for v in range(V_size):
            if v in S_indices:
                continue
                
            gain = score_fn(S_indices + [v]) - current_score
            if gain > best_gain:
                best_gain = gain
                best_element = v
                
        S_indices.append(best_element)
        marginal_gains.append(best_gain)
        
        # Wyświetlamy co 10 kroków lub na początku/końcu, żeby nie zaśmiecać konsoli
        if step == 0 or (step + 1) % 10 == 0 or (step + 1) == k:
            print(f"  Step {step+1}/{k}: Selected feature {best_element}, Marginal Gain: {best_gain:.4f}")
        
    return S_indices, marginal_gains