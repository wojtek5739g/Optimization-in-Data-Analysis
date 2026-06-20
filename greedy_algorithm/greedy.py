import numpy as np

def facility_location_score(S_indices, sim_matrix):
    """Calculates the value of the Facility Location submodular function."""
    if not S_indices:
        return 0.0
    return np.sum(np.max(sim_matrix[:, S_indices], axis=1))

def log_determinant_score(S_indices, sim_matrix):
    """Calculates the value of the Log-Determinant function for subset S."""
    if not S_indices:
        return 0.0
    sub_matrix = sim_matrix[np.ix_(S_indices, S_indices)]
    sub_matrix = sub_matrix + np.eye(len(S_indices))
    return np.log(np.linalg.det(sub_matrix))

def saturated_coverage_score(S_indices, sim_matrix, alpha=0.1):
    """Calculates the value of the Saturated Coverage function."""
    if not S_indices:
        return 0.0
    return np.sum(np.minimum(np.sum(sim_matrix[:, S_indices], axis=1), alpha))

def greedy_submodular_maximization(V_size, k, sim_matrix):
    """
    Classical greedy algorithm for submodular function maximization.
    Returns only the selected subset of indices.
    """
    S_indices = []
    print(f"Starting selection of {k} elements out of {V_size}...")
    
    for step in range(k):
        best_gain = -1
        best_element = None
        current_score = facility_location_score(S_indices, sim_matrix)
        
        for v in range(V_size):
            if v in S_indices:
                continue
                
            gain = facility_location_score(S_indices + [v], sim_matrix) - current_score
            if gain > best_gain:
                best_gain = gain
                best_element = v
                
        S_indices.append(best_element)
        print(f"Step {step+1}: Selected element {best_element}, Marginal Gain: {best_gain:.4f}")
        
    return S_indices

def greedy_submodular_maximization_2(V_size, k, sim_matrix, score_type='facility_location', alpha=0.1):
    """
    Extended greedy selection allowing custom submodular functions.
    Returns BOTH the selected indices and the history of marginal gains.
    """
    S_indices = []
    marginal_gains = []
    online_bounds = []
    
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
        all_remaining_gains = []

        for v in range(V_size):
            if v in S_indices:
                continue
                
            gain = score_fn(S_indices + [v]) - current_score
            all_remaining_gains.append(gain)
            if gain > best_gain:
                best_gain = gain
                best_element = v
        
        all_remaining_gains.sort(reverse=True)
        step_online_bound = current_score + sum(all_remaining_gains[:k])
        online_bounds.append(step_online_bound)

        S_indices.append(best_element)
        marginal_gains.append(best_gain)
        if step == 0 or (step + 1) % 10 == 0 or (step + 1) == k:
            print(f"Step {step+1}: Selected element {best_element}, Marginal Gain: {best_gain:.4f}")
            
    return S_indices, marginal_gains, online_bounds