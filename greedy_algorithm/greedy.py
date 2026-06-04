import numpy as np

def facility_location_score(S_indices, sim_matrix):
    """
    Calculates the value of the Facility Location submodular function.
    """
    if not S_indices:
        return 0.0
    
    # For every element in V (all rows), we take the max similarity to the selected subset S
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

# # --- Example usage ---
# V_size = 100  # e.g., 100 potential sensor locations or 100 features
# k = 5         # We want to select 5 elements

# # Create a random similarity matrix (simulating correlation or distance)
# np.random.seed(42)
# sim_matrix = np.random.rand(V_size, V_size)

# # The similarity matrix should be symmetric with 1.0 on the diagonal
# sim_matrix = (sim_matrix + sim_matrix.T) / 2
# np.fill_diagonal(sim_matrix, 1.0)

# # Run the algorithm
# selected_elements = greedy_submodular_maximization(V_size, k, sim_matrix)
# print("Optimal subset (indices):", selected_elements)