import numpy as np

def component_masses(M_range, q_range, n_samples=1000):
    """
    Compute component mass ranges from chirp mass and mass ratio ranges.
    
    Parameters:
        M_range : tuple (M_min, M_max)
        q_range : tuple (q_min, q_max)  (q = m1/m2 >= 1)
        n_samples : resolution of grid
    
    Returns:
        m1_range, m2_range
    """
    
    M_vals = np.linspace(M_range[0], M_range[1], n_samples)
    q_vals = np.linspace(q_range[0], q_range[1], n_samples)
    
    m1_list = []
    m2_list = []
    
    for M in M_vals:
        for q in q_vals:
            m1 = M * (q**(2/5)) * ((1+q)**(1/5))
            m2 = m1 / q
            
            m1_list.append(m1)
            m2_list.append(m2)
    
    m1_array = np.array(m1_list)
    m2_array = np.array(m2_list)
    
    return (m1_array.min(), m1_array.max()), (m2_array.min(), m2_array.max())


# Example usage
M_range = (1.0, 1.7)
q_range = (1.0, 3.0)

m1_range, m2_range = component_masses(M_range, q_range)

print("m1 range:", m1_range)
print("m2 range:", m2_range)

