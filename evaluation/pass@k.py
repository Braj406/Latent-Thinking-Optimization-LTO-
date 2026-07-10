import math

def calculate_pass_at_k(n, c, k):
    """
    Calculates the Pass@k metric for code generation trajectories.
    n: total number of generated samples for a problem
    c: number of correct samples
    k: the 'k' in Pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))
