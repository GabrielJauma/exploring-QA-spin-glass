import numpy as np
from tqdm import tqdm


def bootstrap_error_B(µ_q2_t, µ_q4_t, n_bootstrap=1000, out='error'):
    N = min(µ_q4_t.shape[0], µ_q2_t.shape[0])
    copies = µ_q2_t.shape[1]
    B_bootstrap = np.zeros([n_bootstrap, copies])
    for i in tqdm(range(n_bootstrap)):
        # bootstrap_indices = np.random.randint(N, size=N)
        µ_q2_t_bootstrap = µ_q2_t[np.random.randint(N, size=N), :]
        µ_q4_t_bootstrap = µ_q4_t[np.random.randint(N, size=N), :]
        # µ_q2_t_bootstrap = µ_q2_t[bootstrap_indices, :]
        # µ_q4_t_bootstrap = µ_q4_t[bootstrap_indices, :]
        B_bootstrap[i, :] = 0.5 * (3 - np.mean(µ_q4_t_bootstrap, 0) / (np.mean(µ_q2_t_bootstrap, 0) ** 2))
    if out == 'error':
        return np.std(B_bootstrap, 0)
    elif out == 'B_bootstrap':
        return B_bootstrap
    elif out == 'both':
        return np.std(B_bootstrap, 0), B_bootstrap
