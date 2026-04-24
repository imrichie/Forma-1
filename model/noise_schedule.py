import numpy as np

def get_noise_schedule(timesteps, beta_start, beta_end):
    """
    Computes the linear noise schedule used in the original DDPM paper.

    Returns:
        beta      — noise added at each timestep (T,)
        alpha     — signal preserved at each timestep (T,)
        alpha_bar — cumulative signal remaining after t steps (T,)
    """
    beta      = np.linspace(beta_start, beta_end, timesteps, dtype=np.float32)
    alpha     = 1.0 - beta
    alpha_bar = np.cumprod(alpha, dtype=np.float32)

    print(f'Beta  : t=0 → {beta[0]:.6f}  |  t={timesteps-1} → {beta[-1]:.6f}')
    print(f'Alpha bar: t=0 → {alpha_bar[0]:.6f}  |  t={timesteps-1} → {alpha_bar[-1]:.6f}')

    return beta, alpha, alpha_bar