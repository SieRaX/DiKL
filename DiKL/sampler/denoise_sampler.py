from sampler.ais import *
import numpy as np
import torch
import tqdm.notebook as tqdm
from scipy.stats import truncnorm
from torch.distributions import Normal, Gumbel

def diffusion_sampler(target_log_p: callable,
                      ais_steps: int,
                      ais_step_size: float,
                      x_t: any,
                      bar_alpha_t: any,
                      n_is: int,
                      hmc_step: int = 1,
                      trunc_lower_bound=-np.inf,
                      trunc_upper_bound=np.inf, 
                      device='cpu',
                      verbose=False,
                      resample=True,
                      LG=False,
                      lg_step=0,
                      mean_center=None):
    energy = lambda x: -target_log_p(x)
    x_t = x_t.detach().clone()
    x_0_init_repeat = x_t.clone()[None, ...].repeat([n_is, 1, 1])
    
    x_importance_sample_loc = (x_0_init_repeat.detach()/bar_alpha_t.sqrt()).cpu().numpy()
    x_importance_sample_scale = ((1-bar_alpha_t).sqrt()/bar_alpha_t.sqrt()).detach().cpu().numpy()
    a = (trunc_lower_bound - x_importance_sample_loc) / x_importance_sample_scale
    b = (trunc_upper_bound - x_importance_sample_loc) / x_importance_sample_scale
    # define energy truncated to a certain range
    truncated_energy_lower = lambda x: torch.where(x> (trunc_lower_bound if type(trunc_lower_bound) == float else torch.from_numpy(trunc_lower_bound).to(device)), torch.zeros_like(x), torch.ones_like(x)*np.inf)
    truncated_energy = lambda x: torch.where(x<(trunc_upper_bound if type(trunc_upper_bound) == float else torch.from_numpy(trunc_upper_bound).to(device)), truncated_energy_lower(x), torch.ones_like(x)*np.inf).sum(-1)

    x_importance_sample = truncnorm.rvs(a=a, b=b, loc=x_importance_sample_loc, scale=x_importance_sample_scale)
    x_importance_sample = torch.from_numpy(x_importance_sample).to(device)

    if mean_center is not None:
        x_importance_sample = mean_center(x_importance_sample)

    print(f"x_importance_sample.shape: {x_importance_sample.shape}")
    is_weights = -energy(x_importance_sample) \
                    + Normal(x_importance_sample*bar_alpha_t.sqrt(), (1-bar_alpha_t)**0.5).log_prob(x_t).sum(-1) \
                    - Normal(x_0_init_repeat/bar_alpha_t.sqrt(), (1-bar_alpha_t)**0.5 / bar_alpha_t**0.5).log_prob(x_importance_sample).sum(-1)

    final_target_log_p = lambda x:  -energy(x) + Normal(x*bar_alpha_t.sqrt(), (1-bar_alpha_t)**0.5).log_prob(x_t).sum(-1) - torch.clamp(truncated_energy(x), 0, np.inf)
    initial_proposal_log_p = lambda x:  Normal(x_0_init_repeat/bar_alpha_t.sqrt(), (1-bar_alpha_t)**0.5 / bar_alpha_t**0.5).log_prob(x).sum(-1) - torch.clamp(truncated_energy(x), 0, np.inf)
    is_weights, x_importance_sample = AIS(ais_steps, hmc_step, ais_step_size, x_importance_sample, initial_proposal_log_p, final_target_log_p, verbose, device=device, LG=LG, LG_step=lg_step)

    if not resample:
        return x_importance_sample, is_weights
    else:
        gumbeled_density_ratio = is_weights + Gumbel(torch.tensor(0.0, device=device), torch.tensor(1.0, device=device)).sample(is_weights.shape)
        idx = gumbeled_density_ratio.argmax(0)
        x_0 = x_importance_sample.permute(1, 0, 2)[np.arange(x_importance_sample.shape[1]), idx, :]
        return x_0