import torch
import numpy as np

from models.dm_utils import extract, remove_mean
from sampler.denoise_sampler import diffusion_sampler
from sampler.ais import DenoisingLangevinDynamics

def dsm_loss(model, x, opt):
    batch_size = x.shape[0]
    t=torch.randint(1, model.num_steps,[batch_size], device=x.device)

    a = extract(model.alphas_bar_sqrt, t, x)
    sigma = extract(model.one_minus_alphas_bar_sqrt, t, x)

    if opt.e3:
        e = remove_mean(torch.randn_like(x), n_particles=opt.n_particles, n_dimensions=opt.n_dim)
    else:
        e = torch.randn_like(x)

    x = x * a + e * sigma
    
    score = model(t, x)
    return (e + sigma * score).square().sum(-1).mean(0)

def diKL_loss(score_model, x, process, opt, target):
    t = torch.randint(1, score_model.num_steps,[x.shape[0]], device=x.device)
    a = extract(score_model.alphas_bar_sqrt.to(opt.device), t, x)
    sigma = extract(score_model.one_minus_alphas_bar_sqrt.to(opt.device), t, x)
    e = process(torch.randn_like(x))
    x_t = (x * a + e * sigma)
    x_t_clone = x_t.detach().clone()

    # estimate data score with posterior sampling (AIS/IS/LG/HMC)
    x_0_init = diffusion_sampler(target_log_p = lambda x: target.log_prob(x),
                                    ais_steps=opt.ais.AIS_step,
                                    ais_step_size=opt.ais.ais_step_size,
                                    x_t=x_t_clone,
                                    bar_alpha_t=a**2,
                                    n_is=opt.n_is,
                                    hmc_step=opt.ais.hmc_step,
                                    trunc_lower_bound=-1.0*opt.boundary,
                                    trunc_upper_bound=opt.boundary, 
                                    device=opt.device,
                                    resample=True,
                                    verbose=opt.verbose,
                                    LG=opt.ais.lg,
                                    lg_step=opt.ais.lg_step,
                                    mean_center=lambda x: process(x)
                                    )
    if opt.num_langevin_steps > 0:
        def denoising_LD_generator(energy, x, tx, alpha, std, step_size, mh, device):
            return DenoisingLangevinDynamics(x, 
                                            energy,
                                            lambda z: ((tx-alpha*z)**2/(2*std**2)).sum(-1),
                                            step_size=step_size,
                                            mh=mh,
                                            device=device)
        langevin_dynamics = denoising_LD_generator(lambda x: -target.log_prob(x, smooth_=True), x_0_init.detach().clone(), x_t.detach().clone(), a, sigma, step_size=opt.langevin_step_size, mh=True, device=opt.device)
        Acc = []
        X0s = []
        GRADs = []
        for lg_it in range(opt.num_langevin_steps):
            x_0, acc, gradx0 = langevin_dynamics.sample()
            # print(f"GRADs: {gradx0.min()} | {gradx0.max()}")
            # print(f"x_0: {x_0.min()} | {x_0.max()}")
            # print(f"x_0_init: {x_0_init.min()} | {x_0_init.max()}")
            Acc.append(acc)

            x_0 = process(x_0).detach().clone()
            gradx0 = process(gradx0).detach().clone()
            X0s.append(x_0)
            GRADs.append(gradx0)
        if opt.dynamic_step_size:
            if np.mean(Acc) > 0.6:
                opt.langevin_step_size *= 1.5
            elif np.mean(Acc) < 0.5:
                opt.langevin_step_size /= 1.5
        if opt.verbose:
            print('LG ACC %.3f'%acc, 'Step size %.4f'%opt.langevin_step_size)
    else:
        x_0 = x_0_init

    x_0 = torch.stack(X0s, 0)[-opt.sample_size:].mean(0)
    _target_score = -torch.stack(GRADs, 0)[-opt.sample_size:].mean(0)

    if opt.grad_estimator == 'msm':
        target_score = a * (x_0 + _target_score) - x_t
    elif opt.grad_estimator == 'tsm':
        target_score = _target_score / a
    else:
        raise NotImplementedError
    
    lvm_score = score_model(t, x_t)

    if opt.weight=="uniform":
        w=1
    elif opt.weight=="1/a":
        w=1/a
    elif opt.weight=="s2/a":
        w=sigma**2/a
    elif opt.weight=="s2/a2":
        w=sigma**2/a**2
    else:
        raise NotImplementedError
    
    score_diff = w*(lvm_score - target_score).detach()
    lvm_loss=(score_diff*x_t).sum(1).mean()

    # print(f"lvm_score: {lvm_score.min()} | {lvm_score.max()}")
    # print(f"target_score: {target_score.min()} | {target_score.max()}")
    # print(f"a: {a}")
    # print(f"before w: {(lvm_score - target_score).detach()} |  shape: {(lvm_score - target_score).detach().shape}")
    # print(f"before w max: {(lvm_score - target_score).detach().sum(1).max()}")
    # print(f"before w min: {(lvm_score - target_score).detach().sum(1).min()}")
    # diff_sum = (lvm_score - target_score).detach().sum(1)
    # num_small = (diff_sum < -100).sum().item()
    # num_big = (diff_sum > 100).sum().item()
    # print(f"Number of values < -50: {num_small}")
    # print(f"Number of values > 50: {num_big}")
    # from matplotlib import pyplot as plt
    # plt.hist((lvm_score - target_score).detach().sum(1).cpu().numpy(), bins=100)
    # plt.savefig("loss.png")
    # print(f"after w: {score_diff} |  shape: {score_diff.shape}")
    # print(f"after w max: {score_diff.argmax()}")
    # print(f"x_t: {x_t} |  shape: {x_t.shape}")
    # print(f"lvm_loss: {lvm_loss}")
    # input()
    return lvm_loss, X0s[-1]
