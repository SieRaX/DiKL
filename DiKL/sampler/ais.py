import torch
import numpy as np
from functools import partial
import copy
from torch.distributions import Normal, Gumbel


def target_density_and_grad_fn_full(x, inv_temperature, target_log_prob_fn):
    x = x.clone().detach().requires_grad_(True)
    log_prob = target_log_prob_fn(x) * inv_temperature
    log_prob_sum = log_prob.sum()
    log_prob_sum.backward()
    grad = x.grad.clone().detach()
    return log_prob.detach(), grad



class LangevinDynamics(object):

    def __init__(self,
                 x: torch.Tensor,
                 energy_func: callable,
                 step_size: float,
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False):
        """
        Standard Langevin Dynamics Sampler
        """
        super(LangevinDynamics, self).__init__()

        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.mh= mh
        self.device = device
        self.point_estimator = point_estimator

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            self.f_x = f_xc.detach()
            self.grad_x = grad_xc.detach()

    def sample(self) -> tuple:
        if self.point_estimator == True:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]
            x_p = x_c - self.step_size * grad_xc 
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), None

        if self.mh == False:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc = self.energy_func(x_c)
            grad_xc = torch.autograd.grad(f_xc.sum(), x_c,create_graph=False)[0]

            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(x_c, device=self.device)
            
            self.x = x_p.detach()
            return copy.deepcopy(x_p.detach()), f_xc.detach()
        
        else:
            x_c = self.x.detach()
            f_xc = self.f_x.detach()
            grad_xc = self.grad_x.detach()
            
            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(self.x, device=self.device)
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp = self.energy_func(x_p)
            grad_xp = torch.autograd.grad(f_xp.sum(), x_p,create_graph=False)[0]
            log_joint_prob_2 = -f_xc-torch.norm(x_p-x_c+self.step_size * grad_xc, dim=-1)**2/(4*self.step_size)
            log_joint_prob_1 = -f_xp-torch.norm(x_c-x_p+self.step_size * grad_xp, dim=-1)**2/(4*self.step_size)

            log_accept_rate = log_joint_prob_1 - log_joint_prob_2
            is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
            is_accept = is_accept.unsqueeze(-1)

            self.x = torch.where(is_accept, x_p.detach(), self.x)
            self.f_x = torch.where(is_accept.squeeze(-1), f_xp.detach(), self.f_x)
            self.grad_x = torch.where(is_accept, grad_xp.detach(), self.grad_x)  

            acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
            
            return copy.deepcopy(self.x.detach()), acc_rate.item()
        

class HamiltonianMonteCarlo(object):

    def __init__(self,
                 x,
                 energy_func: callable,
                 step_size: float,
                 num_leapfrog_steps_per_hmc_step: int,
                 inv_temperature: float = 1.0,
                 device: str = 'cpu'):
        """
        Standard HMC Sampler
        """
        super(HamiltonianMonteCarlo, self).__init__()

        self.x = x
        self.step_size = step_size
        self.target_density_and_grad_fn = partial(target_density_and_grad_fn_full, target_log_prob_fn=lambda x: -energy_func(x))
        self.device = device
        self.inv_temperature = inv_temperature
        self.num_leapfrog_steps_per_hmc_step = num_leapfrog_steps_per_hmc_step

        self.current_log_prob, self.current_grad = self.target_density_and_grad_fn(x, self.inv_temperature)

    def leapfrog_integration(self, p):
        """
        Leapfrog integration for simulating Hamiltonian dynamics.
        """
        x = self.x.detach().clone()
        p = p.detach().clone()

        # Half step for momentum
        p += 0.5 * self.step_size * self.current_grad

        # Full steps for position
        for _ in range(self.num_leapfrog_steps_per_hmc_step - 1):
            x += self.step_size * p
            _, grad = self.target_density_and_grad_fn(x, self.inv_temperature)
            p += self.step_size * grad  # this combines two half steps for momentum

        # Final update of position and half step for momentum
        x += self.step_size * p
        new_log_prob, new_grad = self.target_density_and_grad_fn(x, self.inv_temperature)
        p += 0.5 * self.step_size * new_grad

        return x, p, new_log_prob, new_grad


    def sample(self):
        """
        Hamiltonian Monte Carlo step.
        """

        # Sample a new momentum
        p = torch.randn_like(self.x, device=self.device)

        # Simulate Hamiltonian dynamics
        new_x, new_p, new_log_prob, new_grad = self.leapfrog_integration(p)

        # Hamiltonian (log probability + kinetic energy)
        current_hamiltonian = self.current_log_prob - 0.5 * p.pow(2).sum(-1)
        new_hamiltonian = new_log_prob - 0.5 * new_p.pow(2).sum(-1)
        
        log_accept_rate = -current_hamiltonian + new_hamiltonian
        is_accept = torch.rand_like(log_accept_rate, device=self.device).log() < log_accept_rate
        is_accept = is_accept.unsqueeze(-1)

        self.x = torch.where(is_accept, new_x.detach(), self.x)
        self.current_grad = torch.where(is_accept, new_grad.detach(), self.current_grad)
        self.current_log_prob = torch.where(is_accept.squeeze(-1), new_log_prob.detach(), self.current_log_prob)

        acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
        
        return copy.deepcopy(self.x.detach()), acc_rate.item()

class DenoisingLangevinDynamics(object):
    def __init__(self,
                 x: torch.Tensor,
                 energy_func: callable,
                 gaussian_energy: callable,
                 step_size: float,
                 mh: bool = True,
                 device: str = 'cpu',
                 point_estimator: bool = False):
        """
        Langevin Dynamics Sampler to Sample from $p(x_0|x_t)$.
        This sampler will return both samples and gradient, and we use it to estimate $\nabla\log p(x_t)$ by TSM (MSM).
        """
        super().__init__()

        self.x = x
        self.step_size = step_size
        self.energy_func = energy_func
        self.gaussian_energy = gaussian_energy
        self.mh= mh
        self.device = device
        self.point_estimator = point_estimator

        if self.mh:
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc1 = self.energy_func(x_c)
            grad_xc1 = torch.autograd.grad(f_xc1.sum(), x_c,create_graph=False)[0]
            
            x_c = self.x.detach()
            x_c.requires_grad = True
            f_xc2 = gaussian_energy(x_c)
            grad_xc2 = torch.autograd.grad(f_xc2.sum(), x_c,create_graph=False)[0]

            self.f_x1 = f_xc1.detach()
            self.f_x2 = f_xc2.detach()
            self.grad_x1 = grad_xc1.detach()
            self.grad_x2 = grad_xc2.detach()
            # input(f"self.grad_x1: {self.grad_x1.min()} | {self.grad_x1.max()}")

    def sample(self) -> tuple:

        if self.mh == False:
            raise NotImplementedError
        
        else:
            x_c = self.x.detach()
            f_xc = self.f_x1.detach() + self.f_x2.detach()
            grad_xc = self.grad_x1.detach() + self.grad_x2.detach()
            
            x_p = x_c - self.step_size * grad_xc + torch.sqrt(torch.tensor(2.0*self.step_size, device=self.device)) * torch.randn_like(self.x, device=self.device)
            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp1 = self.energy_func(x_p)
            grad_xp1 = torch.autograd.grad(f_xp1.sum(), x_p,create_graph=False)[0]

            x_p = x_p.detach()
            x_p.requires_grad = True
            f_xp2 = self.gaussian_energy(x_p)
            grad_xp2 = torch.autograd.grad(f_xp2.sum(), x_p, create_graph=False)[0]

            f_xp = f_xp1 + f_xp2
            grad_xp = grad_xp1 + grad_xp2

            log_joint_prob_2 = -f_xc-torch.norm(x_p-x_c+self.step_size * grad_xc, dim=-1)**2/(4*self.step_size)
            log_joint_prob_1 = -f_xp-torch.norm(x_c-x_p+self.step_size * grad_xp, dim=-1)**2/(4*self.step_size)

            log_accept_rate = log_joint_prob_1 - log_joint_prob_2
            is_accept = torch.rand_like(log_accept_rate).log() <= log_accept_rate
            is_accept = is_accept.unsqueeze(-1)

            self.x = torch.where(is_accept, x_p.detach(), self.x)
            self.f_x1 = torch.where(is_accept.squeeze(-1), f_xp1.detach(), self.f_x1)
            self.grad_x1 = torch.where(is_accept, grad_xp1.detach(), self.grad_x1)  
            self.f_x2 = torch.where(is_accept.squeeze(-1), f_xp2.detach(), self.f_x2)
            self.grad_x2 = torch.where(is_accept, grad_xp2.detach(), self.grad_x2) 


            acc_rate = torch.minimum(torch.ones_like(log_accept_rate), log_accept_rate.exp()).mean()
            
            # print(f"DenoisingLangevinDynamics: {self.grad_x1.min()} | {self.grad_x1.max()}")
            # input()
            return copy.deepcopy(self.x.detach()), acc_rate.item(), copy.deepcopy(self.grad_x1.detach())



def AIS(ais_step: int, 
        hmc_step: int, 
        hmc_step_size: float, 
        x_importance_sample: any, 
        proposal_log_p: callable, 
        target_log_p: callable,
        verbose: bool,
        device: str,
        LG: bool = False, # if LG, then run LG as the kernel; otherwise, run HMC as the kernel.
        LG_step: int = 1,
        ):
    final_target_log_p = target_log_p
    initial_proposal_log_p = proposal_log_p

    is_target = lambda x: 1 / ais_step * final_target_log_p(x) + (1 - 1 / ais_step) * initial_proposal_log_p(x)      
    is_weights = is_target(x_importance_sample) - initial_proposal_log_p(x_importance_sample)
    for step in range(ais_step-1):

        s = (step+1) / ais_step
        s_next = (step+2) / ais_step

        # use HMC/LG for pi \propto proposal^(1-s) target^s
        target = lambda x: s * final_target_log_p(x) + (1-s) * initial_proposal_log_p(x) # HMC target
        if not LG:
            hmc = HamiltonianMonteCarlo(x_importance_sample.clone(), energy_func=lambda x: -target(x), step_size=hmc_step_size, num_leapfrog_steps_per_hmc_step=hmc_step, device=device) 
            x_importance_sample, rate = hmc.sample()
            x_importance_sample = x_importance_sample.detach()
            if verbose:
                print('AIS step %.2f'%s, 'HMC Acc rate %.3f'%rate, 'IS sample quantile:', x_importance_sample.min().item(), x_importance_sample.quantile(0.25).item(),  x_importance_sample.quantile(0.75).item(), x_importance_sample.max().item())
        else:
            lg = LangevinDynamics(x_importance_sample.clone(),
                                  energy_func=lambda x: -target(x),
                                  step_size=hmc_step_size,
                                  mh=True,
                                  device=device)
            for _ in range(LG_step):
                x_importance_sample, rate = lg.sample()
                x_importance_sample = x_importance_sample.detach()
            if verbose:
                print('AIS step %.2f'%s, 'LG Acc rate %.3f'%rate, 'IS sample quantile:', x_importance_sample.min().item(), x_importance_sample.quantile(0.25).item(),  x_importance_sample.quantile(0.75).item(), x_importance_sample.max().item())
          
        # calculate the IS weight
        is_target = lambda x: s_next * final_target_log_p(x) + (1-s_next) * initial_proposal_log_p(x)
        is_weights += (is_target(x_importance_sample) - target(x_importance_sample))

        
    return is_weights, x_importance_sample
