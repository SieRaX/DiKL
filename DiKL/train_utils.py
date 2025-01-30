from energy.mog40 import *
from energy.dw import *
from energy.lj import *
from energy.mw import *
from energy.funnel import *
from models.mlp import *
from models.egnn import *

from sampler.ais import LangevinDynamics

def get_target(opt):
    """
    Return target objective.
    """
    if opt.name == 'mog':
        target = GMM_GRID(dim=2, n_mixes=9, log_var_scaling=0.3, device=opt.device)
        target.to(opt.device)
    elif opt.name == 'funnel':
        target = Funnel()
    elif opt.name == 'dw':
        class Target(MultiDoubleWellEnergy):
            def log_prob(self, x, **kwargs):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles * opt.n_dim)
                return super().log_prob(x).squeeze(-1).reshape(*bsz)
            def energy(self, x, **kwargs):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles * opt.n_dim)
                return super().energy(x).squeeze(-1).reshape(*bsz)
        target = Target(opt.n_particles*opt.n_dim,
                        n_particles=opt.n_particles,
                        two_event_dims=False)
    elif opt.name == 'lj':
        class Target(LennardJonesPotential):
            def log_prob(self, x, smooth_=False):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles, opt.n_dim)
                return super().log_prob(x, smooth_=smooth_).squeeze(-1).reshape(*bsz)
            def energy(self, x, smooth_=False):
                assert x.shape[-1] == opt.n_particles * opt.n_dim
                bsz = x.shape[:-1]
                x = x.reshape(-1, opt.n_particles, opt.n_dim)
                return -super().log_prob(x, smooth_=smooth_).squeeze(-1).reshape(*bsz)
        target = Target(opt.n_particles*opt.n_dim,
                        n_particles=opt.n_particles,
                        eps=1.0,
                        rm=1.0,
                        oscillator=True,
                        oscillator_scale=1.0,
                        two_event_dims=True, # always set to True as we wrap it with reshape
                        energy_factor=1.0)
    elif opt.name == 'mw':
        target = ManyWellEnergy(dim=opt.n_dim)
    else:
        raise NotImplementedError
        
    return target

def get_network(opt):
    """
    Return score network and neural sampler
    """
    if opt.name in ['mog', 'mw', 'funnel']:
        lvm = LVM(opt.n_dim, opt.h_dim, opt.n_dim, opt.layer_num, opt.device).to(opt.device)
        score_model= DiffusionModel(
                                    x_dim=opt.n_dim,
                                    num_steps=opt.Tmax, 
                                    layer_num=opt.score_layer_num,
                                    h_dim=opt.score_h_dim,
                                    schedule=opt.dsm_scheme, 
                                    start=opt.start, 
                                    end=opt.end,
                                    device=opt.device,
                                    ).to(opt.device)
    if opt.name in ['dw', 'lj']:
        lvm = EGNN_dynamics(n_particles=opt.n_particles,
                            n_dimension=opt.n_dim,
                            hidden_nf=opt.h_dim,
                            act_fn=torch.nn.ReLU(),
                            n_layers=opt.layer_num,
                            recurrent=True,
                            attention=True,
                            condition_time=False,
                            tanh=True,
                            agg="sum",
                            energy_function=None).to(opt.device)
        score_model = EGNN_dynamics(n_particles=opt.n_particles,
                                    n_dimension=opt.n_dim,
                                    hidden_nf=opt.score_h_dim,
                                    act_fn=torch.nn.ReLU(),
                                    n_layers=opt.score_layer_num,
                                    recurrent=True,
                                    attention=True,
                                    condition_time=True,
                                    tanh=True,
                                    agg="sum",
                                    energy_function=None,
                                    device=opt.device,
                                    schedule=opt.dsm_scheme,
                                    num_steps=opt.Tmax,
                                    start=opt.start, 
                                    end=opt.end).to(opt.device)

    return lvm, score_model

def get_sample(lvm, opt, stop_grad, sample_size=None):
    if opt.e3:
        process = lambda x: remove_mean(x, n_particles=opt.n_particles, n_dimensions=opt.n_dim)
        z_dim = opt.n_particles * opt.n_dim
        sample = lambda x: lvm(None, x)
    else:
        process = lambda x: x
        z_dim = opt.n_dim
        sample = lambda x: lvm(x)
    
    z = process(torch.randn([opt.batch_size if sample_size == None else sample_size, z_dim], device=opt.device))
    if stop_grad:
        with torch.no_grad():
            x = process(sample(z).detach())
    else:
        x = process(sample(z))

    return x

def save_plot_and_check(opt, x_samples, posterior_samples, target, plot_file_name):
    if opt.name == 'mog':
        plot_MoG40(
            log_prob_function=GMM_GRID(dim=2, n_mixes=9, log_var_scaling=0.3, device="cpu").log_prob,
            samples=x_samples, 
            file_name=plot_file_name,
            title=None
            )
    if opt.name == 'mw':
        plot_marginal_paris(target.double_well.log_prob, samples=x_samples, plotting_bounds=(-3, 3), n_contour_levels=40, grid_width_n_points=100, save_dir=plot_file_name)
    if opt.name in ['dw', 'lj']:
        plt.rcParams['figure.figsize'] = [12, 4]
        min, max = (-70, 20) if opt.name == 'lj' else (-30, 0)
        val_data = torch.from_numpy(np.load(opt.val_sample_path))

        plt.subplot(1, 2, 1)
        gt_energy = target.energy(val_data.to(device=opt.device)).detach().cpu().numpy()
        plt.hist(gt_energy, np.linspace(min, max, 100), density=True, alpha=1, histtype='step', label='gt sample')
        model_energy = target.energy(x_samples).detach().cpu().numpy()
        plt.hist(model_energy, np.linspace(min, max, 100), density=True, alpha=1, histtype='step', label='model sample')
        post_energy = target.energy(posterior_samples).detach().cpu().numpy()
        plt.hist(post_energy, np.linspace(min, max, 100), density=True, alpha=1, histtype='step', label='posterior sample')
        plt.xlim(min, max)
        plt.legend()
        
        plt.subplot(1, 2, 2)
        x = (((val_data.reshape(-1, opt.n_particles, 1, opt.n_dim) - val_data.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
        diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
        plt.hist(x[:, diagx[0], diagx[1]].flatten(), 100, density=1, alpha=1, histtype='step', label='gt sample')
        x = (((x_samples.reshape(-1, opt.n_particles, 1, opt.n_dim) - x_samples.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
        diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
        plt.hist(x[:, diagx[0], diagx[1]].flatten(), 100, density=1, alpha=1, histtype='step', label='model sample')
        x = (((posterior_samples.reshape(-1, opt.n_particles, 1, opt.n_dim) - posterior_samples.reshape(-1, 1, opt.n_particles, opt.n_dim))**2).sum(-1).sqrt()).cpu()
        diagx = torch.triu_indices(x.shape[1], x.shape[1], 1)
        plt.hist(x[:, diagx[0], diagx[1]].flatten(), 100, density=1, alpha=1, histtype='step', label='posterior sample')
        plt.legend()

        plt.savefig(plot_file_name)
        plt.close()

    if opt.early_stop:
        if opt.name == 'dw':
            lg = LangevinDynamics(x_samples.clone().detach(), 
                                    target.energy,
                                    step_size=opt.langevin_step_size,
                                    mh=1,
                                    device=opt.device)
            for _ in range(50):
                x_w_lg, acc = lg.sample()
                # if acc > 0.6:
                #     lg.step_size *= 1.5
                # elif acc < 0.5:
                #     lg.step_size /= 1.5
            d = total_variation_distance(target.energy(x_samples).detach().cpu().numpy(), target.energy(x_w_lg).detach().cpu().numpy(), bins=500)
            return d
        if opt.name == 'lj':
            energy = target.energy(x_samples)
            d = energy[energy<300].mean().item()
            return d
    return 0


def total_variation_distance(samples1, samples2, bins=1000):
    min_ = -100
    max_ = 100
    # Create histograms of the two sample sets
    hist1, bins = np.histogram(samples1, bins=bins, range=(min_, max_))
    hist2, _ = np.histogram(samples2, bins=bins, range=(min_, max_))

    if sum(hist1) / samples1.shape[0] < 0.6: #  in case that the samples are outside [min, max]
        return 1e10
    
    # Normalize histograms to get probability distributions
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)
    
    # Calculate the Total Variation distance
    tv_distance = 0.5 * np.sum(np.abs(hist1 - hist2))
    
    return tv_distance