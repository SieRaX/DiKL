import torch
import torch.optim as optim
from torch.distributions import Normal, Gumbel
import numpy as np
import random

import os
import yaml
import argparse
from tqdm import tqdm
from types import SimpleNamespace as config

from train_utils import get_network, get_target, get_sample, save_plot_and_check, total_variation_distance
from models.dm_utils import extract, remove_mean
from loss import dsm_loss, diKL_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Configuration for model training")

    parser.add_argument("--target", type=str, default='mog')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--track_tvd", type=bool, default=False)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.target == 'mog':
        cfg = 'configs/mog.yaml'
    elif args.target == 'mw':
        cfg = 'configs/mw.yaml'
    elif args.target == 'dw':
        cfg = 'configs/dw.yaml'
    elif args.target == 'lj':
        cfg = 'configs/lj.yaml'
    else:
        raise NotImplementedError
    with open(cfg, 'r') as file:
        opt = config(**yaml.safe_load(file))
    opt.ais = config(**opt.ais)

    print("target:", args.target)
    
    # make dir to save results
    opt.device = args.device
    opt.proj_path = opt.save_path + opt.name

    if os.path.exists(opt.proj_path):
        counter = 1
        # Loop until a non-existing path is found
        while os.path.exists(opt.proj_path):
            opt.proj_path = opt.save_path + opt.name + f"_{counter}"
            counter += 1

    os.makedirs(opt.proj_path, exist_ok=True)
    os.makedirs(opt.proj_path+'/model', exist_ok=True)
    os.makedirs(opt.proj_path+'/plot', exist_ok=True)
    os.makedirs(opt.proj_path+'/samples', exist_ok=True)
    target = get_target(opt)

    lvm, score_model = get_network(opt)
    lvm_optim = optim.Adam(lvm.parameters(), lr=opt.lvm_lr)
    score_optim = optim.Adam(score_model.parameters(), lr=opt.score_lr)

    process = (lambda x: remove_mean(x, n_particles=opt.n_particles, n_dimensions=opt.n_dim)) if opt.e3 else (lambda x: x)

    best_metric = 100
    for it in tqdm(range(1,opt.max_iter+1)):
        # train score network
        lvm.eval()
        score_model.train()
        Score_loss = []
        for _ in range(opt.score_iter):
            score_optim.zero_grad()
            x = get_sample(lvm, opt, stop_grad=True)
            # dsm loss
            score_loss = dsm_loss(score_model, x, opt)
            score_loss.backward()
            score_optim.step()
            Score_loss.append(score_loss.item())
        
        # train LVM with diffusive rKL
        lvm.train()
        score_model.eval()
        lvm_optim.zero_grad()
        x = get_sample(lvm, opt, stop_grad=False)

        lvm_loss, posterior_samples = diKL_loss(score_model, x, process, opt, target)
        if ~(torch.isnan(lvm_loss) | torch.isinf(lvm_loss)):
            lvm_loss.backward()

        torch.nn.utils.clip_grad_norm_(lvm.parameters(), opt.grad_norm_clip)
        lvm_optim.step()

        # plot and save checkpoints
        if it % opt.check_iter == 0 or it == 1:
            x_samples = get_sample(lvm, opt, True, opt.eval_samples)
            metric = save_plot_and_check(opt, x_samples, posterior_samples, target, plot_file_name=opt.proj_path + '/plot/%d.png'%it)
            if metric <= best_metric:
                best_metric = metric
                # save ckpt
                torch.save(lvm.state_dict(), opt.proj_path + '/model/' + 'LVM.pt')
                torch.save(score_model.state_dict(), opt.proj_path + '/model/' + 'SCORE.pt')
                torch.save(x_samples, opt.proj_path + '/samples/%d.pt'%it)
                if opt.early_stop:
                    print('Iter %d, '%it, 'Metric %.6f'%metric, flush=True)
            if args.track_tvd:
                val_data = torch.from_numpy(np.load(opt.val_sample_path))
                tvd = total_variation_distance(
                                                target.energy(val_data).detach().cpu().numpy(), 
                                                target.energy(x_samples).detach().cpu().numpy(), 
                                                bins=200 # align with that used in iDEM
                                               )
                with open(opt.proj_path + '/tvd.txt', 'a') as f:
                    f.write(str(tvd) + '\n')
if __name__ == '__main__':
    main()
