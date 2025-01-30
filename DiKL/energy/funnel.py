from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Callable

import torch
import torchquad
from torch import distributions
from torch.nn.init import trunc_normal_

from numbers import Number

EXPECTATION_FNS: dict[str, Callable] = {
    "square": lambda x: (x**2).sum(dim=-1, keepdims=True),
    "abs": lambda x: x.abs().sum(dim=-1, keepdims=True),
    "sum": lambda x: x.sum(dim=-1, keepdims=True),
    "square_minus_sum": lambda x: (x**2 - x).sum(dim=-1, keepdims=True),
}
DATA_DIR = Path(__file__).parents[2] / "data"

def gmm_params(name: str = "heart", dim: int = 2):
    if name == "heart":
        loc = 1.5 * torch.tensor(
            [
                [-0.5, -0.25],
                [0.0, -1],
                [0.5, -0.25],
                [-1.0, 0.5],
                [-0.5, 1.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
            ]
        )
        factor = 1 / len(loc)

    elif name == "dist":
        loc = torch.tensor(
            [
                [0.0, 0.0],
                [2, 0.0],
                [0.0, 3.0],
                [-4, 0.0],
                [0.0, -5],
            ]
        )
        factor = math.sqrt(0.2)

    elif name in ["fab", "multi"]:
        n_mixes, loc_scaling = (40, 40) if name == "fab" else (80, 80)
        generator = torch.Generator()
        generator.manual_seed(42)
        loc = (torch.rand((n_mixes, 2), generator=generator) - 0.5) * 2 * loc_scaling
        factor = torch.nn.functional.softplus(torch.tensor(1.0, device=loc.device))
    elif name == "grid":
        x_coords = torch.linspace(-5, 5, 3)
        loc = torch.cartesian_prod(x_coords, x_coords)
        factor = math.sqrt(0.3)
    elif name == "circle":
        freq = 2 * torch.pi * torch.arange(1, 9) / 8
        loc = torch.stack([4.0 * freq.cos(), 4.0 * freq.sin()], dim=1)
        factor = math.sqrt(0.3)
    else:
        raise ValueError("Unknown mode for the Gaussian mixture.")

    if dim > 2:
        loc = torch.cat([loc, torch.zeros(8, dim - 2)], dim=1)
    scale = factor * torch.ones_like(loc)
    mixture_weights = torch.ones(loc.shape[0], device=loc.device)
    return loc, scale, mixture_weights

class Distribution(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        log_norm_const: float = None,
        domain: float | torch.Tensor | None = None,
        n_reference_samples: int | None = None,
        grid_points: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_reference_samples = n_reference_samples
        self.grid_points = grid_points
        self.set_domain(domain)

        # Initialize
        self.log_norm_const = log_norm_const
        self.register_buffer("stddevs", None, persistent=False)
        self.expectations = {}

    def set_domain(self, d: torch.Tensor | float | None = None):
        if d is not None:
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d, dtype=torch.float)
            if d.ndim == 0:
                d = torch.stack([-d, d], dim=-1)
            if d.ndim == 1:
                d = d.unsqueeze(0)
            if d.shape == (1, 2):
                d = d.repeat(self.dim, 1)
            assert d.shape == (self.dim, 2)
        self.register_buffer("domain", d, persistent=False)

    def compute_stats_sampling(self):
        samples = self.sample((self.n_reference_samples,))
        for name, fn in EXPECTATION_FNS.items():
            if name not in self.expectations:
                self.expectations[name] = fn(samples).mean().item()
        if self.stddevs is None:
            self.stddevs = samples.std(dim=0)

    def compute_stats_integration(self):
        integrate = partial(
            torchquad.Boole().integrate,
            dim=self.dim,
            N=self.grid_points,
            integration_domain=self.domain,
        )

        if self.log_norm_const is None:
            norm_const = integrate(self.unnorm_pdf).item()
            self.log_norm_const = math.log(norm_const)

        for name, fn in EXPECTATION_FNS.items():
            if name not in self.expectations:
                self.expectations[name] = integrate(
                    lambda x: fn(x) * self.pdf(x)
                ).item()

            if self.stddevs is None:
                expectations = integrate(lambda x: x * self.pdf(x)).unsqueeze(0)
                stddevs = integrate(
                    lambda x: (x - expectations) ** 2 * self.pdf(x)
                ).sqrt()
                self.stddevs = torch.atleast_1d(stddevs)

    @torch.no_grad()
    def compute_stats(self):
        if hasattr(self, "sample") and self.n_reference_samples is not None:
            self.compute_stats_sampling()

        elif self.grid_points is not None and self.domain is not None:
            try:
                with torch.device(self.domain.device):
                    self.compute_stats_integration()
            # the `torch.device` context is not available for PyTorch < 2.0
            except AttributeError:
                device = self.domain.device
                self.to("cpu")
                self.compute_stats_integration()
                self.to(device)
        else:
            logging.warning(
                f"Cannot compute statistics for distribution `%s`",
                self.__class__.__name__,
            )

    def _initialize_distr(self):
        # This can be used to reinitialize distributions, e.g,  when transfering between devices
        pass

    def _apply(self, fn):
        torch.nn.Module._apply(self, fn)
        self._initialize_distr()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        if self.log_norm_const is None:
            #raise NotImplementedError
            return self.unnorm_log_prob(x)
        return self.unnorm_log_prob(x) - self.log_norm_const

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.log_prob(x).exp()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unnorm_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x).exp()

    def score(self, x: torch.Tensor, create_graph=False) -> torch.Tensor:
        grad = x.requires_grad
        x.requires_grad_(True)
        with torch.set_grad_enabled(True):
            log_rho = self.unnorm_log_prob(x).sum()
            score = torch.autograd.grad(log_rho, x, create_graph=create_graph)[0]
        x.requires_grad_(grad)
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x)

    # def objective(self, x: torch.Tensor) -> torch.Tensor:
    #    Can be implemented for usage as optimization method

    # def marginal(self, x: np.ndarray | float, dim: int = 0) -> np.ndarray:
    #    Can be implemented for additional metrics

    # def sample(self, shape: tuple | None = None) -> torch.Tensor:
    #     Can be implemented for additional metrics

    # def filter(self, x: torch.Tensor) -> torch.Tensor:
    #     Can be implemented to filter samples

    # def metrics(self, samples: torch.Tensor, *args, **kwargs) -> dict[str, float]
    #     Can be implemented for additional metrics

    # def plots(self, samples: torch.Tensor, *args, **kwargs) -> dict[str, Union[go.Figure, plt.Figure]]
    #     Can be implemented for additional plots


def sample_uniform(domain: torch.Tensor, batchsize: int = 1) -> torch.Tensor:
    dim = domain.shape[0]
    diam = domain[:, 1] - domain[:, 0]
    rand = torch.rand(batchsize, dim, device=domain.device)
    return domain[:, 0] + rand * diam


def rejection_sampling(
    shape: tuple, proposal: Distribution, target: Distribution, scaling: float
) -> torch.Tensor:
    n_samples = math.prod(shape)
    samples = proposal.sample((n_samples * math.ceil(scaling) * 10,))
    unif = torch.rand(samples.shape[0], 1, device=samples.device)
    unif *= scaling * proposal.pdf(samples)
    accept = unif < target.pdf(samples)
    samples = samples[accept]
    if samples.shape[0] >= n_samples:
        return samples[:n_samples].reshape(*shape, -1)
    else:
        new_shape = (n_samples - samples.shape[0],)
        new_samples = rejection_sampling(new_shape, proposal, target, scaling)
        return torch.concat([samples.reshape(*shape, -1), new_samples])

class GMM(Distribution):
    def __init__(
        self,
        dim: int = 2,
        loc: torch.Tensor | None = None,
        scale: torch.Tensor | None = None,
        mixture_weights: torch.Tensor | None = None,
        n_reference_samples: int = int(1e7),
        name: str | None = None,
        log_norm_const: float = 0.0,
        domain_scale: float = 5,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        if name is not None:
            if any(t is not None for t in [loc, scale, mixture_weights]):
                logging.warning(
                    "Ignoring loc, scale, and mixture weights since name is specified."
                )
            loc, scale, mixture_weights = gmm_params(name, dim=dim)

        # Check shapes
        n_mixtures = loc.shape[0]
        if not loc.shape == scale.shape == (n_mixtures, self.dim):
            raise ValueError("Shape missmatch between loc and scale.")
        if mixture_weights is None and n_mixtures > 1:
            raise ValueError("Require mixture weights.")
        if not (mixture_weights is None or mixture_weights.shape == (n_mixtures,)):
            raise ValueError("Shape missmatch for the mixture weights.")

        # Initialize
        self.register_buffer("loc", loc, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("mixture_weights", mixture_weights, persistent=False)
        self._initialize_distr()

        # Check domain
        if self.domain is None:
            deviation = domain_scale * self.scale.max(dim=0).values
            deviation = torch.stack([-deviation, deviation], dim=-1)
            pos = torch.stack(
                [self.loc.min(dim=0).values, self.loc.max(dim=0).values], dim=-1
            )
            self.set_domain(pos + deviation)
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("domain does not satisfy tolerance at the boundary.")

    @property
    def stddevs(self) -> torch.Tensor:
        return self.distr.variance.sqrt()

    def _initialize_distr(
        self,
    ) -> distributions.MixtureSameFamily | distributions.Independent:
        if self.mixture_weights is None:
            self.distr = distributions.Independent(
                distributions.Normal(self.loc.squeeze(0), self.scale.squeeze(0)), 1
            )
        else:
            modes = distributions.Independent(
                distributions.Normal(self.loc, self.scale), 1
            )
            mix = distributions.Categorical(self.mixture_weights)
            self.distr = distributions.MixtureSameFamily(mix, modes)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        log_prob = self.distr.log_prob(x).unsqueeze(-1) + self.log_norm_const
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        if self.mixture_weights is None:
            return distributions.Normal(self.loc[0, dim], self.scale[0, dim])
        modes = distributions.Normal(self.loc[:, dim], self.scale[:, dim])
        mix = distributions.Categorical(self.mixture_weights)
        return distributions.MixtureSameFamily(mix, modes)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        return self.marginal_distr(dim=dim).log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))

class Gauss(GMM):
    def __init__(
        self,
        dim: int = 1,
        loc: torch.Tensor | Number = 0.0,
        scale: torch.Tensor | Number = 1.0,
        **kwargs,
    ):

        # Setup parameters
        params = {"loc": loc, "scale": scale}
        params = {k: Gauss._prepare_input(p, dim) for k, p in params.items()}
        super().__init__(dim=dim, **params, **kwargs)
        self.stddevs = self.scale.squeeze(0)

    @staticmethod
    def _prepare_input(param: torch.Tensor | Number, dim: int = 1):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float)
        param = torch.atleast_2d(param)
        if param.numel() == 1:
            param = param.repeat(1, dim)
        return param

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc - x) / self.scale**2

class IsotropicGauss(Gauss):
    # Typially used as prior (supports truncation and faster methods)
    def __init__(
        self,
        dim: int = 1,
        loc: float = 0.0,
        scale: float = 1.0,
        truncate_quartile: float | None = None,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            loc=loc,
            scale=scale,
            **kwargs,
        )

        assert torch.allclose(self.loc, self.loc[0, 0])
        assert torch.allclose(self.scale, self.scale[0, 0])

        # Calculate truncation values
        if truncate_quartile is not None:
            quartiles = torch.tensor(
                [truncate_quartile / 2, 1 - truncate_quartile / 2],
                device=self.domain.device,
            )
            truncate_quartile = self.marginal_distr().icdf(quartiles).tolist()
        self.truncate_quartile = truncate_quartile

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        var = self.scale[0, 0] ** 2
        norm_const = -0.5 * self.dim * (2.0 * math.pi * var).log()
        norm_const += self.log_norm_const
        sq_sum = torch.sum((x - self.loc[0, 0]) ** 2, dim=-1, keepdim=True)
        return norm_const - 0.5 * sq_sum / var

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return (self.loc[0, 0] - x) / self.scale[0, 0] ** 2

    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.marginal_distr().log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        if self.truncate_quartile is None:
            return self.loc[0, 0] + self.scale[0, 0] * torch.randn(
                *shape, self.dim, device=self.domain.device
            )
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(
            tensor,
            mean=self.loc[0, 0],
            std=self.scale[0, 0],
            a=self.truncate_quartile[0],
            b=self.truncate_quartile[1],
        )


class Funnel(Distribution):
    def __init__(
        self,
        dim: int = 10,
        variance: float | None = None,
        n_reference_samples: int = int(1e7),
        log_norm_const: float = 0.0,
        domain_first_scale: float = 5.0,
        domain_other_scale: float = 5.0,
        domain_tol: float | None = 1e-5,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        self.variance = variance
        if self.variance is None:
            self.variance = self.dim - 1
        self.distr_first = IsotropicGauss(
            dim=1,
            scale=math.sqrt(self.variance),
            domain_scale=domain_first_scale,
            domain_tol=domain_tol,
        )
        self._initialize_distr()

        # Check domains
        if self.domain is None:
            domain_other = (
                self.distr_first.domain.sgn()
                * (self.distr_first.domain.abs() / domain_other_scale).exp()
            )
            self.set_domain(
                torch.cat(
                    [self.distr_first.domain, domain_other.repeat(self.dim - 1, 1)]
                )
            )
        if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
            raise ValueError("Domain does not satisfy tolerance at the boundary.")

    @staticmethod
    def log_prob_other(x_other, x_first):
        norm_const = -x_other.shape[-1] * (x_first + math.log(2.0 * math.pi)) / 2.0
        x_sq_sum = (x_other**2).sum(dim=-1, keepdim=True)

        res = norm_const - 0.5 * x_sq_sum * (-x_first).exp()

        # if torch.isnan(res).any() or torch.isnan(x_other).any() or torch.isnan(x_first).any() or torch.isnan(norm_const).any() or torch.isnan(x_sq_sum).any():
        #     print(f"+"*20)
        #     print(f"x_other has nan: {torch.isnan(x_other).any()}")
        #     print(f"x_first has nan: {torch.isnan(x_first).any()}")
        #     print(f"norm_const has nan: {torch.isnan(norm_const).any()}")
        #     print(f"x_sq_sum has nan: {torch.isnan(x_sq_sum).any()}")
        #     print(f"res has nan: {torch.isnan(res).any()}")

        #     print(f"x_other: {x_other.min()} | {x_other.max()}")    
        #     print(f"x_first: {x_first.min()} | {x_first.max()}")
        #     print(f"norm_const: {norm_const.min()} | {norm_const.max()}")
        #     print(f"x_sq_sum: {x_sq_sum.min()} | {x_sq_sum.max()}")
        #     print(f"res: {res.min()} | {res.max()}")

            # input()
        return norm_const - 0.5 * x_sq_sum * (-x_first).exp()

    def _initialize_distr(self):
        self.distr_first._initialize_distr()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        x_first = x[:, 0].unsqueeze(-1)
        log_prob_first = self.distr_first.unnorm_log_prob(x_first)
        log_prob_other = Funnel.log_prob_other(x[:, 1:], x_first)
        # print(f"x_first has nan: {torch.isnan(x_first).any()}")
        # print(f"log_prob_first has nan: {torch.isnan(log_prob_first).any()}")
        # print(f"log_prob_other has nan: {torch.isnan(log_prob_other).any()}")
        assert log_prob_other.shape == log_prob_first.shape == (x.shape[0], 1)
        log_prob = log_prob_first + log_prob_other
        return log_prob + self.log_norm_const

    def log_prob(self, x: torch.Tensor, **kwargs):
        org_shape = x.shape
        log_prob = self.unnorm_log_prob(x.reshape(-1, x.shape[-1])).reshape(*org_shape[:-1])
        # mask = torch.zeros_like(log_prob)
        # mask[log_prob < -1e4] = - torch.tensor(float("inf"), dtype=log_prob.dtype)

        # nan_mask = torch.zeros_like(log_prob)
        # nan_mask[log_prob < -1e4] = - torch.tensor(float("inf"), dtype=log_prob.dtype)
        # log_prob = log_prob + mask + nan_mask
        log_prob = torch.clamp(log_prob, max=1e4, min=-1e3)
        # print(f"log_prob: {log_prob.min()} | {log_prob.max()}")
        return log_prob

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == self.dim
        x_first = x[:, 0].unsqueeze(-1)
        x_other = x[:, 1:]
        inv_var_other = (-x_first).exp()
        score_first = self.distr_first.score(x_first) - 0.5 * x_other.shape[-1]
        score_first += 0.5 * (x_other**2).sum(dim=-1, keepdim=True) * inv_var_other
        assert score_first.shape == (x.shape[0], 1)
        score_other = -x_other * inv_var_other
        return torch.cat([score_first, score_other], dim=-1)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        assert x.ndim == 2 and x.shape[-1] == 1
        if dim == 0:
            return self.distr_first.marginal(x)
        samples_first = self.distr_first.sample((self.n_reference_samples, 1))
        log_prob = self.log_prob_other(x, samples_first)
        return log_prob.exp().mean(axis=0)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = tuple()
        samples_first = self.distr_first.sample(shape)
        stdd_other = (0.5 * samples_first).exp()
        samples_other = torch.randn(*shape, self.dim - 1, device=samples_first.device)
        return torch.cat((samples_first, samples_other * stdd_other), dim=-1)
