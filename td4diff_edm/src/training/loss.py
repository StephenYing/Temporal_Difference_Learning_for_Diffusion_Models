# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
from typing import Optional

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

# ----------------------------------------------------------------------------
# TD for EDM in index space (i=0 -> sigma_max, i=N-1 -> sigma_min).
# Supports Markov or shared-noise coupling and optional EDM-style TD weighting.

@persistence.persistent_class
class TDLossEDMIndexed:
    """
    Temporal-Difference loss in EDM index space with *forward* (plus) indexing:
        i_prev  = i_t + one_idx
        i_prime = i_t + k_idx
        i_p_prev= i_t + k_idx + one_idx

    Args
    ----
    P_mean, P_std, sigma_data : EDM training-time sigma distribution and data variance.
    lambda_edm : Weight of the EDM reconstruction term in the mixed objective.
    sigma_min, sigma_max, rho : EDM sampling schedule parameters used by sigma(i).
    steps : Number of sampling steps N.
    k_idx : k-step span in continuous EDM index space.
    one_idx : one-step span; defaults to k_idx / 3 when omitted.
    td_coupling : 'markov' or 'shared', controlling kappa and paired sampling.
    weight_td : Apply EDM-style weighting to the TD term.
    """

    def __init__(self,
                 P_mean: float = -1.2,
                 P_std: float = 1.2,
                 sigma_data: float = 0.5,
                 lambda_edm: float = 1.0,
                 sigma_min: float = 2e-3,
                 sigma_max: float = 80.0,
                 rho: float = 7.0,
                 steps: int = 18,  # Number of sampling steps
                 k_idx: float = 1.0,
                 one_idx: Optional[float] = None,
                 td_coupling: str = 'markov',   # 'markov' or 'shared'
                 weight_td: bool = False):
        # EDM train-time sigma sampling
        self.P_mean = float(P_mean)
        self.P_std = float(P_std)
        self.sigma_data = float(sigma_data)
        self.lambda_edm = float(lambda_edm)

        # EDM sampler schedule (Table 1 in the paper)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.rho = float(rho)
        self.steps = int(steps)

        # steps in index space
        self.k_idx = float(k_idx)
        self.one_idx = float(k_idx/3.0) if one_idx is None else float(one_idx)

        # TD coupling mode and weighting
        assert td_coupling in ['markov', 'shared']
        self.td_coupling = td_coupling
        self.weight_td = bool(weight_td)

        # Precompute schedule constants.
        self._smax = self.sigma_max ** (1.0 / self.rho)
        self._smin = self.sigma_min ** (1.0 / self.rho)
        self._den  = (self._smin - self._smax)
        self._i_max = float(self.steps - 1)
        # Largest starting index that leaves room for k_idx + one_idx.
        self._i_start_max = self._i_max - (self.k_idx + self.one_idx)

    # ---------- EDM schedule: index <-> sigma ----------
    def index_to_sigma(self, i):
        # Supports tensor-valued continuous indices.
        s = self._smax + (i / self._i_max) * self._den
        return torch.clamp(s, min=min(self._smax, self._smin), max=max(self._smax, self._smin)).pow(self.rho)

    def sigma_to_index(self, sigma):
        # Defined only on [sigma_min, sigma_max].
        s = torch.clamp(sigma, min=self.sigma_min, max=self.sigma_max).pow(1.0 / self.rho)
        i = (s - self._smax) / self._den * self._i_max
        return i

    # ---------- Main entry point ----------
    def __call__(self, net, target_net, images, labels=None, augment_pipe=None):
        device = images.device
        batch = images.shape[0]

        # Clean data and optional augmentation labels.
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        x0 = y

        # EDM log-normal training-time sigma.
        z = torch.randn([batch, 1, 1, 1], device=device)
        sigma_t = (self.P_mean + self.P_std * z).exp()                 # (B,1,1,1)

        # Continuous EDM index corresponding to sigma_t.
        i_t = self.sigma_to_index(sigma_t)                              # (B,1,1,1)
        # TD is valid only inside the schedule with room for k_idx + one_idx.
        td_valid = (
            (sigma_t >= self.sigma_min) & (sigma_t <= self.sigma_max) &
            (i_t >= 0) & (i_t <= self._i_start_max)
        ).squeeze()

        # Forward indices moving toward the data endpoint.
        i_prev   = i_t + self.one_idx
        i_prime  = i_t + self.k_idx
        i_p_prev = i_t + self.k_idx + self.one_idx

        # Convert indices back to sigmas.
        sigma_prev   = self.index_to_sigma(i_prev)
        sigma_prime  = self.index_to_sigma(i_prime)
        sigma_p_prev = self.index_to_sigma(i_p_prev)

        # Generate paired states (x_t, x_tprime).
        if self.td_coupling == 'shared':
            # Shared-noise coupling.
            n = torch.randn_like(x0)
            x_t      = x0 + n * sigma_t
            x_tprime = x0 + n * sigma_prime
        else:
            # Markov coupling: sample x_tprime first, then add independent noise to x_t.
            n_prime  = torch.randn_like(x0) * sigma_prime
            x_tprime = x0 + n_prime
            add_var  = (sigma_t**2 - sigma_prime**2).clamp(min=0)
            n_add    = torch.randn_like(x0) * add_var.sqrt()
            x_t      = x_tprime + n_add

        # EDM reconstruction loss at t.
        w_edm = (sigma_t**2 + self.sigma_data**2) / (sigma_t * self.sigma_data) ** 2
        x0_pred_t = net(x_t, sigma_t.squeeze(), labels, augment_labels=augment_labels)
        recon_loss_per = (w_edm * (x0_pred_t - x0).pow(2)).mean(dim=[1, 2, 3])    # (B,)

        # Compute TD only on valid samples; otherwise fall back to (1 + lambda) * EDM.
        losses = torch.zeros(batch, device=device)
        not_app = ~td_valid
        if not_app.any():
            losses[not_app] = (1.0 + self.lambda_edm) * recon_loss_per[not_app]

        if td_valid.any():
            # Posterior coefficients kappa and A = 1 - kappa.
            if self.td_coupling == 'shared':
                # Shared-noise coupling.
                kappa_t   = (sigma_prev   / sigma_t)
                kappa_tp  = (sigma_p_prev / sigma_prime)
            else:
                # Markov coupling.
                kappa_t   = (sigma_prev   / sigma_t)    ** 2
                kappa_tp  = (sigma_p_prev / sigma_prime)** 2

            A_t  = 1.0 - kappa_t
            A_tp = 1.0 - kappa_tp

            # Network predictions.
            x0_pred_t_v = x0_pred_t[td_valid]
            with torch.no_grad():
                x0_pred_tp_v = target_net(x_tprime[td_valid],
                                          sigma_prime[td_valid].squeeze(),
                                          labels[td_valid] if labels is not None else None,
                                          augment_labels=None if augment_labels is None else augment_labels[td_valid])

            # True one-step posterior means.
            mu_true_t  = (A_t  * x0 + kappa_t  * x_t     )[td_valid]
            mu_true_tp = (A_tp * x0 + kappa_tp * x_tprime)[td_valid]

            # Model one-step posterior means.
            mu_theta_t  = (A_t  * x0_pred_t + kappa_t  * x_t     )[td_valid]
            mu_theta_tp = (A_tp[td_valid] * x0_pred_tp_v + kappa_tp[td_valid] * x_tprime[td_valid])

            # TD residual.
            td_err = (mu_true_t - mu_true_tp) - (mu_theta_t - mu_theta_tp)

            # Optional EDM-style gradient normalization for the TD term.
            if self.weight_td:
                c_out_t  = (sigma_t  * self.sigma_data) / (sigma_t**2  + self.sigma_data**2).sqrt()
                c_out_tp = (sigma_prime * self.sigma_data) / (sigma_prime**2 + self.sigma_data**2).sqrt()
                # w_td_den = (A_t**2  * c_out_t**2) + (A_tp**2 * c_out_tp**2)
                w_td_den = (A_t**2  * c_out_t**2)
                w_td = (1.0 / w_td_den.clamp(min=1e-8))[td_valid].squeeze()
                td_loss_per = w_td * (td_err.pow(2)).mean(dim=[1, 2, 3])
            else:
                td_loss_per = (td_err.pow(2)).mean(dim=[1, 2, 3])

            losses[td_valid] = td_loss_per + self.lambda_edm * recon_loss_per[td_valid]

        return losses.mean()
