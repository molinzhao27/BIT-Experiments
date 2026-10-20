"""Small sequence decoders (1D-CNN, GRU) on the binned feature time series.

M independent models are stacked along a leading axis of every parameter and trained in one
pass (each model has its own weights, Adam state, weight decay and training-trial mask), so a
whole nested-CV grid trains at once on the GPU. `tests/test_seqmodels.py` checks each stacked
model against the equivalent torch.nn.Conv1d / torch.nn.GRU network.

Architectures (deliberately small: 98–210 trials per condition):
  CNN: conv(k=5) F->C, GELU, dropout -> conv(k=5, dilation 2) C->C, GELU, dropout
       -> [mean, max] over time -> linear -> 7 words
  GRU: linear F->P, dropout -> GRU(P->H) -> [last state, mean state] -> dropout -> linear -> 7 words
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def _init(shape, fan_in, device):
    bound = 1 / math.sqrt(fan_in)
    return ((torch.rand(shape, device=device) * 2 - 1) * bound).requires_grad_(True)


class StackedCNN:
    def __init__(self, M: int, n_feat: int, n_classes: int, channels: int = 16, kernel: int = 5,
                 kernel2: int = 5, dilation2: int = 2, dropout: float = 0.3, device="cpu"):
        C, k1, k2 = channels, kernel, kernel2
        self.M, self.C, self.k1, self.d2, self.p = M, C, k1, dilation2, dropout
        self.params = {
            "W1": _init((M, n_feat * k1, C), n_feat * k1, device), "b1": _init((M, 1, 1, C), n_feat * k1, device),
            "W2": _init((M, C, C, k2), C * k2, device), "b2": _init((M, C), C * k2, device),
            "Wo": _init((M, 2 * C, n_classes), 2 * C, device), "bo": _init((M, 1, n_classes), 2 * C, device),
        }

    def __call__(self, X: torch.Tensor, train: bool) -> torch.Tensor:
        """X (n, T, F) shared by all models -> logits (M, n, K)."""
        P, M, C = self.params, self.M, self.C
        n = X.shape[0]
        U = X.unfold(1, self.k1, 1)  # (n, T1, F, k1)
        U = U.reshape(n, U.shape[1], -1)
        h = F.gelu(torch.einsum("ntd,mdc->mntc", U, P["W1"]) + P["b1"])  # (M, n, T1, C)
        h = F.dropout(h, self.p, train)
        h = h.permute(1, 0, 3, 2).reshape(n, M * C, -1)
        h = F.conv1d(h, P["W2"].reshape(M * C, C, -1), P["b2"].reshape(-1), dilation=self.d2, groups=M)
        h = F.dropout(F.gelu(h), self.p, train).view(n, M, C, -1)
        feat = torch.cat([h.mean(-1), h.amax(-1)], -1)  # (n, M, 2C)
        return torch.einsum("nmc,mck->mnk", feat, P["Wo"]) + P["bo"]


class StackedGRU:
    def __init__(self, M: int, n_feat: int, n_classes: int, proj: int = 32, hidden: int = 32,
                 dropout: float = 0.3, device="cpu"):
        H, Pd = hidden, proj
        self.M, self.H, self.p = M, H, dropout
        self.params = {
            "Wp": _init((M, n_feat, Pd), n_feat, device), "bp": _init((M, 1, 1, Pd), n_feat, device),
            "Wih": _init((M, Pd, 3 * H), H, device), "bih": _init((M, 1, 1, 3 * H), H, device),
            "Whh": _init((M, H, 3 * H), H, device), "bhh": _init((M, 1, 3 * H), H, device),
            "Wo": _init((M, 2 * H, n_classes), 2 * H, device), "bo": _init((M, 1, n_classes), 2 * H, device),
        }

    def __call__(self, X: torch.Tensor, train: bool) -> torch.Tensor:
        P, M, H = self.params, self.M, self.H
        n, T, _ = X.shape
        z = F.dropout(torch.einsum("ntf,mfp->mntp", X, P["Wp"]) + P["bp"], self.p, train)
        gi = torch.einsum("mntp,mpg->mntg", z, P["Wih"]) + P["bih"]  # (M, n, T, 3H), gate order r, z, n
        h = X.new_zeros(M, n, H)
        hsum = X.new_zeros(M, n, H)
        for t in range(T):
            gh = torch.baddbmm(P["bhh"], h, P["Whh"])
            i_r, i_z, i_n = gi[:, :, t].chunk(3, -1)
            h_r, h_z, h_n = gh.chunk(3, -1)
            r, u = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
            h = (1 - u) * torch.tanh(i_n + r * h_n) + u * h
            hsum = hsum + h
        feat = F.dropout(torch.cat([h, hsum / T], -1), self.p, train)
        return torch.bmm(feat, P["Wo"]) + P["bo"]


ARCHS = {"cnn": StackedCNN, "gru": StackedGRU}


def train_eval(arch: str, arch_kw: dict, X_ext: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor,
               weight_decay: torch.Tensor, checkpoints: list[int], seed: int, lr: float = 3e-3,
               jitter: int = 3, noise: float = 0.2, n_classes: int = 7) -> np.ndarray:
    """Train M stacked models full-batch with Adam(W); return eval-mode logits on ALL trials at
    each checkpoint epoch: (n_ckpt, M, n, K).

    X_ext (n, T + 2*jitter, F): each step uses one random time shift in [0, 2*jitter] bins plus
    Gaussian noise; evaluation uses the centered crop. train_mask (M, n) selects each model's
    training trials; weight_decay (M,) is decoupled and applied to weights (not biases).
    """
    torch.manual_seed(seed)
    M = train_mask.shape[0]
    T = X_ext.shape[1] - 2 * jitter
    model = ARCHS[arch](M, X_ext.shape[2], n_classes, device=X_ext.device, **arch_kw)
    names, params = zip(*model.params.items())
    m1 = [torch.zeros_like(p) for p in params]
    m2 = [torch.zeros_like(p) for p in params]
    wds = [weight_decay.view(M, *[1] * (p.dim() - 1)) if nm.startswith("W") else None for nm, p in zip(names, params)]
    denom = train_mask.sum(1).clamp_min(1)
    yy = y.repeat(M)
    b1, b2, eps = 0.9, 0.999, 1e-8
    out, ckpts = [], set(checkpoints)
    for step in range(1, max(checkpoints) + 1):
        off = int(torch.randint(0, 2 * jitter + 1, (1,)))
        Xb = X_ext[:, off:off + T]
        Xb = Xb + noise * torch.randn_like(Xb)
        logits = model(Xb, train=True)
        ce = F.cross_entropy(logits.reshape(-1, n_classes), yy, reduction="none").view(M, -1)
        loss = ((ce * train_mask).sum(1) / denom).sum()
        grads = torch.autograd.grad(loss, params)
        with torch.no_grad():
            for p, g, a, v, wd in zip(params, grads, m1, m2, wds):
                a.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if wd is not None:
                    p.mul_(1 - lr * wd)
                p.addcdiv_(a / (1 - b1**step), (v / (1 - b2**step)).sqrt_().add_(eps), value=-lr)
        if step in ckpts:
            with torch.no_grad():
                out.append(model(X_ext[:, jitter:jitter + T], train=False).float().cpu())
    return torch.stack(out).numpy()
