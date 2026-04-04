import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function
from typing import Tuple

# VERY IMPORTANT: all geometries assume p ∈ Δ (simplex)

class Sparsemax(Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:
        z_sorted = z.sort(-1, descending=True).values
        S1 = z_sorted.cumsum(-1)
        k = torch.arange(1, z_sorted.shape[-1] + 1, device=z.device, dtype=z.dtype)

        tau = (S1 - 1) / k
        k_hat = (z_sorted > tau).sum(-1, keepdim=True)
        tau_star = tau.gather(-1, k_hat - 1)
        p = F.relu(z - tau_star)
        ctx.save_for_backward(k_hat, p)
        return p
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        k_hat, p = ctx.saved_tensors
        mask = (p > 0)
        gp2 = grad_out * mask
        v_hat = gp2.sum(-1, keepdim=True) / k_hat.to(p.dtype)
        return (gp2 - v_hat) * mask
    
class Entmax15(Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:
        z = z * 0.5
        z_sorted = z.sort(-1, descending=True).values
        S1 = z_sorted.cumsum(-1)
        S2 = (z_sorted * z_sorted).cumsum(-1)
        k = torch.arange(1, z_sorted.shape[-1] + 1, device=z.device, dtype=z.dtype)

        mu = S1 / k
        delta = (1 - S2 + S1 * mu) / k
        tau = mu - F.relu(delta).sqrt()

        k_hat = (z_sorted > tau).sum(-1, keepdim=True)
        tau_star = tau.gather(-1, k_hat - 1)
        p12 = F.relu(z - tau_star)
        ctx.save_for_backward(p12)
        return p12 * p12
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        p12, = ctx.saved_tensors
        gp12 = grad_out * p12
        q = gp12.sum(-1, keepdim=True) / p12.sum(-1, keepdim=True)
        return gp12 - p12 * q
    
class Entmax133(Function):
    @staticmethod
    def forward(ctx, z: Tensor) -> Tensor:
        z = z * (1 / 3)
        z_sorted = z.sort(-1, descending=True).values
        z_sorted_sq = z_sorted * z_sorted
        S1 = z_sorted.cumsum(-1)
        S2 = z_sorted_sq.cumsum(-1)
        S3 = (z_sorted_sq * z_sorted).cumsum(-1)
        k = torch.arange(1, z_sorted.shape[-1] + 1, device=z.device, dtype=z.dtype)

        mu = S1 / k 
        M2 = S2 - mu * S1
        M3 = S3 - mu * (S2 + 2 * M2)
        q2 = 0.5 * (M3 - 1) / k # q / 2
        p3 = M2 / k # p / 3
        sd = (q2 * q2 + p3 * p3 * p3).sqrt() # sqrt(D)
        u = torch.pow(sd + q2, 1 / 3) # always >= 0
        v = torch.pow(sd - q2, 1 / 3) # always >= 0
        tau = u - v + mu

        support = z_sorted > tau
        k_hat = support.sum(-1, keepdim=True)
        tau_star = tau.gather(-1, k_hat - 1)
        p13 = F.relu(z - tau_star)
        p23 = p13 * p13
        ctx.save_for_backward(p23)
        return p23 * p13
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tensor:
        p23, = ctx.saved_tensors
        gp12 = grad_out * p23
        q = gp12.sum(-1, keepdim=True) / p23.sum(-1, keepdim=True)
        return gp12 - p23 * q
   
class Entmax(Function):
    @staticmethod
    def forward(ctx, z: Tensor, a: float=1.0, T: int=20) -> Tensor:
        ctx.a = a
        d = z.shape[-1]
        z = (a - 1) * z
        z_max = z.amax(-1, keepdim=True)
        tau_min = z_max - 1
        tau_max = z_max - 1 / (d ** (a - 1))

        for _ in range(T):
            tau_m = (tau_min + tau_max) * 0.5
            p_m = F.relu(z - tau_m).pow(1 / (a - 1))
            f_m = p_m.sum(-1, keepdim=True) - 1
            mask = (f_m >= 0)
            tau_min = torch.where(mask, tau_m, tau_min)
            tau_max = torch.where(mask, tau_max, tau_m)

        tau_m = (tau_min + tau_max) * 0.5
        p_m = F.relu(z - tau_m).pow(1 / (a - 1))
        ctx.save_for_backward(p_m)
        return p_m
    
    @staticmethod
    def backward(ctx, grad_out: Tensor) -> Tuple[Tensor, None, None]:
        p, = ctx.saved_tensors
        p2a = p.pow(2 - ctx.a)
        gp2a = grad_out * p2a
        q = gp2a.sum(-1, keepdim=True) / p2a.sum(-1, keepdim=True)
        grad_input = gp2a - p2a * q
        return grad_input, None, None

def potential(p: Tensor, a: float=1.0) -> Tensor: # Ω(p)
    if abs(a - 1) < 1e-3:
        return (p * (p + 1e-8).log()).sum(-1)
    elif abs(a - 2) < 1e-3:
        return 0.5 * ((p * p).sum(-1) - 1)
    return (p.pow(a).sum(-1) - 1) * (1 / (a * (a - 1)))

def mirror_map(p: Tensor, a: float=1.0) -> Tensor: # ∇Ω(p)
    if abs(a - 1) < 1e-3:
        return 1 + (p + 1e-8).log()
    elif abs(a - 2) < 1e-3:
        return p
    return p.pow(a - 1) * (1 / (a - 1))

def divergence(p: Tensor, q: Tensor, a: float = 1.0) -> Tensor: # D(p||q) = Ω(p) - Ω(q) - <∇Ω(q), p - q>
    if abs(a - 1) < 1e-3:
        t = p * ((p + 1e-8).log() - (q + 1e-8).log())
        return t.sum(-1)
    elif abs(a - 2) < 1e-3:
        t = p - q
        return 0.5 * (t * t).sum(-1)
    t = p.pow(a) - q.pow(a - 1) * (a * p + (1 - a) * q)
    return t.sum(-1) * (1 / (a * (a - 1)))

def inv_mirror_map(z: Tensor, mask: Tensor = None, a: float = 1.0) -> Tensor: # ∇Ω*(z)
    if mask is not None:
        z = z.masked_fill(~mask, -1e9)
    z = z - z.amax(-1, keepdim=True)
    if abs(a - 1) < 1e-3:
        return z.softmax(-1)
    elif abs(a - 2) < 1e-3:
        return Sparsemax.apply(z)
    elif abs(a - 1.5) < 1e-3:
        return Entmax15.apply(z)
    elif abs(a - 1.333) < 1e-3:
        return Entmax133.apply(z)
    return Entmax.apply(z, a, T=20)

def conjugate(z: Tensor, mask: Tensor = None, a: float = 1.0) -> Tensor: # Ω*(z) = <z, p> - Ω(p)
    if mask is not None:
        z = z.masked_fill(~mask, -1e9)
    if abs(a - 1) < 1e-3:
        return z.logsumexp(-1)
    p = inv_mirror_map(z, mask, a)
    return (z * p).sum(-1) - potential(p, a)

def mdpo_update(
    p: Tensor, g: Tensor, mask: Tensor = None, 
    a: float = 1.0, eta: float = 1.0
) -> Tensor: # ∇Ω*(∇Ω(p) + ηg)
    return inv_mirror_map(mirror_map(p, a) + eta * g, mask=mask, a=a)

class SparseFYLoss(Function):
    @staticmethod
    def forward(ctx, z: Tensor, y: Tensor, mask: Tensor = None, a: float = 1.0) -> Tensor:
        # y: (N, D) - assume uniform y over selected classes
        # z: (N, L)
        d = y.shape[-1]
        p = inv_mirror_map(z, mask, a)
        loss = -potential(p, a) # -Ω(p)
        p.scatter_add_(-1, y, torch.full_like(y, -1 / d, dtype=p.dtype)) # p - y
        loss = loss + (p * z).sum(-1)
        ctx.save_for_backward(p)
        return loss
    
    @staticmethod
    def backward(ctx, grad_out: Tensor):
        p_y, = ctx.saved_tensors
        grad_input = grad_out.unsqueeze(-1) * p_y
        return grad_input, None, None, None

class FYLoss(Function): # Ω*(z) + Ω(y) - <z, y>, Ω*(z) = <z, p> - Ω(p)
    @staticmethod
    def forward(ctx, z: Tensor, y: Tensor, mask: Tensor = None, a: float = 1.0) -> Tensor:
        # y: (N, L)
        # z: (N, L)
        p = inv_mirror_map(z, mask, a)
        p_y = p - y
        loss = potential(y, a) + (z * p_y).sum(-1) - potential(p, a) 
        ctx.save_for_backward(p_y)
        return loss
    
    @staticmethod
    def backward(ctx, grad_out: Tensor):
        p_y, = ctx.saved_tensors
        grad_input = grad_out.unsqueeze(-1) * p_y
        return grad_input, None, None, None

def fy_loss(z: Tensor, y: Tensor, mask: Tensor = None, a: float = 1.0, sparse: bool = True) -> Tensor:
    # y: (N,) or (N, D) or (N, L)
    # z: (N, L)
    if abs(a - 1) < 1e-3:
        if mask is not None:
            z = z.masked_fill(~mask, -1e9)
        return F.cross_entropy(z, y, reduction="none")
    elif sparse: # assume indices provided
        if y.ndim < z.ndim:
            y = y.unsqueeze(-1)
        return SparseFYLoss.apply(z, y, mask, a)
    else:
        return FYLoss.apply(z, y, mask, a)
