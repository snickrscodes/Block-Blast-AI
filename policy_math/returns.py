import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple

def ppo_loss(
    adv: Tensor, logp_old: Tensor, logp_new: Tensor, 
    ent: Tensor, eps: float = 0.2, ent_coef: float = 0.0
) -> Tensor:
    ratio = (logp_new - logp_old).exp()
    loss = -(torch.min(ratio * adv, ratio.clamp(1.0 - eps, 1.0 + eps) * adv))
    if ent_coef > 0.0:
        loss = loss - ent_coef * ent
    return loss.mean()

def ce_loss(logits: Tensor, y: Tensor, msk: Tensor, ent_coef: float = 0.0):
    # return F.cross_entropy(logits.masked_fill(~msk, -1e9), y, reduction="mean")
    logp = torch.log_softmax(logits.masked_fill(~msk, -1e9), dim=-1)
    loss = -y * logp
    if ent_coef > 0.0:
        nent = torch.where(msk, logp.exp() * logp, 0)
        loss = loss + ent_coef * nent
    return loss.sum(-1).mean()

def kl_div(logits_p: Tensor, logits_q: Tensor, msk: Tensor):
    logp = logits_p.masked_fill(~msk, -1e9).log_softmax(-1)
    logq = logits_q.masked_fill(~msk, -1e9).log_softmax(-1)
    return (logp.exp() * (logp - logq)).sum(-1).mean()

def gumbel01(x: Tensor) -> Tensor:
    u = torch.clamp(torch.rand_like(x), 1e-8, 1.0-1e-8)
    return -torch.log(-torch.log(u))

def vtrace(
    rewards: Tensor, values: Tensor, values_tp1: Tensor, terminal: Tensor, 
    logp_pi: Tensor, logp_mu: Tensor, 
    gamma: float = 0.99, rho_bar: float = 1.0, c_bar: float = 1.0, 
    lmbda: float = 1.0, norm_adv: bool = False,
) -> Tuple[Tensor, Tensor]:
    # rewards, terminal, values, values_tp1, logp_pi, logp_mu, valid, nstep: (T, B)
    T, B = rewards.shape
    gammas = gamma * (~terminal)
    ratio = torch.exp(logp_pi - logp_mu)
    rhos = torch.clamp_max(ratio, rho_bar)
    cs = lmbda * torch.clamp_max(ratio, c_bar)
    deltas = rhos * (rewards + gammas * values_tp1 - values)

    vs = torch.zeros_like(rewards)
    acc = torch.zeros((B,), dtype=torch.float, device=rewards.device)
    for t in reversed(range(T)):
        acc = deltas[t] + gammas[t] * cs[t] * acc
        vs[t] = values[t] + acc
    v_tp1 = torch.cat([vs[1:], values_tp1[-1].unsqueeze(0)], dim=0)
    pg_adv = rhos * (rewards + gammas * v_tp1 - values)

    if norm_adv:
        var, mean = torch.var_mean(pg_adv, dim=-1, correction=0, keepdim=True)
        pg_adv = (pg_adv - mean) * (var + 1e-5).rsqrt()
    return pg_adv.detach(), vs.detach()

def retrace(
    rewards: Tensor, values_tp1: Tensor, qvalues_tp1: Tensor, terminal: Tensor, 
    logp_pi: Tensor, logp_mu: Tensor, gamma: float=1.0, lmbda: float=1.0
) -> Tensor:
    # rewards, logp_model, logp_search, terminal, values_tp1, qvalues_tp1: (T, B)
    T, B = rewards.shape
    gammas = gamma * (~terminal).to(rewards.dtype)
    cs = lmbda * torch.exp(torch.clamp_max(logp_pi - logp_mu, 0)) # (T, B)
    cs = F.pad(cs, (0, 0, 0, 1), value=0.0) # c_T = 0

    deltas = rewards + gammas * values_tp1
    targets = torch.zeros_like(rewards)
    q_retrace = torch.zeros((B,), device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        q_retrace = deltas[t] + gammas[t] * cs[t+1] * (q_retrace - qvalues_tp1[t])
        targets[t] = q_retrace
    return targets.detach()

def gae(
    values: Tensor, values_tp1: Tensor, terminal: Tensor, rewards: Tensor,   
    gamma: float = 0.99, lmbda: float = 0.95, norm_adv: bool = False,
) -> Tuple[Tensor, Tensor]:
    # values, values_tp1, rewards, terminal, valid, nstep: (T, B)
    T, B = rewards.shape
    gammas = gamma * (~terminal)
    deltas = rewards + gammas * values_tp1 - values # (T, ...)
    gae = torch.zeros_like(rewards)
    adv = torch.zeros((B,), dtype=torch.float, device=rewards.device)
    for t in reversed(range(T)):
        adv = deltas[t] + gammas[t] * lmbda * adv
        gae[t] = adv
    
    returns = (values + gae).detach()
    if norm_adv:
        var, mean = torch.var_mean(gae, dim=-1, correction=0, keepdim=True)
        gae = (gae - mean) * (var + 1e-5).rsqrt()
    return gae.detach(), returns

# very unlikely to be used
def nstep_td(
    rewards: Tensor, values: Tensor, terminal: Tensor, 
    gamma: float=0.99, N: int=3
) -> Tensor:
    # rewards: (T, B)
    # values: (T+N, B)
    # terminal: (T, B)
    q = values[N:] # (T, B)
    if N == 1:
        return rewards + gamma * (~terminal).to(rewards.dtype) * q

    rews = F.pad(rewards, (0, 0, 0, N-1)).unfold(0, N, 1) # (T, B, N)
    rews = torch.cat([rews, q.unsqueeze(-1)], dim=-1) # (T, B, N+1)

    mask = F.pad(terminal, (0, 0, 0, N), value=True).unfold(0, N+1, 1) # (T, B, N+1)
    mask = F.pad(mask, (1, 0), value=False)[..., :-1] # (T, B, N+1)

    mask = ~torch.cummax(mask, dim=-1).values
    gammas = gamma ** torch.arange(N+1, dtype=rewards.dtype, device=rewards.device)
    targets = torch.sum(rews * gammas * mask, dim=-1) # (T, B)
    return targets.detach()

def nstep_td_loop(
    rewards: Tensor, q: Tensor, terminal: Tensor, 
    gamma: float=0.99, N: int=3
) -> Tensor:
    # rewards: (T, B)
    # values: (T+N, B)
    # terminal: (T, B)
    TpN, B = rewards.shape
    T = TpN - N
    targets = torch.zeros_like(rewards)
    terms = torch.zeros((T, B, N+1))

    for b in range(B):
        for t in range(T):
            G, discount = 0, 1
            for k in range(N):
                if t + k >= T: break
                G += discount * rewards[t+k, b]
                terms[t, b, k] = discount * rewards[t+k, b]
                discount *= gamma
                if terminal[t+k, b]: # cut trajectory here
                    break
            else: # bootstrap
                terms[t, b, -1] = discount * q[t+N, b]
                G += discount * q[t+N, b]
            targets[t, b] = G
    return targets
