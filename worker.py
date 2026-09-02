import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import bbengine.bbengine as bb
from config import Config
from memory import MemoryBuffer
from policy_math.returns import gae, ce_loss
from utils import unpack_obs, pack_obs, reset_and_deal
from policy_math.value_scale import vscale_inv, vscale

@torch.no_grad()
def collect_rollout_hands(
    cfg: Config, P: nn.Module, V: nn.Module, 
    plan: bb.BeamSearch, env: bb.BatchEnv, 
    buf: MemoryBuffer, device: torch.device, 
) -> Tensor:
    B = env.size()
    all_idx = torch.arange(B, dtype=torch.int64)

    def policy_logits_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return P(b, i, k).cpu()
    
    def value_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return vscale_inv(V(b, i, k)).cpu()

    for _ in range(cfg.horizon_len):
        # env step
        b = env.boards()
        m = env.metas()
        legal_mask = bb.legal_mask_batch(b, m) # cpu
        board, info, block = unpack_obs(b, m, device) # (B, 64), (B, 2), (B, 3)
        v0 = vscale_inv(V(board, info, block)) # (B,)

        # sample and step
        pi_behavior = plan.search_batch(b, m, policy_logits_fn, value_fn, use_noise=True) # (M, 192)
        a = pi_behavior.multinomial(1).squeeze(-1) # (M,) cpu
        reward = env.step_indices(all_idx, a) # (M,)

        # handle afterstate
        board_after, info_after, block_after = unpack_obs(env.boards(), env.metas(), device)

        done_step = env.done() # no legal moves
        need_blocks = env.need_blocks() # boundary due to empty hand

        # deal only for non-terminal boundary afterstates
        if need_blocks.any():
            ix = need_blocks.nonzero(as_tuple=False).squeeze(-1)
            env.rand_blocks_indices(ix)

        # terminal due to deal
        done_after_deal = env.done() & need_blocks & (~done_step)
        done_t = done_after_deal | done_step

        # record transition + afterstate
        buf.append_step(
            board, info, block, legal_mask, pi_behavior,
            a, reward, done_t, v0,
            board_after, info_after, block_after,
            need_blocks
        )

        # reset done envs
        if done_t.any():
            ix = done_t.nonzero(as_tuple=False).squeeze(-1)
            env.reset_indices(ix)
            env.rand_blocks_indices(ix)

    # bootstrap
    return value_fn(env.boards(), env.metas())

# policy epoch
def run_policy_epoch(
    cfg: Config, buf: MemoryBuffer, P: nn.Module, opt_P: Optimizer, 
    sched_P: LRScheduler, device: torch.device, t0: int,
) -> None:
    t_ix = torch.arange(t0, t0 + cfg.horizon_len, dtype=torch.int64, device=device)
    batch = buf.sample_policy_rollout(t_ix)

    # flatten (T, B) -> N
    board = batch["board"].view(-1, 64)
    info = batch["info"].view(-1, 2)
    block = batch["block"].view(-1, 3)
    mask = batch["legal_mask"].view(-1, 192)
    pi_behavior = batch["pi_behavior"].view(-1, 192)

    N = board.shape[0]
    perm = torch.randperm(N, device=board.device)
    
    max_bs = cfg.max_bs_train
    n_batch = math.ceil(N / max_bs)
    ent = cfg.ent_coef_P
    n_accum = cfg.P_grad_accum_steps
    max_grad_norm = cfg.P_max_grad_norm

    opt_P.zero_grad(set_to_none=True)
    for bi in range(n_batch):
        t0 = bi * max_bs
        t1 = min((bi + 1) * max_bs, N)
        sl = perm[t0:t1]
        b = board[sl].to(device)
        i = info[sl].to(device)
        k = block[sl].to(device)

        # policy distillation loss
        logits = P(b, i, k) # (B, 192)
        loss = ce_loss(
            logits, pi_behavior[sl].to(device), 
            mask[sl].to(device), ent_coef=ent
        ) / n_accum # normalize for gradient accumulation
        loss.backward()

        # step every n_accum microbatches
        if (bi + 1) % n_accum == 0 or (bi + 1) == n_batch:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(P.parameters(), max_grad_norm)
            opt_P.step()
            sched_P.step()
            opt_P.zero_grad(set_to_none=True)

# get advantages / returns
@torch.no_grad()
def run_adv_segment(
    cfg: Config, buf: MemoryBuffer, V: nn.Module, 
    V_bootstrap: Tensor, device: torch.device, t0: int,
) -> None:
    K = cfg.n_hands_V
    T = cfg.horizon_len
    
    roll = buf.sample_rollout_segment(t0, T)
    reward = roll["reward"] # (T, B)
    done = roll["done"] # (T, B)

    # already scaled to raw units internally
    value = roll["value"] # (T, B)
    value_tp1 = torch.cat([value[1:], V_bootstrap.unsqueeze(0)], dim=0)

    # next-state bootstrap (also targets for afterstate value)
    msk = roll["need_blocks"] # chance nodes

    board_chance = roll["board_after"][msk]
    info_chance = roll["info_after"][msk]
    N = board_chance.shape[0] # N always >= 0

    boards_u64, metas_u64 = pack_obs(board_chance, info_chance) # (N,)

    rnd_hands = bb.sample_hands_crn(boards_u64, metas_u64, K).reshape(-1, 3) # (NK, 3)
    board_chance = board_chance.unsqueeze(1).expand(-1, K, -1).reshape(-1, 64) # (NK, 64)
    info_chance = info_chance.unsqueeze(1).expand(-1, K, -1).reshape(-1, 2) # (NK, 2)

    # evaluate hands that can still survive the deal
    boards_u64, metas_u64 = pack_obs(board_chance, info_chance, rnd_hands) # (NK,)
    alive_deal = bb.can_complete_hand_batch(boards_u64, rnd_hands)
    
    # bootstrap for chance nodes
    board_eval = board_chance[alive_deal]
    info_eval = info_chance[alive_deal]
    block_eval = rnd_hands[alive_deal]

    N_eval = board_eval.shape[0]
    bs = cfg.max_bs_inference_V
    n_batch = math.ceil(N_eval / bs)
    gamma = cfg.V_gamma
    v_boot_eval = torch.zeros((N_eval,), dtype=torch.float, device=buf.device)
    
    for bi in range(n_batch):
        k0 = bi * bs
        k1 = min((bi + 1) * bs, N_eval)

        b = board_eval[k0:k1].to(device)
        i = info_eval[k0:k1].to(device)
        bk = block_eval[k0:k1].to(device)
        v_boot_eval[k0:k1] = V(b, i, bk).to(buf.device)

    # value targets for chance nodes (afterstates) - TD(0)
    v_boot_eval = vscale_inv(v_boot_eval) # transform to raw scale in 1 batch
    V_scatter = torch.zeros((N*K,), dtype=torch.float, device=buf.device)
    V_scatter[alive_deal] = v_boot_eval
    boot_next = V_scatter.view(N, K).mean(-1).to(buf.device) # (N,)

    # bootstrap for gae
    value_boot = value_tp1.clone()
    value_boot[msk] = boot_next # only on deal steps

    # afterstate supervision targets (don't exist outside of afterstates)
    v_after_targ = torch.zeros_like(value_tp1)
    v_after_targ[msk] = boot_next

    _, v_targ = gae(
        value, value_boot, done, reward, 
        gamma=gamma, lmbda=cfg.V_lmbda
    )

    # scale targets
    v_after_targ = vscale(v_after_targ)
    v_targ = vscale(v_targ)
    buf.write_adv_segment(v_targ, v_after_targ, t0, T)

# value epoch
def run_value_epoch(
    cfg: Config, buf: MemoryBuffer, V: nn.Module, opt_V: Optimizer, 
    sched_V: LRScheduler, device: torch.device
) -> None:
    t_ix = torch.arange(0, buf.size, dtype=torch.int64, device=device)
    batch = buf.sample_value_rollout(t_ix)

    # pre-state
    b_t = batch["board"].view(-1, 64)
    i_t = batch["info"].view(-1, 2)
    k_t = batch["block"].view(-1, 3)
    v_pre_targ = batch["returns"].view(-1)

    # after-state
    msk_after = batch["need_blocks"].view(-1) # chance nodes

    b_after = batch["board_after"].view(-1, 64)[msk_after]
    i_after = batch["info_after"].view(-1, 2)[msk_after]
    k_after = batch["block_after"].view(-1, 3)[msk_after]
    v_after_targ = batch["v_targ_after"].view(-1)[msk_after]

    # DDP requires every rank to execute the same number of backward calls.
    # The number of chance-node afterstates is trajectory-dependent, so use
    # the global minimum and randomly subsample only the excess examples.
    if dist.is_initialized():
        n_after = torch.tensor([b_after.shape[0]], dtype=torch.int64, device=device)
        dist.all_reduce(n_after, op=dist.ReduceOp.MIN)
        n_after = int(n_after.item())
        if b_after.shape[0] > n_after:
            keep = torch.randperm(b_after.shape[0])[:n_after]
            b_after = b_after[keep]
            i_after = i_after[keep]
            k_after = k_after[keep]
            v_after_targ = v_after_targ[keep]

    board = torch.cat([b_t, b_after], dim=0)
    info = torch.cat([i_t, i_after], dim=0)
    block = torch.cat([k_t, k_after], dim=0)
    v_targ = torch.cat([v_pre_targ, v_after_targ], dim=0)
    
    N = board.shape[0]
    perm = torch.randperm(N, device=b_after.device)
    # build observations

    max_bs = cfg.max_bs_train
    n_batch = math.ceil(N / max_bs)
    n_accum = cfg.V_grad_accum_steps
    max_grad_norm = cfg.V_max_grad_norm

    opt_V.zero_grad(set_to_none=True)
    for bi in range(n_batch):
        t0 = bi * max_bs
        t1 = min((bi + 1) * max_bs, N)
        sl = perm[t0:t1]

        targ = v_targ[sl].to(device)

        # value (mse) loss
        v = V(board[sl].to(device), info[sl].to(device), block[sl].to(device))
        r = targ - v
        loss = (r * r).mean() / n_accum # normalize for gradient accumulation
        loss.backward()

        # step every n_accum microbatches
        if (bi + 1) % n_accum == 0 or (bi + 1) == n_batch:
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(V.parameters(), max_grad_norm)
            opt_V.step()
            sched_V.step()
            opt_V.zero_grad(set_to_none=True)

def train_phase(
    cfg: Config, env: bb.BatchEnv, P: nn.Module, V: nn.Module, 
    plan: bb.BeamSearch, buf: MemoryBuffer, opt_P: Optimizer, 
    opt_V: Optimizer, sched_P: LRScheduler, sched_V: LRScheduler,
    device: torch.device,
) -> None:
    reset_and_deal(env)
    V.eval()
    P.eval()
    for _ in range(cfg.n_rollouts_phase):
        V_bootstrap = collect_rollout_hands(cfg, P, V, plan, env, buf, device)

        p_t0 = buf.size - cfg.horizon_len
        # policy epochs
        P.train()
        run_policy_epoch(cfg, buf, P, opt_P, sched_P, device, p_t0)
        P.eval()
        
        v_t0 = buf.size - cfg.horizon_len
        run_adv_segment(cfg, buf, V, V_bootstrap, device, v_t0)

    # value epochs
    V.train()
    for _ in range(cfg.n_epochs_V):
        run_value_epoch(cfg, buf, V, opt_V, sched_V, device)
    V.eval()
    buf.clear()

@torch.no_grad()
def test(
    env: bb.BatchEnv, P: nn.Module, V: nn.Module, 
    device: torch.device, plan: bb.BeamSearch = None,
    n_episodes: int = 10, max_placements: int = 500,
) -> Tuple[float, float, float, float, List[float]]:
    P.eval()
    V.eval()

    def policy_logits_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return P(b, i, k).cpu()
    
    def value_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return vscale_inv(V(b, i, k)).cpu()

    B = env.size()

    all_ep_rews: List[Tensor] = []
    all_ep_lens: List[Tensor] = []

    for _ in range(n_episodes):
        reset_and_deal(env)

        dones = torch.zeros((B,), dtype=torch.bool)
        ep_rew = torch.zeros((B,), dtype=torch.float64)
        ep_len = torch.zeros((B,), dtype=torch.int64)

        placements = 0
        while (not dones.all().item()) and (placements < max_placements):
            active_idx = (~dones).nonzero(as_tuple=False).squeeze(-1)

            b = env.boards().to(torch.int64)[active_idx].to(torch.uint64)
            m = env.metas().to(torch.int64)[active_idx].to(torch.uint64)

            # search for active envs only
            if plan is not None:
                pi = plan.search_batch(b, m, policy_logits_fn, value_fn, use_noise=False) # (M, 192)
            else:
                msk = bb.legal_mask_batch(b, m)
                logits = policy_logits_fn(b, m)
                logits.masked_fill_(~msk, -1e9)
                pi = logits.softmax(-1)

            # sample and step
            a = pi.argmax(-1) # (M,) cpu
            reward = env.step_indices(active_idx, a) # (M,)
            ep_rew[active_idx] += reward.to(torch.float64)

            need_blocks = env.need_blocks()

            if need_blocks.any():
                ix = need_blocks.nonzero(as_tuple=False).squeeze(-1)
                env.rand_blocks_indices(ix)
            
            dones = env.done()
            ep_len += (~dones).to(torch.int64)

            placements += 1

        all_ep_rews.append(ep_rew.cpu())
        all_ep_lens.append(ep_len.cpu())

    rews = torch.stack(all_ep_rews, dim=0)
    lens = torch.stack(all_ep_lens, dim=0)

    avg_eps_len = lens.float().mean().item()
    mean = rews.mean().item()
    min_score, max_score = torch.aminmax(rews)
    quant = torch.quantile(rews, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64))
    return avg_eps_len, mean, min_score.item(), max_score.item(), quant.tolist()
