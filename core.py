import os
import torch
import torch.nn as nn
import bbengine.bbengine as bb
from memory import MemoryBuffer
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from model.cnn import PolicyNet, ValueNet
from worker import train_phase, test
from config import Config
from utils import CosineLR, save_checkpoint, setup_logger
from typing import Tuple
import math

def make_models(rank: int | None) -> Tuple[PolicyNet, ValueNet]:
    P = PolicyNet(n_blocks=bb.N_BLOCKS, null_block=bb.NULL_BLOCK)
    V = ValueNet(n_blocks=bb.N_BLOCKS, null_block=bb.NULL_BLOCK)

    if rank is not None:
        device = torch.device(f"cuda:{rank}")
        P = P.to(device)
        V = V.to(device)

    return P, V

def test_model(
    cfg: Config, logger, env: bb.BatchEnv, P: nn.Module, V: nn.Module, 
    plan: bb.BeamSearch, phase: int, dev: torch.device | str = "cpu", 
) -> None:
    avg_eps_len, mean, min_score, max_score, quant = test(
        env, P, V, plan, dev, 
        n_episodes=cfg.n_eps_test, max_placements=cfg.max_placements_test
    )
    logger.info(
        f"phase {phase} | mean episode length: {avg_eps_len:.4f} | mean score: {mean:.4f}"
    )
    logger.info(
        f"min={min_score:.4f} | q25={quant[0]:.4f} | q50={quant[1]:.4f} | "
        f"q75={quant[2]:.4f} | max={max_score:.4f}"
    )

def compile_model(model: nn.Module):
    return torch.compile(
        model, 
        fullgraph=True, 
        dynamic=True, 
        backend="inductor", 
        mode="max-autotune-no-cudagraphs",
    )

def train_worker(
    P: nn.Module, V: nn.Module, cfg: Config, rank: int=None, 
    base_rank: int=None, chkpt_dir: str=None, log_dir: str=None
) -> None:
    if rank is not None:
        dev = torch.device(f"cuda:{rank}")
        is_base = (rank == base_rank)
    else:
        dev = torch.device("cpu") # fallback
        is_base = True

    P_comp = compile_model(P)
    V_comp = compile_model(V)

    seed = cfg.seed + (rank if rank is not None else 0)
    bb.seed_rng(seed)
    opt_P = AdamW(P_comp.parameters(), lr=cfg.base_lr_P, weight_decay=cfg.weight_decay_P, fused=True)
    opt_V = AdamW(V_comp.parameters(), lr=cfg.base_lr_V, weight_decay=cfg.weight_decay_V, fused=True)

    nstep_max_P = (
        math.ceil(cfg.n_envs_worker * cfg.horizon_len * 
        cfg.n_rollouts_phase / (cfg.max_bs_train * cfg.P_grad_accum_steps)) * cfg.n_phases
    )

    nstep_max_V = (
        math.ceil(cfg.n_envs_worker * cfg.horizon_len * 
        cfg.n_rollouts_phase / (cfg.max_bs_train * cfg.V_grad_accum_steps)) * cfg.n_phases
    )
    sched_P = CosineLR(opt_P, nstep_max_P, eta_min=cfg.final_lr_P)
    sched_V = CosineLR(opt_V, int(nstep_max_V * 4 / 3), eta_min=cfg.final_lr_V)

    search_cfg = bb.SearchConfig()
    search_cfg.beam_width = cfg.P_beam_width
    search_cfg.per_parent_top_m = cfg.P_per_parent_top_m
    search_cfg.root_eps = cfg.P_root_eps
    search_cfg.max_eval_P = cfg.max_bs_inference_P
    search_cfg.max_eval_V = cfg.max_bs_inference_V
    search_cfg.gamma = cfg.V_gamma
    search_cfg.teacher_tau = cfg.P_teacher_tau

    plan = bb.BeamSearch(search_cfg, seed)

    buf = MemoryBuffer(
        cfg.horizon_len * cfg.n_rollouts_phase, 
        cfg.n_envs_worker, torch.device("cpu")
    )
    env = bb.BatchEnv(cfg.n_envs_worker, seed)

    logger = None
    if is_base and log_dir is not None:
        logger = setup_logger(log_dir, restart=True)
        logger.info(f"policy has {sum(a.numel() for a in P_comp.parameters())} parameters")
        logger.info(f"value has {sum(a.numel() for a in V_comp.parameters())} parameters")
        test_model(cfg, logger, env, P_comp, V_comp, plan, 0, dev)

    for phase in range(cfg.n_phases):
        train_phase(
            cfg, env, P_comp, V_comp, plan, buf, opt_P, opt_V, sched_P, sched_V, dev
        )
        if is_base and logger is not None:
            test_model(cfg, logger, env, P_comp, V_comp, plan, phase + 1, dev) # phase 0 is pre-test

        if is_base and chkpt_dir is not None:
            save_checkpoint(
                cfg, chkpt_dir, f"chkpt{phase}", P, V, opt_P, # save original models
                opt_V, sched_P, sched_V,
            )

def run_local(cfg: Config, rank: int=None, chkpt_dir: str=None, log_dir: str=None):
    P, V = make_models(rank)
    train_worker(P, V, cfg, rank, rank, chkpt_dir, log_dir)

def setup_worker(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_worker() -> None:
    dist.destroy_process_group()

class DistributedRunner:
    def __init__(self, cfg: Config, world_size: int, base_rank: int=0, chkpt_dir: str=None, log_dir: str=None):
        super().__init__()
        self.size = world_size
        self.chkpt_dir = chkpt_dir
        self.log_dir = log_dir
        self.cfg = cfg
        self.base_rank = base_rank

    def run(self) -> None:
        mp.spawn(self.ddp_worker, args=(self.size,), nprocs=self.size, join=True)

    def ddp_worker(self, rank: int, world_size: int) -> None:
        setup_worker(rank, world_size)
        P, V = make_models(rank)
        P = DDP(P, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
        V = DDP(V, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
        train_worker(P, V, self.cfg, rank, self.base_rank, self.chkpt_dir, self.log_dir)
        cleanup_worker()