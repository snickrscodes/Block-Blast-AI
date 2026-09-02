import math
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

import bbengine.bbengine as bb
from config import Config
from memory import MemoryBuffer
from utils import (
    CosineLR, load_checkpoint, load_model_checkpoint, save_checkpoint,
    setup_logger,
)
from worker import test, train_phase


SEED_TORCH = 1
SEED_ENV = 2
SEED_SEARCH = 3
SEED_CRN = 4
_SEED_MASK = (1 << 63) - 1


def phase_seed(base: int, phase: int, rank: int, stream: int) -> int:
    return (
        base
        + 1_000_003 * (phase + 1)
        + 10_007 * (rank + 1)
        + stream
    ) & _SEED_MASK


def seed_torch(seed: int, device: torch.device | None = None) -> None:
    torch.manual_seed(seed)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed)


def make_models(
    cfg: Config, device: torch.device | str | None = None
) -> Tuple[nn.Module, nn.Module]:
    if cfg.model_name == "cnn":
        from model.cnn import PolicyNet, ValueNet
    elif cfg.model_name == "p4m":
        from model.p4mcnn import PolicyNet, ValueNet
    else:
        raise ValueError(f"unknown model_name: {cfg.model_name}")

    P = PolicyNet(n_blocks=bb.N_BLOCKS, null_block=bb.NULL_BLOCK)
    V = ValueNet(n_blocks=bb.N_BLOCKS, null_block=bb.NULL_BLOCK)

    if device is not None:
        P = P.to(device)
        V = V.to(device)

    return P, V


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def make_search_config(cfg: Config) -> bb.SearchConfig:
    search_cfg = bb.SearchConfig()
    search_cfg.beam_width = cfg.P_beam_width
    search_cfg.per_parent_top_m = cfg.P_per_parent_top_m
    search_cfg.root_eps = cfg.P_root_eps
    search_cfg.max_eval_P = cfg.max_bs_inference_P
    search_cfg.max_eval_V = cfg.max_bs_inference_V
    search_cfg.gamma = cfg.V_gamma
    search_cfg.teacher_tau = cfg.P_teacher_tau
    return search_cfg


def test_model(
    cfg: Config, logger, P: nn.Module, V: nn.Module,
    phase: int, dev: torch.device | str = "cpu",
) -> None:
    eval_env = bb.BatchEnv(cfg.n_envs_test, cfg.eval_seed)
    eval_plan = bb.BeamSearch(make_search_config(cfg), cfg.eval_seed + 1)

    avg_eps_len, mean, min_score, max_score, quant = test(
        eval_env, P, V, dev, eval_plan,
        n_episodes=cfg.n_eps_test, max_placements=cfg.max_placements_test,
    )
    logger.info(
        f"phase {phase} | mean episode length: {avg_eps_len:.4f} | "
        f"mean score: {mean:.4f}"
    )
    logger.info(
        f"min={min_score:.4f} | q25={quant[0]:.4f} | q50={quant[1]:.4f} | "
        f"q75={quant[2]:.4f} | max={max_score:.4f}"
    )


def fits(bs: int, model: nn.Module, device: torch.device, trials: int = 5) -> bool:
    try:
        with torch.no_grad():
            for _ in range(trials):
                board = (
                    torch.rand((bs, 64), device=device) < 0.6
                ).to(torch.float)

                combo = torch.randint(0, 10, (bs, 1))
                ctr = torch.randint(1, 6, (bs, 1))
                info = torch.cat([combo, ctr], dim=-1).to(device)

                block = torch.randint(0, bb.NULL_BLOCK, (bs, 3))
                block = torch.where(
                    torch.rand(bs, 3) < 0.3, bb.NULL_BLOCK, block
                ).to(device)
                _ = model(board, info, block)
        return True
    except RuntimeError:
        return False


def find_max_batch(
    model: nn.Module, device: torch.device,
    lo: int = 2048, hi: int = 131072, tol: int = 256,
) -> int:
    model.eval()
    low, high = lo, hi
    while (high - low) > tol:
        mid = (low + high + 1) // 2
        if fits(mid, model, device):
            low = mid
        else:
            high = mid - 1
    return low


def compile_model(model: nn.Module, is_ddp: bool):
    # return torch.compile(
    #     model,
    #     fullgraph=not is_ddp,
    #     dynamic=True,
    #     backend="inductor",
    #     mode="max-autotune-no-cudagraphs",
    # )
    return model


def train_worker(
    P: nn.Module, V: nn.Module, cfg: Config, is_ddp: bool, rank: int = None,
    base_rank: int = None, chkpt_dir: str = None, log_dir: str = None,
    resume_name: str = None, start_path: str = None,
) -> None:
    if rank is not None:
        dev = torch.device(f"cuda:{rank}")
        is_base = rank == base_rank
        seed_rank = rank
    else:
        dev = torch.device("cpu")  # fallback
        is_base = True
        seed_rank = 0

    if resume_name is not None and start_path is not None:
        raise ValueError("resume_name and start_path are mutually exclusive")

    start_cfg = None
    if start_path is not None:
        start_cfg = load_model_checkpoint(start_path, P, V, dev)
        if start_cfg.model_name != cfg.model_name:
            raise ValueError(
                "starting checkpoint model architecture does not match the "
                f"requested run: {start_cfg.model_name!r} != {cfg.model_name!r}"
            )

    P_comp = compile_model(P, is_ddp)
    V_comp = compile_model(V, is_ddp)

    opt_P = AdamW(
        P_comp.parameters(),
        lr=cfg.base_lr_P,
        weight_decay=cfg.weight_decay_P,
        fused=dev.type == "cuda",
    )
    opt_V = AdamW(
        V_comp.parameters(),
        lr=cfg.base_lr_V,
        weight_decay=cfg.weight_decay_V,
        fused=dev.type == "cuda",
    )

    nstep_max_P = (
        math.ceil(
            cfg.n_envs_worker * cfg.horizon_len * cfg.n_rollouts_phase
            / (cfg.max_bs_train * cfg.P_grad_accum_steps)
        )
        * cfg.n_phases
    )

    nstep_max_V = (
        math.ceil(
            cfg.n_envs_worker * cfg.horizon_len * cfg.n_rollouts_phase
            / (cfg.max_bs_train * cfg.V_grad_accum_steps)
        )
        * cfg.n_phases
    )
    sched_P = CosineLR(
        opt_P, nstep_max_P, eta_min=cfg.final_lr_P,
        warmup_steps=cfg.warmup_steps_P,
    )
    sched_V = CosineLR(
        opt_V, int(nstep_max_V * 4 / 3), eta_min=cfg.final_lr_V,
        warmup_steps=cfg.warmup_steps_V,
    )

    start_phase = 0
    if resume_name is not None:
        if chkpt_dir is None:
            raise ValueError("chkpt_dir is required when resume_name is set")
        saved_cfg, start_phase = load_checkpoint(
            chkpt_dir, resume_name, P, V, opt_P, opt_V,
            sched_P, sched_V, dev,
        )
        if saved_cfg != cfg:
            raise ValueError(
                "checkpoint config does not match the requested run\n"
                f"saved: {saved_cfg}\nrequested: {cfg}"
            )
        if not 0 <= start_phase <= cfg.n_phases:
            raise ValueError(
                f"invalid completed_phase={start_phase} for n_phases={cfg.n_phases}"
            )

    search_cfg = make_search_config(cfg)
    plan = bb.BeamSearch(search_cfg, cfg.seed)

    buf = MemoryBuffer(
        cfg.horizon_len * cfg.n_rollouts_phase,
        cfg.n_envs_worker,
        torch.device("cpu"),
    )

    logger = None
    if is_base and log_dir is not None:
        logger = setup_logger(log_dir, restart=resume_name is None)
        logger.info(f"model architecture: {cfg.model_name}")
        logger.info(
            f"policy has {sum(a.numel() for a in P_comp.parameters())} parameters"
        )
        logger.info(
            f"value has {sum(a.numel() for a in V_comp.parameters())} parameters"
        )
        if resume_name is not None:
            logger.info(
                f"resumed from {resume_name} at completed phase {start_phase}"
            )
        elif start_path is not None:
            logger.info(f"initialized model weights from {start_path}")
            logger.info(
                "fresh optimizer/scheduler state | "
                f"warmup steps P={cfg.warmup_steps_P}, "
                f"V={cfg.warmup_steps_V}"
            )

    # Rank-0-only probing/evaluation uses the underlying module rather than
    # forwarding through DDP while the other ranks wait.
    P_eval = unwrap_model(P_comp)
    V_eval = unwrap_model(V_comp)

    if is_ddp:
        dist.barrier()
    if is_base and logger is not None and resume_name is None:
        logger.info(f"max batch size for policy is {find_max_batch(P_eval, dev)}")
        logger.info(f"max batch size for value is {find_max_batch(V_eval, dev)}")
        test_model(cfg, logger, P_eval, V_eval, 0, dev)
    if is_ddp:
        dist.barrier()

    for phase in range(start_phase, cfg.n_phases):
        torch_seed = phase_seed(cfg.seed, phase, seed_rank, SEED_TORCH)
        env_seed = phase_seed(cfg.seed, phase, seed_rank, SEED_ENV)
        search_seed = phase_seed(cfg.seed, phase, seed_rank, SEED_SEARCH)
        crn_seed = phase_seed(cfg.seed, phase, seed_rank, SEED_CRN)

        seed_torch(torch_seed, dev)
        bb.seed_rng(crn_seed)
        env = bb.BatchEnv(cfg.n_envs_worker, env_seed)
        plan.base_seed = search_seed

        train_phase(
            cfg, env, P_comp, V_comp, plan, buf, opt_P, opt_V,
            sched_P, sched_V, dev,
        )

        if is_ddp:
            dist.barrier()

        if is_base and logger is not None:
            test_model(cfg, logger, P_eval, V_eval, phase + 1, dev)

        if is_base and chkpt_dir is not None:
            save_checkpoint(
                cfg, chkpt_dir, f"chkpt{phase}", P, V, opt_P, opt_V,
                sched_P, sched_V, completed_phase=phase + 1,
            )

        if is_ddp:
            dist.barrier()


def run_local(
    cfg: Config, rank: int = None, chkpt_dir: str = None,
    log_dir: str = None, resume_name: str = None, start_path: str = None,
):
    if rank is not None:
        dev = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(rank)
    else:
        dev = torch.device("cpu")
    seed_torch(cfg.seed, dev)
    P, V = make_models(cfg, dev)
    train_worker(
        P, V, cfg, False, rank, rank, chkpt_dir, log_dir,
        resume_name, start_path,
    )


def setup_worker(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.set_float32_matmul_precision("high")  # enable TF32
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_worker() -> None:
    dist.destroy_process_group()


class DistributedRunner:
    def __init__(
        self, cfg: Config, world_size: int, base_rank: int = 0,
        log_dir: str = None, chkpt_dir: str = None, resume_name: str = None,
        start_path: str = None,
    ):
        super().__init__()
        self.size = world_size
        self.chkpt_dir = chkpt_dir
        self.log_dir = log_dir
        self.cfg = cfg
        self.base_rank = base_rank
        self.resume_name = resume_name
        self.start_path = start_path

    def run(self) -> None:
        mp.spawn(self.ddp_worker, args=(self.size,), nprocs=self.size, join=True)

    def ddp_worker(self, rank: int, world_size: int) -> None:
        setup_worker(rank, world_size)
        dev = torch.device(f"cuda:{rank}")
        seed_torch(self.cfg.seed, dev)
        P, V = make_models(self.cfg, dev)
        P = DDP(
            P, device_ids=[rank], find_unused_parameters=False,
            broadcast_buffers=False,
        )
        V = DDP(
            V, device_ids=[rank], find_unused_parameters=False,
            broadcast_buffers=False,
        )
        train_worker(
            P, V, self.cfg, True, rank, self.base_rank,
            self.chkpt_dir, self.log_dir, self.resume_name, self.start_path,
        )
        cleanup_worker()
