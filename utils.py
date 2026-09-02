import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import os
from config import Config
from dataclasses import asdict
import math
from typing import List, Tuple
import bbengine.bbengine as bb

BOARD_SHIFTS = 1 << torch.arange(64, dtype=torch.int64) # cpu


def reset_and_deal(env: bb.BatchEnv) -> None:
    env.reset_all()
    idx = torch.arange(env.size(), dtype=torch.int64)
    env.rand_blocks_indices(idx)


def device_report(logger: logging.Logger):
    logger.info("=" * 50)
    logger.info("pytorch / cuda device report")
    logger.info("=" * 50)
    logger.info(f"pytorch version      : {torch.__version__}")
    logger.info(f"cuda available       : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"cuda runtime version : {torch.version.cuda}")
        logger.info(f"cudnn version        : {torch.backends.cudnn.version()}")
        logger.info(f"device count         : {torch.cuda.device_count()}")
        logger.info("-" * 50)

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"gpu {i}: {props.name}")
            logger.info(f"  compute capability : {props.major}.{props.minor}")
            logger.info(f"  total memory       : {props.total_memory / 1e9:.2f} GB")
            logger.info(f"  multiprocessors    : {props.multi_processor_count}")
            logger.info("-" * 50)

        current = torch.cuda.current_device()
        logger.info(f"current device index : {current}")
        logger.info(f"current device name  : {torch.cuda.get_device_name(current)}")
    else:
        logger.info("running on cpu only, no cuda devices detected")
    logger.info("=" * 50)

def setup_logger(log_dir: str, restart=True) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    if restart:
        open(log_path, "w").close() # clear file

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(), # prints to console
        ]
    )
    logger = logging.getLogger()
    if restart:
        device_report(logger)
    return logger

def checkpoint_config(
    chkpt_dir: str, name: str, dev: torch.device | str = "cpu"
) -> Config:
    data = torch.load(
        os.path.join(chkpt_dir, name), map_location=dev, weights_only=True
    )
    return Config(**data["hparams"])


def save_checkpoint(
    cfg: Config, chkpt_dir: str, name: str,
    P: nn.Module, V: nn.Module,
    opt_P: Optimizer, opt_V: Optimizer,
    sched_P: LRScheduler, sched_V: LRScheduler,
    completed_phase: int,
) -> None:
    os.makedirs(chkpt_dir, exist_ok=True)
    if isinstance(P, DDP):
        P_mod = P.module
        V_mod = V.module
    else:
        P_mod = P
        V_mod = V

    data = dict(
        checkpoint_version=1,
        completed_phase=completed_phase,
        P_state=P_mod.state_dict(),
        V_state=V_mod.state_dict(),
        opt_P_state=opt_P.state_dict(),
        opt_V_state=opt_V.state_dict(),
        sched_P_state=sched_P.state_dict(),
        sched_V_state=sched_V.state_dict(),
        hparams=asdict(cfg),
    )

    path = os.path.join(chkpt_dir, name)
    tmp_path = path + ".tmp"
    torch.save(data, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    chkpt_dir: str, name: str, P: nn.Module, V: nn.Module,
    opt_P: Optimizer, opt_V: Optimizer,
    sched_P: LRScheduler, sched_V: LRScheduler,
    dev: torch.device | str = "cpu",
) -> Tuple[Config, int]:
    data = torch.load(
        os.path.join(chkpt_dir, name), map_location=dev, weights_only=True
    )

    if isinstance(P, DDP):
        P_mod = P.module
        V_mod = V.module
    else:
        P_mod = P
        V_mod = V

    P_mod.load_state_dict(data["P_state"], strict=True)
    V_mod.load_state_dict(data["V_state"], strict=True)
    opt_P.load_state_dict(data["opt_P_state"])
    opt_V.load_state_dict(data["opt_V_state"])
    sched_P.load_state_dict(data["sched_P_state"])
    sched_V.load_state_dict(data["sched_V_state"])

    if "completed_phase" not in data:
        raise ValueError(
            "checkpoint predates resumable training and has no completed_phase"
        )

    return Config(**data["hparams"]), int(data["completed_phase"])


def load_model_checkpoint(
    path: str, P: nn.Module, V: nn.Module,
    dev: torch.device | str = "cpu",
) -> Config:
    """Load only policy/value weights from a checkpoint.

    Optimizer, scheduler, and completed-phase state are intentionally ignored.
    This is the loader used to start a new fine-tuning run from an existing
    model while allowing the new run to use a different Config.
    """
    data = torch.load(path, map_location=dev, weights_only=True)

    if isinstance(P, DDP):
        P_mod = P.module
        V_mod = V.module
    else:
        P_mod = P
        V_mod = V

    P_mod.load_state_dict(data["P_state"], strict=True)
    V_mod.load_state_dict(data["V_state"], strict=True)
    return Config(**data["hparams"])


def load_models(
    chkpt_dir: str, name: str, P: nn.Module, V: nn.Module,
    dev: torch.device | str = "cpu",
) -> Config:
    """Backward-compatible directory/name wrapper for model-only loading."""
    return load_model_checkpoint(
        os.path.join(chkpt_dir, name), P, V, dev
    )


class CosineLR(LRScheduler):
    def __init__(self,
        optimizer: Optimizer,
        total_steps: int,
        eta_min: float = 0.0,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if warmup_steps >= total_steps:
            raise ValueError("warmup_steps must be smaller than total_steps")

        self.total_steps = total_steps
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def _compute_lr_at(self, base_lr: float, step: int) -> float:
        step = max(step, 0)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            # LRScheduler performs an initial step during construction, so at
            # step 0 the optimizer begins at 1 / warmup_steps of base_lr and
            # reaches base_lr on the last warmup update.
            return base_lr * (step + 1) / self.warmup_steps

        decay_steps = self.total_steps - self.warmup_steps
        decay_step = min(max(step - self.warmup_steps, 0), decay_steps)
        cos_w = 0.5 * (1 + math.cos(math.pi * decay_step / decay_steps))
        return self.eta_min + (base_lr - self.eta_min) * cos_w

    def get_lr(self) -> List[float]:
        step = self.last_epoch  # incremented just before this call
        return [self._compute_lr_at(base_lr, step) for base_lr in self.base_lrs]

class LinearLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _compute_lr_at(self, base_lr: float, step: int) -> float:
        t = min(step, self.total_steps) / self.total_steps
        return base_lr * (1 - t) + self.eta_min * t

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        return [self._compute_lr_at(base_lr, step) for base_lr in self.base_lrs]


class ExponentialLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        gamma: float,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        self.gamma = gamma
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def _compute_lr_at(self, base_lr: float, step: int) -> float:
        return self.eta_min + (
            base_lr - self.eta_min
        ) * (self.gamma ** step)

    def get_lr(self) -> List[float]:
        step = self.last_epoch
        return [self._compute_lr_at(base_lr, step) for base_lr in self.base_lrs]
    
@torch.no_grad()
def unpack_obs(
    boards: Tensor, metas: Tensor, device: torch.device | str = "cpu"
) -> Tuple[Tensor, Tensor, Tensor]:
    # boards, metas: (N,) uint64 on cpu

    N = boards.shape[0]
    boards = boards.to(torch.int64)
    metas = metas.to(torch.int64)

    board = (boards.unsqueeze(-1) & BOARD_SHIFTS) != 0
    
    info = torch.empty((N, 2), dtype=torch.int64)
    block = torch.empty((N, 3), dtype=torch.int64)

    info[:, 0] = (metas >> bb.COMBO_SHIFT) & bb.COMBO_MASK
    info[:, 1] = (metas >> bb.CTR_SHIFT) & bb.CTR_MASK

    block[:, 0] = (metas >> bb.B0_SHIFT) & bb.BLOCK_MASK
    block[:, 1] = (metas >> bb.B1_SHIFT) & bb.BLOCK_MASK
    block[:, 2] = (metas >> bb.B2_SHIFT) & bb.BLOCK_MASK

    return board.to(device, dtype=torch.float), info.to(device), block.to(device)

@torch.no_grad()
def pack_obs(boards: Tensor, infos: Tensor, blocks: Tensor = None) -> Tuple[Tensor, Tensor]:
    # boards: (N, 64) float
    # infos: (N, 2) int64
    # blocks: (N, 3) int64

    board = (boards.cpu().to(torch.int64) * BOARD_SHIFTS).sum(-1)
    
    combo = infos[:, 0]
    ctr = infos[:, 1]

    if blocks is None:
        meta = ((combo & bb.COMBO_MASK) | ((ctr & bb.CTR_MASK) << bb.CTR_SHIFT)).cpu()
        return board.to(torch.uint64), meta.to(torch.uint64)

    b0 = blocks[:, 0]
    b1 = blocks[:, 1]
    b2 = blocks[:, 2]

    meta = (
        (combo & bb.COMBO_MASK)
      | ((ctr & bb.CTR_MASK) << bb.CTR_SHIFT)
      | ((b0 & bb.BLOCK_MASK) << bb.B0_SHIFT)
      | ((b1 & bb.BLOCK_MASK) << bb.B1_SHIFT)
      | ((b2 & bb.BLOCK_MASK) << bb.B2_SHIFT)
    ).cpu()

    return board.to(torch.uint64), meta.to(torch.uint64)
