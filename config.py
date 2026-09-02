from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    model_name: Literal["cnn", "p4m"] = "cnn"
    seed: int = 0
    eval_seed: int = 100003

    horizon_len: int = 128
    n_rollouts_phase: int = 32
    n_phases: int = 200
    n_envs_worker: int = 64
    n_envs_test: int = 64
    n_eps_test: int = 1
    max_placements_test: int = 3000

    n_epochs_V: int = 1
    n_hands_V: int = 64

    ent_coef_P: float = 0.0
    P_beam_width: int = 128
    P_per_parent_top_m: int = 16
    P_teacher_tau: float = 2.0
    P_root_eps: float = 0.5

    V_gamma: float = 0.997
    V_lmbda: float = 0.99

    max_bs_train: int = 2048
    max_bs_inference_P: int = 40000
    max_bs_inference_V: int = 100000

    base_lr_P: float = 5e-4
    base_lr_V: float = 3e-4

    final_lr_P: float = 5e-5
    final_lr_V: float = 3e-5

    weight_decay_P: float = 1e-3
    weight_decay_V: float = 1e-3

    P_max_grad_norm: float = 0.0
    V_max_grad_norm: float = 0.0

    P_grad_accum_steps: int = 1
    V_grad_accum_steps: int = 1

    warmup_steps_P: int = 0
    warmup_steps_V: int = 0
