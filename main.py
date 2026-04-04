import torch
from core import Config, run_local
import torch.multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    cfg = Config(
        n_envs_worker=256, 
        n_rollouts_phase=32, 
        n_phases=200,
        horizon_len=512, 
        n_eps_test=1, 
        max_bs_inference_P=60000,
        max_bs_inference_V=60000,
        max_bs_train=2048, 
        max_placements_test=3000,
        n_hands_V=512,
        V_gamma=0.997,
        V_lmbda=0.99,
        P_per_parent_top_m=24,
        P_beam_width=576,
        P_root_eps=0.5,
        P_teacher_tau=2.0,
    )
    torch.manual_seed(cfg.seed)
    run_local(cfg, rank=0, log_dir="/workspace/block_blast/logs", chkpt_dir="/workspace/block_blast/chkpts")