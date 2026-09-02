import torch
import torch.multiprocessing as mp

from config import Config
from core import DistributedRunner, run_local

# phase 1 warmup

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = Config(
        model_name="cnn",
        n_envs_worker=256,
        n_envs_test=256,
        n_rollouts_phase=32,
        n_phases=40,
        horizon_len=128,
        n_eps_test=1,
        max_bs_inference_P=40000,
        max_bs_inference_V=100000,
        max_bs_train=8192,
        max_placements_test=3000,
        n_hands_V=64,
        V_gamma=0.997,
        V_lmbda=0.99,
        P_per_parent_top_m=16,
        P_beam_width=128,
        P_root_eps=0.5,
        P_teacher_tau=2.0,
        warmup_steps_P=0,
        warmup_steps_V=0,
    )

    # runner = DistributedRunner(
    #     cfg,
    #     world_size=4,
    #     base_rank=0,
    #     log_dir="/workspace/block_blast/logs",
    #     chkpt_dir="/workspace/block_blast/chkpts",
    #     resume_name=None,
    # )
    # runner.run()

    run_local(cfg, rank=0, log_dir="/workspace/block_blast/logs", chkpt_dir="/workspace/block_blast/chkpts", resume_name=None, start_path=None)

# phase 2 fine tune

# import torch
# import torch.multiprocessing as mp

# from config import Config
# from core import DistributedRunner, run_local


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")

#     cfg = Config(
#         model_name="cnn",
#         n_envs_worker=256,
#         n_envs_test=256,
#         n_rollouts_phase=16,
#         n_phases=15,
#         horizon_len=512,
#         n_eps_test=1,
#         max_bs_inference_P=35000,
#         max_bs_inference_V=100000,
#         max_bs_train=8192,
#         max_placements_test=3000,
#         n_hands_V=256,
#         V_gamma=0.997,
#         V_lmbda=0.99,
#         P_per_parent_top_m=24,
#         P_beam_width=256,
#         P_root_eps=0.5,
#         P_teacher_tau=2.0,
#         base_lr_P=1e-4,
#         base_lr_V=5e-5,
#         final_lr_P=1e-5,
#         final_lr_V=5e-6,
#         warmup_steps_P=128,
#         warmup_steps_V=192,
#     )

#     # runner = DistributedRunner(
#     #     cfg,
#     #     world_size=4,
#     #     base_rank=0,
#     #     log_dir="/workspace/block_blast/logs",
#     #     chkpt_dir="/workspace/block_blast/chkpts",
#     #     resume_name=None,
#     # )
#     # runner.run()

#     run_local(
#         cfg, rank=0, log_dir="/workspace/block_blast/logs_finetune", 
#         chkpt_dir="/workspace/block_blast/chkpts_finetune", resume_name=None,
#         start_pathr="/workspace/block_blast/chkpts/chkpt35",
#     )

# phase 3 fine tune

# import torch
# import torch.multiprocessing as mp

# from config import Config
# from core import DistributedRunner, run_local


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     torch.set_float32_matmul_precision("high")

#     cfg = Config(
#         model_name="cnn",
#         n_envs_worker=256,
#         n_envs_test=256,
#         n_rollouts_phase=32,
#         n_phases=5,
#         horizon_len=512,
#         n_eps_test=1,
#         max_bs_inference_P=35000,
#         max_bs_inference_V=100000,
#         max_bs_train=8192,
#         max_placements_test=3000,
#         n_hands_V=1024,
#         V_gamma=0.997,
#         V_lmbda=0.99,
#         P_per_parent_top_m=32,
#         P_beam_width=1024,
#         P_root_eps=0.5,
#         P_teacher_tau=2.0,
#         base_lr_P=3e-4,
#         base_lr_V=1.5e-4,
#         final_lr_P=3e-4,
#         final_lr_V=1.5e-4,
#         warmup_steps_P=128,
#         warmup_steps_V=192,
#     )

#     # runner = DistributedRunner(
#     #     cfg,
#     #     world_size=4,
#     #     base_rank=0,
#     #     log_dir="/workspace/block_blast/logs",
#     #     chkpt_dir="/workspace/block_blast/chkpts",
#     #     resume_name=None,
#     # )
#     # runner.run()

#     run_local(
#         cfg, rank=0, log_dir="/workspace/block_blast/logs_finetune2", 
#         chkpt_dir="/workspace/block_blast/chkpts_finetune2", resume_name=None,
#         start_path="/workspace/block_blast/chkpts_finetune/chkpt2",
#     )
