import itertools
import torch
import torch.nn as nn
from torch import Tensor
from model.group import DihedralGroup, calc_spatial_idx
from model.p4mcnn import PolicyNet, ValueNet
import bbengine.bbengine as bb

G = DihedralGroup(4)
NULL_ID = bb.NULL_BLOCK
PID2G = torch.tensor(bb.pid2g.tolist(), dtype=torch.int64)
PID2B = torch.tensor(bb.pid2b.tolist(), dtype=torch.int64)
G2POSE = torch.tensor(bb.g2pose.tolist(), dtype=torch.int64)
POSE_OFF = torch.tensor(bb.pose_off.tolist(), dtype=torch.int64)

# ---- helpers: D4 on board and block orientations, S3 on slots ----

def transform_board_Z2(board_flat: Tensor, spatial_idx: Tensor, g: int) -> Tensor:
    return board_flat[:, spatial_idx[g]]

def transform_blocks_D4_S3(block: Tensor, g: int, perm: Tensor) -> Tensor:
    block = block[:, perm]
    bid = PID2B[block]
    gid = PID2G[block]

    gid_new = G.product_table[g, gid]

    block_new = POSE_OFF[bid] + G2POSE[bid, gid_new]
    return block_new

def transform_logits_D4_S3(logits: Tensor, spatial_idx: Tensor, g: int, perm: Tensor) -> Tensor:
    logits = logits.view(logits.size(0), 3, -1)
    logits = logits[:, perm.unsqueeze(-1), spatial_idx[g]]
    return logits

# ---- main test ----

def test_policy_equivariance(net: nn.Module, atol: float = 1e-5):
    net.eval()
    torch.manual_seed(42)

    B, H, W = 8, 8, 8
    spatial_idx = calc_spatial_idx(G.mats, H, W)  # (8, 64), consistent with your conv tests

    # random board in the tensor layout the net uses (flattened row-major)
    board = torch.randn(B, 64).double()

    # info is trivial rep: leave unchanged under D4 and S3
    combo = torch.randint(0, 10, (B, 1))
    ctr = torch.randint(1, 6, (B, 1))
    info = torch.cat([combo, ctr], dim=-1)

    # random blocks: canonical + random orientation, plus some nulls
    block = torch.randint(0, NULL_ID, (B, 3))
    block = torch.where(torch.rand(B, 3) < 0.2, NULL_ID, block)

    with torch.no_grad():
        y0 = net(board, info, block)    # (B,192)
        y0 = y0.view(B, 3, 64)

        perms = list(itertools.permutations([0, 1, 2]))  # 6 perms
        perms = [torch.tensor(p, dtype=torch.long) for p in perms]

        worst = 0.0
        worst_case = None

        for g in range(8):
            board_g = transform_board_Z2(board, spatial_idx, g)

            for perm in perms:
                block_gp = transform_blocks_D4_S3(block, g=g, perm=perm)

                y_pred = net(board_g, info, block_gp).view(B, 3, 64)
                y_gt   = transform_logits_D4_S3(y0, spatial_idx, g=g, perm=perm)

                err = (y_pred - y_gt).abs().max().item()
                if err > worst:
                    worst = err
                    worst_case = (g, perm.tolist())

        print(f"[Policy] Max error over D4xS3 (48 cases): {worst:.2e} at (g,perm)={worst_case}")
        assert worst < atol, f"[Policy] failed D4xS3 equivariance: max_err={worst:.2e} >= {atol:.1e}"

    print("[Policy] Success: D4xS3-equivariant within tolerance.")

def test_value_invariance(net, atol: float = 1e-5):
    net.eval()
    torch.manual_seed(42)

    B, H, W = 8, 8, 8
    spatial_idx = calc_spatial_idx(G.mats, H, W)

    board = torch.randn(B, 64).double()

    combo = torch.randint(0, 10, (B, 1))
    ctr = torch.randint(1, 6, (B, 1))
    info = torch.cat([combo, ctr], dim=-1)

    block = torch.randint(0, NULL_ID, (B, 3), dtype=torch.long)
    block = torch.where(torch.rand(B, 3) < 0.2, NULL_ID, block)

    with torch.no_grad():
        v0 = net(board, info, block)  # (B,K)

        perms = [torch.tensor(p, dtype=torch.long) for p in itertools.permutations([0, 1, 2])]

        worst = 0.0
        worst_case = None

        for g in range(8):
            board_g = transform_board_Z2(board, spatial_idx, g)
            for perm in perms:
                block_gp = transform_blocks_D4_S3(block, g=g, perm=perm)
                v_pred = net(board_g, info, block_gp)

                err = (v_pred - v0).abs().max().item()
                if err > worst:
                    worst = err
                    worst_case = (g, perm.tolist())

        print(f"[Value] Max error over D4xS3 (48 cases): {worst:.2e} at (g,perm)={worst_case}")
        assert worst < atol, f"[Value] failed D4xS3 invariance: max_err={worst:.2e} >= {atol:.1e}"

    print("[Value] Success: D4xS3 invariant within tolerance.")

test_policy_equivariance(PolicyNet().double())
test_value_invariance(ValueNet().double())
