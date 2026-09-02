import torch
import torch.nn as nn
import torch.nn.functional as F
import bbengine.bbengine as bb
from torch import Tensor
from typing import Tuple
from .group import DihedralGroup, calc_spatial_idx, orbit_idx_d4, get_coord_map

G = DihedralGroup(4)
K = 3
GSIZE = G.order
PID2G = torch.tensor(bb.pid2g.tolist(), dtype=torch.int64)
PID2B = torch.tensor(bb.pid2b.tolist(), dtype=torch.int64)
POSE_OFF = torch.tensor(bb.pose_off.tolist(), dtype=torch.int64)
G2POSE = torch.tensor(bb.g2pose.tolist(), dtype=torch.int64)

# convention is right-regular, left actions
# lifts signal on z2 to p4m, (z2, +) * d4 = p4m
class P4MLiftConv2d(nn.Module):
    def __init__(self, 
        ch_in: int, ch_out: int, k: int, 
        groups: int = 1, bias: bool = True, dilation: int = 1
    ):
        super().__init__()
        assert k % 2 == 1, f"kernel size must be odd, got kernel size {k}"

        self.k = k
        self.padding = (k // 2) * dilation
        self.groups = groups
        self.dilation = dilation
        self.use_bias = bias
        self.ch_out = ch_out
        self.ch_in = ch_in // groups
        
        kernel_idx = calc_spatial_idx(G.mats, k, k)[G.inverse_map] # (G_out, k2)
        self.register_buffer("w_idx", kernel_idx)
        
        self.weight = nn.Parameter(torch.zeros(self.ch_out, self.ch_in, k * k))
        self.bias = nn.Parameter(torch.zeros(self.ch_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        u = (1 / (self.ch_in * self.k * self.k)) ** 0.5
        nn.init.uniform_(self.weight, -u, u)
        if self.use_bias:
            nn.init.uniform_(self.bias, -u, u)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in, H, W)
        w_t = self.weight[..., self.w_idx].transpose(1, 2) # (C_out, G_out, C_in, k2)
        w_t = w_t.reshape(-1, self.ch_in, self.k, self.k) # (C_out * G_out, C_in, k, k)

        bias = None
        if self.use_bias:
            bias = self.bias.repeat_interleave(GSIZE, 0)
        return F.conv2d(
            x, w_t, bias=bias, padding=self.padding, 
            dilation=self.dilation, groups=self.groups
        ) # (N, C_out * G_out, H, W)
    
# operates on signal in regular representation of d4
class P4MConv2d(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int, 
        groups: int = 1, bias: bool = True, dilation: int = 1
    ):
        super().__init__()
        assert k % 2 == 1, f"kernel size must be odd, got kernel size {k}"

        self.k = k
        self.padding = (k // 2) * dilation
        self.groups = groups
        self.dilation = dilation
        self.use_bias = bias
        self.ch_out = ch_out
        self.ch_in = ch_in // groups

        spatial_idx = calc_spatial_idx(G.mats, k, k)[G.inverse_map] # (G_in, k2)
        group_idx = G.product_table[:, G.inverse_map] # (G_out, G_in)
        kernel_idx = (group_idx[..., None] * (k * k) + spatial_idx).view(GSIZE, -1) # (G_out, G_in*k2)

        self.register_buffer("w_idx", kernel_idx)
        self.weight = nn.Parameter(torch.zeros(self.ch_out, self.ch_in, GSIZE * k * k))
        self.bias = nn.Parameter(torch.zeros(self.ch_out)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        u = (self.ch_in * GSIZE * self.k * self.k) ** -0.5
        nn.init.uniform_(self.weight, -u, u)
        if self.use_bias:
            nn.init.uniform_(self.bias, -u, u)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C_in * G_in, H, W)
        w_t = self.weight[..., self.w_idx].transpose(1, 2) # (C_out, G_out, C_in, G_in * k2)
        w_t = w_t.reshape(self.ch_out * GSIZE, -1, self.k, self.k) # (C_out * G_out, C_in * G_in, k, k)

        bias = None
        if self.use_bias:
            bias = self.bias.repeat_interleave(GSIZE, 0)
        return F.conv2d(
            x, w_t, bias=bias, padding=self.padding,
            dilation=self.dilation, groups=self.groups
        ) # (N, C_out * G_out, H, W)

# poses form coset space G / H_b (H_b = stabilizer subgroup)
# enbeds poses and lifts to G-indexed features via the induced rep
# (right-H_b-invariant functions on G with left-regular group action)
class P4MEmbedding(nn.Module):
    def __init__(self, n_pose_total: int, d_emb: int, null_pid: int):
        super().__init__()
        self.d_emb = d_emb
        pid2gidx = G.product_table[:, PID2G].T # (N, G)
        pid2pose = G2POSE[PID2B.unsqueeze(-1), pid2gidx] # (N, G)
        pose2pid = POSE_OFF[PID2B].unsqueeze(-1) + pid2pose # (N, G)
        self.register_buffer("pose2pid", pose2pid)
        self.pose_emb = nn.Embedding(n_pose_total + 1, d_emb, padding_idx=null_pid)

    def forward(self, x: Tensor):
        return self.pose_emb(self.pose2pid[x])
    
class P4MGroupNorm(nn.Module):
    def __init__(self, channels: int, num_groups: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_groups = num_groups
        assert channels % num_groups == 0, "channels must be divisible by num_groups"
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta  = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        N, CG, H, W = x.shape
        C = CG // GSIZE
        x = x.view(N, self.num_groups, C // self.num_groups, GSIZE, H, W)
        var, mean = torch.var_mean(x, dim=(2, 3, 4, 5), correction=0, keepdim=True)
        x = (x - mean) * (var + self.eps).rsqrt()

        x = x.view(N, C, GSIZE, H, W)
        if self.affine: # invariant over spatial and group dims
            x = x * self.gamma.view(1, C, 1, 1, 1) + self.beta.view(1, C, 1, 1, 1)
        return x.view(N, CG, H, W)

class GroupNorm(P4MGroupNorm):
    def __init__(self, num_channels: int, num_groups: int | None = None):
        if num_groups is None:
            for g in (8, 4, 2, 1):
                if num_channels % g == 0:
                    num_groups = g
                    break
        super().__init__(num_channels, num_groups)

class GroupNormRegular(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int | None = None):
        if num_groups is None:
            for g in (8, 4, 2, 1):
                if num_channels % g == 0:
                    num_groups = g
                    break
        super().__init__(num_groups, num_channels)
    
class P4MResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int, d_emb: int, dilation: int = 1):
        super().__init__()
        self.norm = GroupNorm(ch_in)
        self.film = nn.Linear(d_emb, 2 * ch_in)
        self.block = nn.Sequential(
            nn.SiLU(inplace=True), 
            P4MConv2d(ch_in, ch_out, k, dilation=dilation, bias=False), 
            GroupNorm(ch_out),
            nn.SiLU(inplace=True), 
            P4MConv2d(ch_out, ch_out, k, dilation=dilation)
        )

        self.use_proj = ch_in != ch_out
        if self.use_proj:
            self.proj = P4MConv2d(ch_in, ch_out, 1, bias=False)
        nn.init.normal_(self.film.weight, std=1e-3)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # x: (N, CG, H, W)
        # emb: (N, G, D)
        N, _, H, W = x.shape
        y = self.norm(x).view(N, -1, GSIZE, H, W) # (N, C, G, H, W)
        a, b = self.film(emb).transpose(1, 2).chunk(2, dim=1) # (N, C, G)
        y = y * (1.0 + a[:, :, :, None, None]) + b[:, :, :, None, None]
        y = self.block(y.view(N, -1, H, W)) # (N, CG, H, W)
        if self.use_proj:
            x0 = self.proj(x)
        else:
            x0 = x
        return y + x0
    
class P4MSelfAttn(nn.Module):
    def __init__(self, ch_in: int, d_in: int, n_heads: int = 1):
        super().__init__()
        assert ch_in % n_heads == 0, f"channels must be divisible by num heads, got {ch_in} channels and {n_heads} heads"
        d_head = ch_in * GSIZE // n_heads
        self.scale = d_head ** (-0.5)
        self.qkv_head = P4MConv2d(ch_in, 3 * ch_in, 1)
        self.out_head = P4MConv2d(ch_in, ch_in, 1)
        self.norm = GroupNorm(ch_in)
        if n_heads > 1:
            self.dims = (-1, 3, n_heads, d_head, d_in)
        else:
            self.dims = (-1, 3, d_head, d_in)

    def forward(self, x: Tensor):
        # x: (N, CG, H, W)
        y = self.norm(x)
        q, k, v = self.qkv_head(y).view(*self.dims).unbind(1)

        scores = torch.matmul(q.transpose(-2, -1), k) * self.scale
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(v, weights.transpose(-2, -1)).reshape_as(x)
        x = x + self.out_head(attn)
        return x

class TriadMP(nn.Module):
    def __init__(self, d_emb: int, rounds: int = 2):
        super().__init__()
        # edge message: f(x_i, x_j) -> msg_ij
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb),
        )

        # node update: g(x_i, agg_i) -> dx_i
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb),
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb),
        )

        self.rounds = rounds

        # directed edges of K3
        self.register_buffer("neighbors", torch.tensor([[1,2],[0,2],[0,1]], dtype=torch.int64)) # (3, 2)

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        # x: (B, 3, G, D) float
        # valid: (B, 3) bool
        edge_mask = (valid[:, :, None] & valid[:, self.neighbors]).to(dtype=x.dtype) # (B, 3, 2)
        deg = edge_mask.sum(-1).clamp_min(1.0)[:, :, None, None] # (B, 3, 1, 1)
        edge_mask = edge_mask[:, :, :, None, None]
        v = valid[:, :, None, None].to(dtype=x.dtype)
        x = x * v # zero out null nodes

        for _ in range(self.rounds):
            x_self = x.unsqueeze(2).expand(-1, -1, 2, -1, -1) # (B, 3, 2, G, D)
            x_nei  = x[:, self.neighbors] # (B, 3, 2, G, D)
            msg = self.edge_mlp(torch.cat([x_self, x_nei], dim=-1)) * edge_mask # (B, 3, 2, G, D)
            agg = msg.sum(2) / deg # (B, 3, G, D)
            dx = self.node_mlp(torch.cat([x, agg], dim=-1))
            x = (x + dx) * v
        return x
    
class P4MBlockBoardCrossAttn(nn.Module):
    def __init__(self, ch_in: int, d_head: int, d_image: int, d_emb: int, n_heads: int = 1):
        super().__init__()
        self.scale = d_head ** (-0.5)
        self.use_multi_head = n_heads > 1
        self.d_total = d_head * n_heads

        # queries from block embeddings, keys/values from board
        self.q_head = nn.Linear(d_emb, self.d_total)
        self.kv_head = P4MConv2d(ch_in, 2 * self.d_total, 1)
        self.out_head = nn.Linear(self.d_total, d_emb)
        self.norm = nn.LayerNorm(d_emb)
        self.out_norm = nn.LayerNorm(d_emb)
        self.out_mp = TriadMP(d_emb, rounds=1)

        if self.use_multi_head:
            self.qdims = (-1, 3 * GSIZE, n_heads, d_head)
            self.kvdims = (-1, 2, n_heads, d_head, GSIZE * d_image)
        else:
            self.qdims = (-1, 3 * GSIZE, d_head)
            self.kvdims = (-1, 2, d_head, GSIZE * d_image)

    def forward(self, block_emb: Tensor, board: Tensor, bmask: Tensor) -> Tensor:
        # block_emb: (B, 3, G, D)
        # board: (B, CG, H, W)
        msk = bmask[:, :, None, None]
        x = self.norm(block_emb) * msk
        q = self.q_head(x).view(*self.qdims) # (B, 3G, N, D_H)
        k, v = self.kv_head(board).view(*self.kvdims).unbind(1) # (B, N, D_H, GHW)
        v = v.transpose(-2, -1) # (B, N, GHW, D_H) or (B, GHW, D_H)

        if self.use_multi_head:
            weights = torch.softmax(torch.matmul(q.transpose(1, 2), k * self.scale), dim=-1) # (B, N, 3G, GHW)
            attn = torch.matmul(weights, v) # (B, N, 3G, D_H)
            attn = attn.transpose(1, 2).contiguous() # (B, 3G, N, D_H)
            attn = attn.view(block_emb.shape[0], 3, GSIZE, self.d_total)
        else:
            weights = torch.softmax(torch.bmm(q, k * self.scale), dim=-1) # (B, 3G, GHW)
            attn = torch.bmm(weights, v) # (B, 3G, D_H)

        z = (block_emb + self.out_head(attn)) * msk
        z_norm = self.out_norm(z)
        return (z + self.out_mp(z_norm, bmask) - z_norm) * msk

class PolicyNet(nn.Module):
    def __init__(self,
        n_blocks: int = 41,
        null_block: int = 41,
        d_emb: int = 128,
        d_ctr_emb: int = 32,
        base_channels: int = 32,
        encoder_channels: Tuple[int, ...] = (32, 64, 64, 64),
        backbone_channels: Tuple[int, ...] = (64, 64, 64, 64),
    ):
        super().__init__()
        self.null_block = null_block

        coord = get_coord_map(8, 8).view(-1, 64)
        n_coord_ch = coord.shape[0]
        idx = calc_spatial_idx(G.mats, 8, 8)[G.inverse_map]
        coord = coord[:, idx].view(3, -1, 8, 8) # (3, G, 8, 8)
        self.register_buffer("coord_map", coord)

        self.lift = nn.Sequential(
            P4MLiftConv2d(1, base_channels, 3, bias=False), 
            GroupNorm(base_channels), 
            nn.SiLU(inplace=True), 
        )
        self.coord_mix = nn.Sequential(
            P4MConv2d(base_channels+n_coord_ch, base_channels, 1, bias=False),
            GroupNorm(base_channels),
            nn.SiLU(inplace=True),
        )

        self.block_emb = P4MEmbedding(n_blocks, d_emb, null_block)
        self.block_mp = TriadMP(d_emb, rounds=1)

        # counter and combo live in trivial irrep
        self.ctr_emb = nn.Embedding(5, d_ctr_emb) 
        self.info_mlp = nn.Sequential(
            nn.Linear(d_ctr_emb + 1, d_emb),
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb)
        )

        en_ch = (base_channels,) + encoder_channels
        ba_ch = (en_ch[-1],) + backbone_channels
        self.encoder = nn.ModuleList([
            P4MResBlock(en_ch[i], en_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(en_ch)-1)
        ])
        self.mid_res = P4MResBlock(en_ch[-1], en_ch[-1], 3, d_emb=d_emb)
        self.self_attn0 = P4MSelfAttn(en_ch[-1], 64, n_heads=4)

        self.backbone = nn.ModuleList([
            P4MResBlock(ba_ch[i], ba_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(ba_ch)-1)
        ])

        self.cross_attns = nn.ModuleList([
            P4MBlockBoardCrossAttn(ba_ch[i], d_emb // 4, 64, d_emb, n_heads=4)
            for i in range(len(ba_ch))
        ])
        self.ctx_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_emb, d_emb, bias=False),
                    nn.LayerNorm(d_emb),
                    nn.SiLU(inplace=True),
                    nn.Linear(d_emb, d_emb),
                ) for _ in range(len(backbone_channels)+1)
            ]
        )
        ch = ba_ch[-1]

        self.self_attn1 = P4MSelfAttn(ch, 64, n_heads=4)
        n_in = ch + 2 * d_emb
        n_hid = n_in // 2
        self.score_mlp = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, 1, bias=False),
        )

    def forward(self, board: Tensor, info: Tensor, block: Tensor) -> Tensor:
        # board: (B, 64) float
        # info: (B, 2) int64
        # block: (B, 3) int64
        B = board.shape[0]
        x = board.view(B, 1, 8, 8)
        
        combo = info[:, 0].float()
        ctr = info[:, 1].long() - 1

        x = self.lift(x) # (B, C * G, 8, 8)
        x = x.view(B, -1, GSIZE, 8, 8)
        x = torch.cat([x, self.coord_map.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=1)
        x = x.view(B, -1, 8, 8) # (B, (C+3)G, 8, 8)
        x = self.coord_mix(x)

        bmask = (block != self.null_block)
        n_valid = bmask.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1).float()
        be = self.block_emb(block) # (B, 3, G, D)
        be = self.block_mp(be, bmask) # null blocks masked

        ie = self.info_mlp(torch.cat([self.ctr_emb(ctr), combo.log1p().unsqueeze(-1)], dim=-1)) # (B, D)
        ie = ie.unsqueeze(1).expand(-1, GSIZE, -1) # (B, G, D)
        ctx = self.ctx_mlps[0](torch.cat([be.sum(1) / n_valid, ie], dim=-1)) # (B, G, D)

        for layer in self.encoder:
            x = layer(x, ctx) # (B, CG, 8, 8)
        
        x = self.self_attn0(x)
        x = self.mid_res(x, ctx)

        for i in range(len(self.backbone)):
            be = self.cross_attns[i](be, x, bmask)
            ctx = self.ctx_mlps[i+1](torch.cat([be.sum(1) / n_valid, ie], dim=-1))
            x = self.backbone[i](x, ctx)

        x = self.self_attn1(x)
        be = self.cross_attns[-1](be, x, bmask) # null blocks masked

        x = x.view(B, -1, GSIZE, 64).permute(0, 2, 3, 1) # (B, G, 64, C)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1, -1) # (B, 3, G, 64, C)
        be = be.unsqueeze(3).expand(-1, -1, -1, 64, -1) # (B, 3, G, 64, D)
        ie = ie.view(B, 1, GSIZE, 1, -1).expand(-1, 3, -1, 64, -1)
        x = self.score_mlp(torch.cat([x, be, ie], dim=-1))
        return x.mean(2).view(B, -1) # (B, 192) # NOTE: invalid logits must be masked outside
    
class ValueNet(nn.Module):
    def __init__(self,
        n_blocks: int = 41,
        null_block: int = 41,
        d_emb: int = 128,
        d_ctr_emb: int = 32,
        base_channels: int = 32,
        encoder_channels: Tuple[int, ...] = (32, 64, 64, 64),
        backbone_channels: Tuple[int, ...] = (64, 64, 64, 64),
    ):
        super().__init__()
        self.null_block = null_block

        coord = get_coord_map(8, 8).view(-1, 64)
        n_coord_ch = coord.shape[0]
        idx = calc_spatial_idx(G.mats, 8, 8)[G.inverse_map]
        coord = coord[:, idx].view(3, -1, 8, 8) # (3, G, 8, 8)
        self.register_buffer("coord_map", coord)

        self.lift = nn.Sequential(
            P4MLiftConv2d(1, base_channels, 3, bias=False), 
            GroupNorm(base_channels), 
            nn.SiLU(inplace=True), 
        )
        self.coord_mix = nn.Sequential(
            P4MConv2d(base_channels+n_coord_ch, base_channels, 1, bias=False),
            GroupNorm(base_channels),
            nn.SiLU(inplace=True),
        )

        self.block_emb = P4MEmbedding(n_blocks, d_emb, null_block)
        self.block_mp = TriadMP(d_emb, rounds=1)

        # counter and combo live in trivial irrep
        self.ctr_emb = nn.Embedding(5, d_ctr_emb) 
        self.info_mlp = nn.Sequential(
            nn.Linear(d_ctr_emb + 1, d_emb), 
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb)
        )

        en_ch = (base_channels,) + encoder_channels
        ba_ch = (en_ch[-1],) + backbone_channels
        self.encoder = nn.ModuleList([
            P4MResBlock(en_ch[i], en_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(en_ch)-1)
        ])
        self.mid_res = P4MResBlock(en_ch[-1], en_ch[-1], 3, d_emb=d_emb)
        self.self_attn0 = P4MSelfAttn(en_ch[-1], 64, n_heads=4)

        self.backbone = nn.ModuleList([
            P4MResBlock(ba_ch[i], ba_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(ba_ch)-1)
        ])

        self.cross_attns = nn.ModuleList([
            P4MBlockBoardCrossAttn(ba_ch[i], d_emb // 4, 64, d_emb, n_heads=4)
            for i in range(len(ba_ch))
        ])
        self.ctx_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * d_emb, d_emb, bias=False),
                    nn.LayerNorm(d_emb),
                    nn.SiLU(inplace=True),
                    nn.Linear(d_emb, d_emb),
                ) for _ in range(len(backbone_channels)+1)
            ]
        )
        ch = ba_ch[-1]

        self.self_attn1 = P4MSelfAttn(ch, 64, n_heads=4)
        n_in = ch + 2 * d_emb
        n_hid = n_in // 2
        self.pos_mlp = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, n_hid),
        )

        align_idx = calc_spatial_idx(G.mats, 8, 8)  # (G, 64)
        self.register_buffer("align_idx", align_idx)

        # after canonicalization, a global D4 transform only permutes G-axis, 
        # so each G slice can be processed by an ordinary spatial network
        d_pos = n_hid # output dim of pos_mlp
        spatial_ch1 = 3 * d_emb // 4
        spatial_ch2 = d_emb // 2
        pool_ch = d_emb // 4
        phi_dim = d_emb
        value_dim = d_emb // 2

        self.value_spatial = nn.Sequential(
            nn.Conv2d(d_pos, spatial_ch1, 3, padding=1),
            GroupNormRegular(spatial_ch1),
            nn.SiLU(inplace=True),
            nn.Conv2d(spatial_ch1, spatial_ch2, 3, padding=2, dilation=2),
            GroupNormRegular(spatial_ch2),
            nn.SiLU(inplace=True),
        )
        self.value_spatial_score = nn.Sequential(
            nn.Conv2d(spatial_ch2, pool_ch, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(pool_ch, 1, 1),
        )

        self.value_phi = nn.Sequential(
            nn.Linear(spatial_ch2, phi_dim),
            nn.SiLU(inplace=True),
            nn.Linear(phi_dim, phi_dim),
            nn.SiLU(inplace=True),
        )
        self.v_head = nn.Sequential(
            nn.Linear(phi_dim, value_dim),
            nn.SiLU(inplace=True),
            nn.Linear(value_dim, 1, bias=False),
        )

    def forward(self, board: Tensor, info: Tensor, block: Tensor) -> Tensor:
        # board: (B, 64) float
        # info: (B, 2) int64
        # block: (B, 3) int64
        B = board.shape[0]
        x = board.view(B, 1, 8, 8)
        
        combo = info[:, 0].float()
        ctr = info[:, 1].long() - 1

        x = self.lift(x) # (B, C * G, 8, 8)
        x = x.view(B, -1, GSIZE, 8, 8)
        x = torch.cat([x, self.coord_map.unsqueeze(0).expand(B, -1, -1, -1, -1)], dim=1)
        x = x.view(B, -1, 8, 8) # (B, (C+3)G, 8, 8)
        x = self.coord_mix(x)

        bmask = (block != self.null_block)
        n_valid = bmask.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1).float()
        be = self.block_emb(block) # (B, 3, G, D)
        be = self.block_mp(be, bmask) # null blocks masked

        ie = self.info_mlp(torch.cat([self.ctr_emb(ctr), combo.log1p().unsqueeze(-1)], dim=-1)) # (B, D)
        ie = ie.unsqueeze(1).expand(-1, GSIZE, -1) # (B, G, D)
        ctx = self.ctx_mlps[0](torch.cat([be.sum(1) / n_valid, ie], dim=-1)) # (B, G, D)

        for layer in self.encoder:
            x = layer(x, ctx)
        
        x = self.self_attn0(x)
        x = self.mid_res(x, ctx)

        for i in range(len(self.backbone)):
            be = self.cross_attns[i](be, x, bmask)
            ctx = self.ctx_mlps[i+1](torch.cat([be.sum(1) / n_valid, ie], dim=-1))
            x = self.backbone[i](x, ctx)

        x = self.self_attn1(x)
        be = self.cross_attns[-1](be, x, bmask)

        x = x.view(B, -1, GSIZE, 64).permute(0, 2, 3, 1) # (B, G, 64, C)
        be = (be.sum(1) / n_valid).unsqueeze(2).expand(-1, -1, 64, -1) # (B, G, 64, D)
        ie = ie.unsqueeze(2).expand(-1, -1, 64, -1) # (B, G, 64, D)
        x = self.pos_mlp(torch.cat([x, be, ie], dim=-1)) # (B, G, 64, H)

        H = x.shape[-1]

        idx = self.align_idx[None, :, :, None].expand(B, GSIZE, 64, H)
        x = torch.gather(x, dim=2, index=idx) # (B, G, 64, H)
        x = x.view(B * GSIZE, 8, 8, H).permute(0, 3, 1, 2).contiguous() # (B*G, H, 8, 8)
        x = self.value_spatial(x) # (B*G, S, 8, 8)

        scores = self.value_spatial_score(x).view(-1, 1, 64) # (B*G, 1, 64)
        weights = torch.softmax(scores, dim=-1) # (B*G, 1, 64)
        features = x.view(B * GSIZE, -1, 64) # (B*G, S, 64)
        x = torch.bmm(features, weights.transpose(1, 2)).squeeze(-1) # (B*G, S)
        x = self.value_phi(x)

        return self.v_head(x.view(B, GSIZE, -1).mean(dim=1)).squeeze(-1)
