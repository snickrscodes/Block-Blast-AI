import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from .group import get_coord_map
from policy_math.value_scale import vscale, vscale_inv

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
        # x: (B, 3, D) float
        # valid: (B, 3) bool
        edge_mask = (valid[:, :, None] & valid[:, self.neighbors]).to(dtype=x.dtype) # (B, 3, 2)
        deg = edge_mask.sum(-1).clamp_min(1.0)[:, :, None] # (B, 3, 1)
        edge_mask = edge_mask[:, :, :, None]
        v = valid[:, :, None].to(dtype=x.dtype)
        x = x * v # zero out null nodes

        for _ in range(self.rounds):
            x_self = x.unsqueeze(2).expand(-1, -1, 2, -1) # (B, 3, 2, D)
            x_nei  = x[:, self.neighbors] # (B, 3, 2, D)
            msg = self.edge_mlp(torch.cat([x_self, x_nei], dim=-1)) * edge_mask # (B, 3, 2, D)
            agg = msg.sum(2) / deg # (B, 3, D)
            dx = self.node_mlp(torch.cat([x, agg], dim=-1))
            x = (x + dx) * v
        return x

class GN(nn.GroupNorm):
    def __init__(self, num_channels: int, num_groups: int | None = None, affine: bool = True):
        if num_groups is None:
            for g in (8, 4, 2, 1):
                if num_channels % g == 0:
                    num_groups = g
                    break
        super().__init__(num_groups, num_channels, affine=affine)

class ResBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, k: int, d_emb: int, dilation: int = 1):
        super().__init__()
        padding = (k // 2) * dilation
        self.norm = GN(ch_in)
        self.film = nn.Linear(d_emb, 2 * ch_in)
        self.block = nn.Sequential(
            nn.SiLU(inplace=True), 
            nn.Conv2d(ch_in, ch_out, k, padding=padding, dilation=dilation, bias=False), 
            GN(ch_out),
            nn.SiLU(inplace=True), 
            nn.Conv2d(ch_out, ch_out, k, padding=padding, dilation=dilation)
        )
        self.use_proj = ch_in != ch_out
        if self.use_proj:
            self.proj = nn.Conv2d(ch_in, ch_out, 1, bias=False)
        nn.init.normal_(self.film.weight, std=1e-3)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # x: (N, C, H, W)
        # emb: (N, D)
        y = self.norm(x)
        a, b = self.film(emb).chunk(2, dim=1) # (N, C)
        y = y * (1.0 + a[:, :, None, None]) + b[:, :, None, None]
        y = self.block(y) # (N, C, H, W)
        if self.use_proj:
            x0 = self.proj(x)
        else:
            x0 = x
        return y + x0
    
class SelfAttention(nn.Module):
    def __init__(self, ch_in: int, d_in: int, n_heads: int = 1):
        super().__init__()
        assert ch_in % n_heads == 0, f"channels must be divisible by num heads, got {ch_in} channels and {n_heads} heads"
        d_head = ch_in // n_heads
        self.scale = d_head ** (-0.5)
        self.qkv_head = nn.Conv2d(ch_in, 3 * ch_in, 1)
        self.out_head = nn.Conv2d(ch_in, ch_in, 1)
        self.norm = GN(ch_in)
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

class BlockBoardCrossAttn(nn.Module):
    def __init__(self, ch_in: int, d_head: int, d_image: int, d_emb: int, n_heads: int = 1):
        super().__init__()
        self.scale = d_head ** (-0.5)
        self.use_multi_head = n_heads > 1
        self.d_total = d_head * n_heads

        # queries from block embeddings, keys/values from board
        self.q_head = nn.Linear(d_emb, self.d_total)
        self.kv_head = nn.Conv2d(ch_in, 2 * self.d_total, 1)
        self.out_head = nn.Linear(self.d_total, d_emb)
        self.norm = nn.LayerNorm(d_emb)
        self.out_norm = nn.LayerNorm(d_emb)
        self.out_mp = TriadMP(d_emb, rounds=1)

        if self.use_multi_head:
            self.qdims = (-1, 3, n_heads, d_head)
            self.kvdims = (-1, 2, n_heads, d_head, d_image)
        else:
            self.qdims = (-1, 3, d_head)
            self.kvdims = (-1, 2, d_head, d_image)

    def forward(self, block_emb: Tensor, board: Tensor, bmask: Tensor) -> Tensor:
        # block_emb: (B, 3, D)
        # board: (B, C, H, W)
        x = self.norm(block_emb) * bmask.unsqueeze(-1)
        q = self.q_head(x).view(*self.qdims) # (B, 3, N, D_H)
        k, v = self.kv_head(board).view(*self.kvdims).unbind(1) # (B, N, D_H, HW)

        if self.use_multi_head:
            scores = torch.matmul(q.transpose(1, 2), k * self.scale) # (B, N, 3, HW)
            weights = torch.softmax(scores, dim=-1) # (B, N, 3, HW)
            attn = torch.matmul(weights, v.transpose(2, 3)) # (B, N, 3, D_H)
            attn = attn.transpose(1, 2).contiguous() # (B, 3, N, D_H)
            attn = attn.view(block_emb.shape[0], 3, self.d_total)
        else:
            scores = torch.bmm(q, k * self.scale) # (B, 3, HW)
            weights = torch.softmax(scores, dim=-1)
            attn = torch.bmm(weights, v.transpose(1, 2))

        msk = bmask.unsqueeze(-1).to(block_emb.dtype)
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
        self.register_buffer("coord_map", get_coord_map(8, 8)) # (3, 8, 8)
        n_coord_ch = self.coord_map.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1+n_coord_ch, base_channels, 3, padding=1, bias=False),
            GN(base_channels),
            nn.SiLU(inplace=True),
        )
        self.block_emb = nn.Embedding(n_blocks + 1, d_emb, padding_idx=null_block)
        self.block_mp = TriadMP(d_emb, rounds=1)

        self.ctr_emb = nn.Embedding(5, d_ctr_emb)
        self.info_mlp = nn.Sequential(
            nn.Linear(d_ctr_emb + 1, d_emb),
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb),
        )

        en_ch = (base_channels,) + encoder_channels
        ba_ch = (en_ch[-1],) + backbone_channels
        self.encoder = nn.ModuleList([
            ResBlock(en_ch[i], en_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(en_ch)-1)
        ])
        self.mid_res = ResBlock(en_ch[-1], en_ch[-1], 3, d_emb=d_emb)
        self.self_attn0 = SelfAttention(en_ch[-1], 64, n_heads=4)

        self.backbone = nn.ModuleList([
            ResBlock(ba_ch[i], ba_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(ba_ch)-1)
        ])

        self.cross_attns = nn.ModuleList([
            BlockBoardCrossAttn(ba_ch[i], d_emb // 4, 64, d_emb, n_heads=4)
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

        self.self_attn1 = SelfAttention(ch, 64, n_heads=4)
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
        x = torch.cat([x, self.coord_map.unsqueeze(0).expand(B, -1, -1, -1)], dim=1) # (B, 4, 8, 8)
        
        combo = info[:, 0].float()
        ctr = info[:, 1].long() - 1

        x = self.stem(x)

        bmask = (block != self.null_block)
        n_valid = bmask.sum(1, keepdim=True).clamp_min(1).float()
        be = self.block_emb(block) # (B, 3, D)
        be = self.block_mp(be, bmask) # null blocks masked

        ie = self.info_mlp(torch.cat([self.ctr_emb(ctr), combo.log1p().unsqueeze(-1)], dim=-1))
        ctx = self.ctx_mlps[0](torch.cat([be.sum(1) / n_valid, ie], dim=-1)) # (B, D)

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

        x = x.view(B, -1, 64).transpose(1, 2) # (B, 64, C)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1) # (B, 3, 64, C)
        be = be.unsqueeze(2).expand(-1, -1, 64, -1) # (B, 3, 64, D)
        ie = ie.view(B, 1, 1, -1).expand(-1, 3, 64, -1)
        x = self.score_mlp(torch.cat([x, be, ie], dim=-1))
        return x.view(B, -1)
    
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
        self.register_buffer("coord_map", get_coord_map(8, 8)) # (3, 8, 8)
        n_coord_ch = self.coord_map.shape[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1+n_coord_ch, base_channels, 3, padding=1, bias=False),
            GN(base_channels),
            nn.SiLU(inplace=True),
        )

        self.block_emb = nn.Embedding(n_blocks + 1, d_emb, padding_idx=null_block)
        self.block_mp = TriadMP(d_emb, rounds=1)
        self.ctr_emb = nn.Embedding(5, d_ctr_emb)
        self.info_mlp = nn.Sequential(
            nn.Linear(d_ctr_emb + 1, d_emb),
            nn.SiLU(inplace=True),
            nn.Linear(d_emb, d_emb),
        )

        en_ch = (base_channels,) + encoder_channels
        ba_ch = (en_ch[-1],) + backbone_channels
        self.encoder = nn.ModuleList([
            ResBlock(en_ch[i], en_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(en_ch)-1)
        ])
        self.mid_res = ResBlock(en_ch[-1], en_ch[-1], 3, d_emb=d_emb)
        self.self_attn0 = SelfAttention(en_ch[-1], 64, n_heads=4)

        self.backbone = nn.ModuleList([
            ResBlock(ba_ch[i], ba_ch[i+1], 3, d_emb=d_emb) 
            for i in range(len(ba_ch)-1)
        ])

        self.cross_attns = nn.ModuleList([
            BlockBoardCrossAttn(ba_ch[i], d_emb // 4, 64, d_emb, n_heads=4)
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

        self.self_attn1 = SelfAttention(ch, 64, n_heads=4)
        n_in = ch + 2 * d_emb
        n_hid = n_in // 2
        self.pos_mlp = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, n_hid),
        )
        n_in = n_hid
        n_hid = n_in // 2
        self.v_head = nn.Sequential(
            nn.Linear(n_in, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, n_hid),
            nn.SiLU(inplace=True),
            nn.Linear(n_hid, 1, bias=False)
        )

    def forward(self, board: Tensor, info: Tensor, block: Tensor) -> Tensor:
        # board: (B, 64) float
        # info: (B, 2) int64
        # block: (B, 3) int64
        B = board.shape[0]
        x = board.view(B, 1, 8, 8)
        x = torch.cat([x, self.coord_map.unsqueeze(0).expand(B, -1, -1, -1)], dim=1) # (B, 4, 8, 8)

        combo = info[:, 0].float()
        ctr = info[:, 1].long() - 1

        x = self.stem(x)

        bmask = (block != self.null_block)
        n_valid = bmask.sum(1, keepdim=True).clamp_min(1).float()
        be = self.block_emb(block) # (B, 3, D)
        be = self.block_mp(be, bmask) # null blocks masked

        ie = self.info_mlp(torch.cat([self.ctr_emb(ctr), combo.log1p().unsqueeze(-1)], dim=-1))
        ctx = self.ctx_mlps[0](torch.cat([be.sum(1) / n_valid, ie], dim=-1)) # (B, D)

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

        x = x.view(B, -1, 64).transpose(1, 2) # (B, 64, C)
        be = (be.sum(1) / n_valid).unsqueeze(1).expand(-1, 64, -1)
        ie = ie.unsqueeze(1).expand(-1, 64, -1)
        x = self.pos_mlp(torch.cat([x, be, ie], dim=-1)).mean(1) # (B, H)
        return self.v_head(x).squeeze(-1)
