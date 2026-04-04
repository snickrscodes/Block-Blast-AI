import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
import pygame
from pygame import Rect
from functools import lru_cache
import bbengine.bbengine as bb
from utils import load_models, unpack_obs
from core import make_models
from model.cnn import vscale_inv

FONT_PATH = r"C:\Users\PC\Desktop\python_projects\block_blast\assets\LeagueSpartan-Bold.otf"

BG_COLOR = (52, 74, 131)
BOARD_COLOR = (33, 36, 66)
BOARD_OUTLINE = (41, 56, 101)
GRID_COLOR = (24, 28, 57)
SCORE_COLOR = (255, 255, 255)
FRAME_COLOR = (38, 42, 52)

ENV_ASPECT = 16 / 9 # 16:9 (h:w)
MIN_ENV_W = 72
MIN_ENV_H = 128
FPS = 60
WIDTH = 384
HEIGHT = 683

B0_SHIFT = bb.B0_SHIFT
B1_SHIFT = bb.B1_SHIFT
B2_SHIFT = bb.B2_SHIFT
BLOCK_MASK = bb.BLOCK_MASK
NULL_BLOCK = bb.NULL_BLOCK
BLOCKS = bb.blocks.tolist()

# calculate centers (row and col) in doubled coordinates
def calc_centers2() -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    r2 = []
    c2 = []

    for b in BLOCKS:
        x = b
        maxr = 0
        maxc = 0
        minr = 8
        minc = 8

        while x:
            lsb = x & -x
            p = lsb.bit_length() - 1
            x ^= lsb

            r = p // 8
            c = p % 8

            if r > maxr: maxr = r
            if c > maxc: maxc = c
            if r < minr: minr = r
            if c < minc: minc = c
        
        r2.append(minr + maxr)
        c2.append(minc + maxc)

    return tuple(r2), tuple(c2)

CENTER_R2, CENTER_C2 = calc_centers2()

@lru_cache(maxsize=4)
def get_font(size):
    return pygame.font.Font(FONT_PATH, size)

def make_window(title: str = "Block Blast"):
    pygame.init()
    pygame.font.init()
    pygame.display.set_caption(title)
    screen = pygame.display.set_mode((WIDTH, HEIGHT), flags=pygame.RESIZABLE)
    clock = pygame.time.Clock()
    return screen, clock

def draw_block(screen: pygame.Surface, bid: int, x_pos: int, y_pos: int, w: int, stroke: int):
    cr = CENTER_R2[bid]
    cc = CENTER_C2[bid]
    b = BLOCKS[bid]

    while b:
        lsb = b & -b
        pos = lsb.bit_length() - 1

        r = 2 - (pos >> 3) + cr / 2
        c = 2 - (pos & 7) + cc / 2

        bx = int(c * w + x_pos)
        by = int(r * w + y_pos)
        
        a = Rect(bx, by, w, w)
        screen.fill((255, 255, 255), a)
        pygame.draw.rect(screen, BOARD_OUTLINE, a, width=stroke)

        b ^= lsb

def render_env(screen: pygame.Surface, board_: int, meta_: int, score_: int, rect: Rect):
    screen.fill(BG_COLOR, rect)
    width = int(min(rect.w, rect.h / ENV_ASPECT))
    height = int(width * ENV_ASPECT)
    ox = rect.x
    oy = rect.y

    cell_size = width // 9
    board_size = 8 * cell_size
    board_outline_w = width // 64
    margin = (cell_size - 2 * board_outline_w) // 2
    grid_stroke = board_outline_w // 2

    # draw background
    board_x = margin + board_outline_w + ox
    board_y = height // 6 + board_outline_w + oy
    board_rect = Rect(board_x, board_y, board_size, board_size)
    board_rect = board_rect.inflate(2 * board_outline_w, 2 * board_outline_w)
    screen.fill(BOARD_COLOR, board_rect)
    pygame.draw.rect(screen, BOARD_OUTLINE, board_rect, width=board_outline_w)

    # draw score (and high score)
    font_size = cell_size
    score_margin = height // 64

    font = get_font(font_size)
    score_surf = font.render(str(score_), True, SCORE_COLOR)
    score_rect = score_surf.get_rect()
    score_rect.center = (
        board_x + board_size // 2,
        board_y - board_outline_w - score_margin - font_size // 2
    )
    screen.blit(score_surf, score_rect)

    # draw grid
    for i in range(8 + 1):
        px = i * cell_size + board_x
        py = i * cell_size + board_y
        pygame.draw.line(
            screen,
            GRID_COLOR,
            (px, board_y),
            (px, board_y + board_size),
            width=grid_stroke
        )
        pygame.draw.line(
            screen, GRID_COLOR, 
            (board_x, py), (board_x + board_size, py),
            width=grid_stroke
        )

    # draw board
    b = board_
    while b:
        lsb = b & -b
        pos = lsb.bit_length() - 1

        r = 7 - (pos >> 3)
        c = 7 - (pos & 7)

        cx = c * cell_size + board_x
        cy = r * cell_size + board_y
        
        a = Rect(cx, cy, cell_size, cell_size)
        a = a.inflate(-grid_stroke, -grid_stroke)
        screen.fill((255, 255, 255), a)

        b ^= lsb

    # draw blocks
    m = meta_
    bid0 = (m >> B0_SHIFT) & BLOCK_MASK
    bid1 = (m >> B1_SHIFT) & BLOCK_MASK
    bid2 = (m >> B2_SHIFT) & BLOCK_MASK

    block_size = cell_size // 2
    block_outline_w = board_outline_w // 4

    block_x = board_x + block_size // 2
    block_y = board_y + board_size + cell_size * 5 // 4
    block_dist = (board_size - block_size) // 3

    if bid0 != NULL_BLOCK:
        draw_block(screen, bid0, block_x, block_y, block_size, block_outline_w)

    if bid1 != NULL_BLOCK:
        draw_block(screen, bid1, block_x + block_dist, block_y, block_size, block_outline_w)

    if bid2 != NULL_BLOCK:
        draw_block(screen, bid2, block_x + 2 * block_dist, block_y, block_size, block_outline_w)

def grid_layout(container: Rect, n: int, gap: int = 8) -> List[Rect]:
    avail = container.inflate(-2 * gap, -2 * gap)
    best = None # (tile_w, tile_h, rows, cols)

    # try all column counts, pick the one that maximizes tile area
    for cols in range(1, n + 1):
        rows = (n + cols - 1) // cols

        total_gap_w = gap * (cols - 1)
        total_gap_h = gap * (rows - 1)

        slot_w = (avail.w - total_gap_w) / cols
        slot_h = (avail.h - total_gap_h) / rows

        tile_w = int(min(slot_w, slot_h / ENV_ASPECT))
        tile_h = int(tile_w * ENV_ASPECT)

        if tile_w < MIN_ENV_W or tile_h < MIN_ENV_H:
            continue

        score = tile_w * tile_h # maximize area
        cand = (score, tile_w, tile_h, rows, cols)

        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        return []
    
    _, tile_w, tile_h, rows, cols = best

    # compute the grid's total footprint (to center it inside avail)
    grid_w = cols * tile_w + gap * (cols - 1)
    grid_h = rows * tile_h + gap * (rows - 1)

    start_x = avail.x + (avail.w - grid_w) // 2
    start_y = avail.y + (avail.h - grid_h) // 2

    rects = []
    for i in range(n):
        r = i // cols
        c = i % cols
        x = start_x + c * (tile_w + gap)
        y = start_y + r * (tile_h + gap)
        rects.append(Rect(x, y, tile_w, tile_h))

    return rects

def main() -> int:
    screen, clock = make_window()
    e = bb.BatchEnv(1)
    B = e.size()
    layout = grid_layout(screen.get_rect(), B, gap=0)

    rank = None
    if rank is not None:
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu") # fallback

    P, V = make_models(rank)
    cfg = load_models(
        "C:/Users/PC/Desktop/python_projects/block_blast/chkpts", 
        "chkpt18", 
        P, V, device
    )

    search_cfg = bb.SearchConfig()
    search_cfg.beam_width = cfg.P_beam_width
    search_cfg.per_parent_top_m = cfg.P_per_parent_top_m
    search_cfg.max_eval_P = cfg.max_bs_inference_P
    search_cfg.max_eval_V = cfg.max_bs_inference_V
    search_cfg.gamma = cfg.V_gamma
    search_cfg.teacher_tau = cfg.P_teacher_tau

    plan = bb.BeamSearch(search_cfg, cfg.seed)

    def policy_logits_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return P(b, i, k).cpu()
    
    def value_fn(boards_u64: Tensor, metas_u64: Tensor) -> Tensor:
        b, i, k = unpack_obs(boards_u64, metas_u64, device)
        return vscale_inv(V(b, i, k)).cpu()

    all_idx = torch.arange(B)

    dones = torch.zeros((B,), dtype=torch.bool)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.type == pygame.VIDEORESIZE:
                layout = grid_layout(screen.get_rect(), e.size(), gap=8)

        # chance steps and resets
        need_blocks = e.need_blocks()
        if need_blocks.any():
            e.rand_blocks_indices(need_blocks.nonzero(as_tuple=False).squeeze(-1))

        dones = e.done()
        if dones.any():
            ix = dones.nonzero(as_tuple=False).squeeze(-1)
            e.reset_indices(ix)
            e.rand_blocks_indices(ix)

        # render
        rect = screen.get_rect()
        screen.fill(FRAME_COLOR, rect)

        b, m = e.boards(), e.metas()
        sc = e.score()
        for board, meta, score, r in zip(b.tolist(), m.tolist(), sc.tolist(), layout):
            render_env(screen, board, meta, score, r)

        pygame.display.flip()

        pi_beh = plan.search_batch(b, m, policy_logits_fn, value_fn, use_noise=False) # (M, 192)
        
        # msk = bb.legal_mask_batch(b, m)
        # logits = torch.ones_like(msk, dtype=torch.float)
        # logits = policy_logits_fn(b, m)
        # logits.masked_fill_(~msk, -1e9)
        # pi_beh = logits.softmax(-1)
        
        a = pi_beh.argmax(-1)

        # env step
        r = e.step_indices(all_idx, a)
        dones = e.done()

        # time
        clock.tick(FPS)

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
