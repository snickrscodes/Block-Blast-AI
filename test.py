import ctypes
from ctypes import wintypes
import win32gui
import mss
import numpy as np
import time
from typing import List, Tuple
import bbengine.bbengine as bb
import torch
import torch.nn as nn
from torch import Tensor
from utils import load_models, unpack_obs
from core import make_models
from model.cnn import vscale_inv

MEASUREMENT_WIDTH = 576 # board width that measurements were recorded on
EMPTY_COL = 37.5282 # luminance of empty cell RGB(33, 36, 66)
BG_COL = 80.735 # luminance of background color RGB(57, 81, 148)
BOARD_LEFT = 36 # x position of left edge of board cells
BOARD_TOP = 184 # y position of top edge of board cells
CELL_SIZE = 62 # cell size in the board
GAP_SIZE = 2 # gap between cells
TRAY_GAP = 1 # gap between cells (in the tray)
BLOCK_TOP = 770 # y position of 5x5 container for each block
BLOCK_LEFT = (48, 216, 384) # x position of 5x5 container for each block
BLOCK_SIZE = 28 # block size in the tray
HALF_BLOCK_SIZE = 14

IDX = np.arange(63, -1, -1, dtype=np.uint64).reshape(8, 8)
BOARD_SHIFTS = 1 << IDX # for screen coords
BLOCK_SHIFTS = 1 << IDX[-5:, -5:]
BLOCKS = bb.blocks.tolist()
REF_POS = bb.ref_pos.tolist()

PICKUP_Y = 900 # y position to pick blocks up from
PICKUP_X = (118, 286, 454) # x position to pick up each block from
BIG_CENTER_Y = 696 # y position of center of block after picking up
AFFINE_X = 1.4 # ratio of block x velocity to mouse velocity
AFFINE_Y = 1.4 # ratio of block y velocity to mouse velocity
NULL_BLOCK = bb.NULL_BLOCK

user32 = ctypes.WinDLL("user32", use_last_error=True)
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
INPUT_MOUSE = 0
MOUSEEVENTF_LEFTDOWN   = 0x0002
MOUSEEVENTF_LEFTUP     = 0x0004
MOD_CONTROL = 0x0002
VK_Q = 0x51
HOTKEY_ID = 1

if not user32.RegisterHotKey(None, HOTKEY_ID, MOD_CONTROL, VK_Q):
    raise ctypes.WinError(ctypes.get_last_error())

def hotkey_pressed():
    msg = wintypes.MSG()
    if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):  # PM_REMOVE = 1
        if msg.message == 0x0312 and msg.wParam == HOTKEY_ID:  # WM_HOTKEY
            return True
    return False

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", wintypes.ULARGE_INTEGER),
    ]

class INPUT(ctypes.Structure):
    class _I(ctypes.Union):
        _fields_ = [("mi", MOUSEINPUT)]
    _anonymous_ = ("i",)
    _fields_ = [("type", wintypes.DWORD), ("i", _I)]

def calc_centers():
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

            r = p >> 3
            c = p & 7

            if r > maxr: maxr = r
            if c > maxc: maxc = c
            if r < minr: minr = r
            if c < minc: minc = c
        
        r2.append((minr + maxr) * 0.5)
        c2.append((minc + maxc) * 0.5)

    return tuple(r2), tuple(c2)

CENTER_R, CENTER_C = calc_centers()

def normalize_lr(x: int) -> int:
    min_r = 7
    min_c = 7
    y = x
    while y:
        lsb = y & -y
        p = lsb.bit_length() - 1
        r = p >> 3
        c = p & 7
        if r < min_r: min_r = r
        if c < min_c: min_c = c
        y ^= lsb
    return x >> (min_r * 8 + min_c)

def get_screen(hwnd: int) -> Tuple[np.ndarray, int]:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    tl = win32gui.ClientToScreen(hwnd, (left, top))
    br = win32gui.ClientToScreen(hwnd, (right, bottom))
    l, t = tl
    r, b = br
    w = r - l
    h = b - t
    
    with mss.mss() as sct:
        img = sct.grab({"left": l, "top": t, "width": w, "height": h})
        frame = np.array(img, dtype=np.float32)
        # frame is BGRA
        frame_gray = 0.2126 * frame[:, :, 2] + 0.7152 * frame[:, :, 1] + 0.0722 * frame[:, :, 0]
        frame_gray = np.ascontiguousarray(frame_gray[58:])
    return frame_gray, w

def compute_board_centers(w: int) -> List[List[Tuple[int, int]]]:
    s = w / MEASUREMENT_WIDTH
    board_left = BOARD_LEFT * s
    board_top = BOARD_TOP * s
    pitch = (CELL_SIZE + GAP_SIZE) * s
    half = (CELL_SIZE * 0.5) * s

    centers = []
    for r in range(8):
        row = []
        y = board_top + r * pitch + half
        y_screen = int(round(y))
        for c in range(8):
            x = board_left + c * pitch + half
            x_screen = int(round(x))
            row.append((x_screen, y_screen))
        centers.append(row)
    return centers

def compute_center(r, c, w):
    s = w / MEASUREMENT_WIDTH
    board_left = BOARD_LEFT * s
    board_top  = BOARD_TOP  * s
    pitch = (CELL_SIZE + GAP_SIZE) * s
    half  = (CELL_SIZE * 0.5) * s
    return (board_left + c * pitch + half, board_top + r * pitch + half)

def sample_patch(frame: np.ndarray, x: int, y: int, patch: int, x_max: int = None, y_max: int = None) -> float:
    H, W = frame.shape
    if x_max is None:
        x_max = W
    if y_max is None:
        y_max = H

    x0 = max(0, x - patch)
    x1 = min(x_max, x + patch + 1)
    y0 = max(0, y - patch)
    y1 = min(y_max, y + patch + 1)
    if x0 >= x1 or y0 >= y1:
        return 1e9
    return float(np.median(frame[y0:y1, x0:x1]))

def get_board(
    frame: np.ndarray, 
    centers: List[List[Tuple[int, int]]], 
    tol=0.2, patch=2
) -> int:
    filled = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for c in range(8):
            x, y = centers[r][c]
            med = sample_patch(frame, x, y, patch)
            if med >= 255:
                continue
            filled[r, c] = not (abs(med - EMPTY_COL) <= tol)
    return int(np.sum(filled.astype(np.uint64) * BOARD_SHIFTS))

def tray_slot_rois(w: int) -> Tuple[Tuple[int, int, int, int]]:
    s = w / MEASUREMENT_WIDTH
    roi_sz = int(round((5 * BLOCK_SIZE + 4 * TRAY_GAP) * s))
    y0 = int(round(BLOCK_TOP * s))
    y1 = y0 + roi_sz

    out = []
    for x0 in BLOCK_LEFT:
        x0s = int(round(x0 * s))
        out.append((x0s, y0, x0s + roi_sz, y1))
    return tuple(out)

def get_block(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    w: int,
    tol=0.2,
    patch=2
) -> int:
    x0, y0, x1, y1 = roi
    slot = frame[y0:y1, x0:x1]
    fg = np.abs(slot - BG_COL) > tol

    ys, xs = np.nonzero(fg)
    if xs.size == 0:
        return NULL_BLOCK
    minx, miny = int(xs.min()), int(ys.min())

    s = w / MEASUREMENT_WIDTH
    pitch = (BLOCK_SIZE + TRAY_GAP) * s
    half = BLOCK_SIZE * 0.5 * s

    # top-left mini-cell center in ROI-local coords
    base_x = minx + half
    base_y = miny + half

    out = np.zeros((5, 5), dtype=bool)

    for r in range(5):
        cy = int(round(y0 + base_y + r * pitch))
        for c in range(5):
            cx = int(round(x0 + base_x + c * pitch))
            med = sample_patch(frame, cx, cy, patch, x1, y1) # clamp bounds
            if med >= 255:
                continue
            out[r, c] = (abs(med - BG_COL) > tol)
    
    b = normalize_lr(int(np.sum(out.astype(np.uint64) * BLOCK_SHIFTS)))
    return NULL_BLOCK if b == 0 else BLOCKS.index(b)

def get_blocks(frame: np.ndarray, w: int):
    blocks = [get_block(frame, roi, w) for roi in tray_slot_rois(w)]
    return blocks

def _send_mouse(flags: int, data: int = 0):
    inp = INPUT(type=INPUT_MOUSE, mi=MOUSEINPUT(
        dx=0, dy=0, mouseData=data, dwFlags=flags, time=0, dwExtraInfo=0
    ))
    n = user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))
    if n != 1:
        raise ctypes.WinError(ctypes.get_last_error())

def move_cursor(x: int, y: int):
    if not user32.SetCursorPos(int(x), int(y)):
        raise ctypes.WinError(ctypes.get_last_error())

def left_down():
    _send_mouse(MOUSEEVENTF_LEFTDOWN)

def left_up():
    _send_mouse(MOUSEEVENTF_LEFTUP)

def drag(sx0: int, sy0: int, sx1: int, sy1: int,
         steps: int = 20, total_ms: int = 200):
    move_cursor(sx0, sy0)
    time.sleep(0.01)
    left_down()
    time.sleep(0.02)

    # Interpolate motion
    for i in range(1, steps + 1):
        t = i / steps
        x = int(round(sx0 + t * (sx1 - sx0)))
        y = int(round(sy0 + t * (sy1 - sy0)))
        move_cursor(x, y)
        time.sleep((total_ms / steps) * 0.001)

    time.sleep(0.02)
    left_up()
    time.sleep(0.01)

def send_input(hwnd, action: int, blocks, bmap, w: int):
    slot = bmap[action >> 6]
    bid  = blocks[slot]

    shift = (action & 63) - REF_POS[bid]
    r0 = 7 - (shift >> 3)
    c0 = 7 - (shift & 7)

    s = w / MEASUREMENT_WIDTH

    # pickup point (client coords) as float
    x0 = PICKUP_X[slot] * s
    y0 = PICKUP_Y * s

    # block visual center at pickup (float)
    cy0 = BIG_CENTER_Y * s

    tx, ty = compute_center(r0 - CENTER_R[bid], c0 - CENTER_C[bid], w)
    dx = tx - x0
    dy = ty - cy0

    # inverse affine in float
    x1 = x0 + (dx / AFFINE_X)
    y1 = y0 + (dy / AFFINE_Y)

    # round ONLY when converting to actual cursor pixels
    sx0, sy0 = win32gui.ClientToScreen(hwnd, (int(round(x0)), int(round(y0))))
    sx1, sy1 = win32gui.ClientToScreen(hwnd, (int(round(x1)), int(round(y1))))
    drag(sx0, sy0, sx1, sy1)

def board_str(x: int):
    y = bin(x)[2:].zfill(64)
    z = ""
    for i in range(0, 64, 8):
        z += y[i:i+8] + "\n"
    return z

def main():
    time.sleep(2)
    # find window by title (exact match)
    wnd = win32gui.FindWindow(None, "Block Blast! - CoarserGeneral9")
    if not wnd:
        raise RuntimeError("Window not found")
    
    # setup for model
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
    
    # setup for env
    frame, w = get_screen(wnd)
    centers = compute_board_centers(w)
    board0 = get_board(frame, centers)
    blocks = get_blocks(frame, w)
    
    # initialize bmap and canonicalize blocks
    n = NULL_BLOCK
    x, y, z = blocks
    bmap = [0, 1, 2]

    if x == n:
        if y == n:
            blocks = [z, n, n]
            bmap = [2, 3, 3]
        elif z == n:
            blocks = [y, n, n]
            bmap = [1, 3, 3]
        else:
            blocks = [y, z, n]
            bmap = [1, 2, 3]
    elif y == n:
        if z == n:
            bmap = [0, 3, 3]
        else:
            blocks = [x, z, n]
            bmap = [0, 2, 3]
    elif z == n:
        bmap = [0, 1, 3]

    env = bb.Env(board0, cfg.seed)
    env.set_blocks(blocks[0], blocks[1], blocks[2])

    running = True
    while running:
        if env.need_blocks:
            print(board_str(env.board))
            time.sleep(1.0)
            frame, w = get_screen(wnd)
            blocks = get_blocks(frame, w)
            env.set_blocks(blocks[0], blocks[1], blocks[2])
            bmap = [0, 1, 2]
            
        if hotkey_pressed():
            print("hotkey pressed, terminating")
            running = False
            break

        if env.done:
            running = False

        board = torch.tensor([env.board], dtype=torch.uint64, device=device)
        meta = torch.tensor([env.meta], dtype=torch.uint64, device=device)

        pi_beh = plan.search_batch(
            board, meta, policy_logits_fn, value_fn, use_noise=False
        ).squeeze(0) # (192,)
        a = int(pi_beh.argmax().item())
        env.step(a) # env step

        # send input
        send_input(wnd, a, blocks, bmap, w)

        # update bmap
        slot_engine = a >> 6
        slot = bmap[slot_engine]
        if slot_engine == 0:
            bmap[0] = bmap[1]
            bmap[1] = bmap[2]
        elif slot_engine == 1:
            bmap[1] = bmap[2]
        bmap[2] = 3 # null sentinel

        # update blocks
        blocks[slot] = NULL_BLOCK
        time.sleep(0.2)
    user32.UnregisterHotKey(None, HOTKEY_ID)

if __name__ == "__main__":
    main()