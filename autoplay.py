import argparse
import ctypes
from ctypes import wintypes
from pathlib import Path
import time
from typing import List, Tuple

from mss import MSS
import numpy as np
import torch
import win32gui
from torch import Tensor

import bbengine.bbengine as bb
from core import make_models
from policy_math.value_scale import vscale_inv
from utils import checkpoint_config, load_models, unpack_obs


# ===========================================================================
# User-facing defaults
# ===========================================================================

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT_DIR / "chkpts" / "chkpt3"

# Controller calibration
# differs between emulators (tested BlueStacks and Google Play Games Beta)
DEFAULT_WINDOW_MATCH = "Block Blast!"
AREA_TOP = 58
AREA_LEFT = 0

# differs across versions
# luminance of empty cell RGB(33, 40, 82)
EMPTY_COL = 41.5442 # 35.6148 # RGB(24, 36, 66)


# ===========================================================================
# Calibration (should be relatively constant)
# ===========================================================================

MEASUREMENT_WIDTH = 576  # board width that measurements were recorded on

BG_COL = 80.735      # luminance of background color RGB(57, 81, 148)

BOARD_LEFT = 36
BOARD_TOP = 184
CELL_SIZE = 62
GAP_SIZE = 2

TRAY_GAP = 1
BLOCK_TOP = 770
BLOCK_LEFT = (48, 216, 384)
BLOCK_SIZE = 28
HALF_BLOCK_SIZE = 14

PICKUP_Y = 900
PICKUP_X = (118, 286, 454)
BIG_CENTER_Y = 696

AFFINE_X = 1.45
AFFINE_Y = 1.45


# ===========================================================================
# Engine constants
# ===========================================================================

IDX = np.arange(63, -1, -1, dtype=np.uint64).reshape(8, 8)

BOARD_SHIFTS = 1 << IDX
BLOCK_SHIFTS = 1 << IDX[-5:, -5:]

BLOCKS = bb.blocks.tolist()
REF_POS = bb.ref_pos.tolist()
NULL_BLOCK = bb.NULL_BLOCK


# ===========================================================================
# Windows input / hotkey
# ===========================================================================

user32 = ctypes.WinDLL("user32", use_last_error=True)

# Per-monitor-v2 DPI awareness.
user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))

INPUT_MOUSE = 0

MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

MOD_CONTROL = 0x0002
VK_Q = 0x51

HOTKEY_ID = 1
WM_HOTKEY = 0x0312
PM_REMOVE = 1


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
    _fields_ = [
        ("type", wintypes.DWORD),
        ("i", _I),
    ]

class BlockRecognitionError(RuntimeError):
    pass


def register_hotkey() -> None:
    if not user32.RegisterHotKey(
        None,
        HOTKEY_ID,
        MOD_CONTROL,
        VK_Q,
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def unregister_hotkey() -> None:
    user32.UnregisterHotKey(None, HOTKEY_ID)


def hotkey_pressed() -> bool:
    msg = wintypes.MSG()

    if user32.PeekMessageW(
        ctypes.byref(msg),
        None,
        0,
        0,
        PM_REMOVE,
    ):
        return (
            msg.message == WM_HOTKEY
            and msg.wParam == HOTKEY_ID
        )

    return False


# ===========================================================================
# CLI / device / window discovery
# ===========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Autoplay Block Blast using the trained search/RL agent. "
            "The current screen and mouse calibration targets "
            "BlueStacks on Windows."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Path to checkpoint "
            f"(default: {DEFAULT_CHECKPOINT})"
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Neural-network inference device: "
            "auto, cpu, cuda, cuda:0, etc. "
            "(default: auto)"
        ),
    )

    parser.add_argument(
        "--window-title",
        default=None,
        help=(
            "Exact game-window title. Normally unnecessary; "
            "use this if automatic discovery finds multiple windows."
        ),
    )

    parser.add_argument(
        "--window-match",
        default=DEFAULT_WINDOW_MATCH,
        help=(
            "Substring used for automatic window discovery "
            f"(default: {DEFAULT_WINDOW_MATCH!r})"
        ),
    )

    parser.add_argument(
        "--startup-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before starting (default: 2.0).",
    )

    return parser.parse_args()


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")

    device = torch.device(spec)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {spec!r} was requested, "
                "but CUDA is unavailable."
            )

        if device.index is not None:
            torch.cuda.set_device(device.index)

    return device


def find_game_window(
    exact_title: str | None = None,
    title_match: str = DEFAULT_WINDOW_MATCH,
) -> Tuple[int, str]:
    """
    Find a visible Block Blast window.

    If exact_title is supplied, use an exact Win32 title match.
    Otherwise find visible windows whose title contains title_match.
    """
    if exact_title is not None:
        hwnd = win32gui.FindWindow(None, exact_title)

        if not hwnd:
            raise RuntimeError(
                f"No visible window found with exact title "
                f"{exact_title!r}."
            )

        return hwnd, win32gui.GetWindowText(hwnd)

    matches: List[Tuple[int, str]] = []
    needle = title_match.lower()

    def callback(hwnd: int, _):
        if not win32gui.IsWindowVisible(hwnd):
            return

        title = win32gui.GetWindowText(hwnd).strip()

        if title and needle in title.lower():
            matches.append((hwnd, title))

    win32gui.EnumWindows(callback, None)

    if not matches:
        raise RuntimeError(
            f"No visible window matching {title_match!r} was found.\n"
            "Start Block Blast in BlueStacks, or pass "
            "--window-title with the exact title."
        )

    if len(matches) > 1:
        choices = "\n".join(
            f"  - {title}"
            for _, title in matches
        )

        raise RuntimeError(
            "Multiple matching windows were found:\n"
            f"{choices}\n"
            "Pass --window-title with the exact title to select one."
        )

    return matches[0]


# ===========================================================================
# Block geometry
# ===========================================================================

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

            if r > maxr:
                maxr = r
            if c > maxc:
                maxc = c
            if r < minr:
                minr = r
            if c < minc:
                minc = c

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

        if r < min_r:
            min_r = r
        if c < min_c:
            min_c = c

        y ^= lsb

    return x >> (min_r * 8 + min_c)


# ===========================================================================
# Screen capture / recognition
# ===========================================================================

def get_screen(hwnd: int) -> Tuple[np.ndarray, int]:
    left, top, right, bottom = win32gui.GetClientRect(hwnd)

    tl = win32gui.ClientToScreen(hwnd, (left, top))
    br = win32gui.ClientToScreen(hwnd, (right, bottom))

    l, t = tl
    r, b = br

    w = r - l
    h = b - t

    with MSS() as sct:
        img = sct.grab(
            {
                "left": l,
                "top": t,
                "width": w,
                "height": h,
            }
        )

        frame = np.array(
            img,
            dtype=np.float32,
        )

        # MSS returns BGRA.
        frame_gray = (
            0.2126 * frame[:, :, 2]
            + 0.7152 * frame[:, :, 1]
            + 0.0722 * frame[:, :, 0]
        )

        # client-area calibration
        frame_gray = np.ascontiguousarray(
            frame_gray[AREA_TOP:, AREA_LEFT:]
        )

    return frame_gray, w


def compute_board_centers(
    w: int,
) -> List[List[Tuple[int, int]]]:
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

            row.append(
                (x_screen, y_screen)
            )

        centers.append(row)

    return centers


def compute_center(r, c, w):
    s = w / MEASUREMENT_WIDTH

    board_left = BOARD_LEFT * s
    board_top = BOARD_TOP * s

    pitch = (CELL_SIZE + GAP_SIZE) * s
    half = (CELL_SIZE * 0.5) * s

    return (
        board_left + c * pitch + half,
        board_top + r * pitch + half,
    )


def sample_patch(
    frame: np.ndarray,
    x: int,
    y: int,
    patch: int,
    x_max: int = None,
    y_max: int = None,
) -> float:
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

    return float(
        np.median(frame[y0:y1, x0:x1])
    )


def get_board(
    frame: np.ndarray,
    centers: List[List[Tuple[int, int]]],
    tol=0.2,
    patch=2,
) -> int:
    filled = np.zeros(
        (8, 8),
        dtype=bool,
    )

    for r in range(8):
        for c in range(8):
            x, y = centers[r][c]

            med = sample_patch(
                frame,
                x,
                y,
                patch,
            )

            if med >= 255:
                continue

            filled[r, c] = not (
                abs(med - EMPTY_COL) <= tol
            )

    return int(
        np.sum(
            filled.astype(np.uint64)
            * BOARD_SHIFTS
        )
    )


def tray_slot_rois(
    w: int,
) -> Tuple[Tuple[int, int, int, int], ...]:
    s = w / MEASUREMENT_WIDTH

    roi_sz = int(
        round(
            (5 * BLOCK_SIZE + 4 * TRAY_GAP) * s
        )
    )

    y0 = int(round(BLOCK_TOP * s))
    y1 = y0 + roi_sz

    out = []

    for x0 in BLOCK_LEFT:
        x0s = int(round(x0 * s))

        out.append(
            (
                x0s,
                y0,
                x0s + roi_sz,
                y1,
            )
        )

    return tuple(out)


def get_block(
    frame: np.ndarray,
    roi: Tuple[int, int, int, int],
    w: int,
    tol=0.2,
    patch=2,
) -> int:
    x0, y0, x1, y1 = roi

    slot = frame[y0:y1, x0:x1]
    fg = np.abs(slot - BG_COL) > tol

    ys, xs = np.nonzero(fg)

    if xs.size == 0:
        return NULL_BLOCK

    minx = int(xs.min())
    miny = int(ys.min())

    s = w / MEASUREMENT_WIDTH

    pitch = (BLOCK_SIZE + TRAY_GAP) * s
    half = BLOCK_SIZE * 0.5 * s

    # Top-left mini-cell center in ROI-local coordinates.
    base_x = minx + half
    base_y = miny + half

    out = np.zeros(
        (5, 5),
        dtype=bool,
    )

    for r in range(5):
        cy = int(
            round(
                y0 + base_y + r * pitch
            )
        )

        for c in range(5):
            cx = int(
                round(
                    x0 + base_x + c * pitch
                )
            )

            med = sample_patch(
                frame,
                cx,
                cy,
                patch,
                x1,
                y1,
            )

            if med >= 255:
                continue

            out[r, c] = (
                abs(med - BG_COL) > tol
            )

    b = normalize_lr(
        int(
            np.sum(
                out.astype(np.uint64)
                * BLOCK_SHIFTS
            )
        )
    )

    if b == 0:
        return NULL_BLOCK

    try:
        return BLOCKS.index(b)
    except ValueError as exc:
        raise BlockRecognitionError(
            f"Unrecognized block bitmap: {b}"
        ) from exc


def get_blocks(
    frame: np.ndarray,
    w: int,
):
    return [
        get_block(frame, roi, w)
        for roi in tray_slot_rois(w)
    ]

def get_blocks_retry(
    hwnd: int,
    *,
    initial_delay: float = 0.0,
    retry_delay: float = 0.25,
    max_attempts: int = 20,
):
    if initial_delay > 0:
        time.sleep(initial_delay)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        frame, w = get_screen(hwnd)

        try:
            blocks = get_blocks(frame, w)

            print(
                f"recognized blocks on attempt {attempt}: "
                f"{blocks}"
            )

            return blocks, w

        except BlockRecognitionError as exc:
            last_error = exc

            print(
                f"block recognition failed "
                f"(attempt {attempt}/{max_attempts}); retrying..."
            )

            time.sleep(retry_delay)

    raise RuntimeError(
        "Could not recognize tray after "
        f"{max_attempts} attempts."
    ) from last_error

# ===========================================================================
# Mouse control
# ===========================================================================

def _send_mouse(
    flags: int,
    data: int = 0,
):
    inp = INPUT(
        type=INPUT_MOUSE,
        mi=MOUSEINPUT(
            dx=0,
            dy=0,
            mouseData=data,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        ),
    )

    n = user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp),
    )

    if n != 1:
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def move_cursor(
    x: int,
    y: int,
):
    if not user32.SetCursorPos(
        int(x),
        int(y),
    ):
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def left_down():
    _send_mouse(
        MOUSEEVENTF_LEFTDOWN
    )


def left_up():
    _send_mouse(
        MOUSEEVENTF_LEFTUP
    )


def drag(
    sx0: int,
    sy0: int,
    sx1: int,
    sy1: int,
    steps: int = 20,
    total_ms: int = 200,
):
    move_cursor(sx0, sy0)

    time.sleep(0.01)
    left_down()
    time.sleep(0.02)

    for i in range(1, steps + 1):
        t = i / steps

        x = int(
            round(
                sx0 + t * (sx1 - sx0)
            )
        )

        y = int(
            round(
                sy0 + t * (sy1 - sy0)
            )
        )

        move_cursor(x, y)

        time.sleep(
            (total_ms / steps) * 0.001
        )

    time.sleep(0.02)
    left_up()
    time.sleep(0.01)


def send_input(
    hwnd,
    action: int,
    blocks,
    bmap,
    w: int,
):
    slot = bmap[action >> 6]
    bid = blocks[slot]

    shift = (
        (action & 63)
        - REF_POS[bid]
    )

    r0 = 7 - (shift >> 3)
    c0 = 7 - (shift & 7)

    s = w / MEASUREMENT_WIDTH

    # Pickup point in client coordinates.
    x0 = PICKUP_X[slot] * s
    y0 = PICKUP_Y * s

    # Visual center of block after pickup.
    cy0 = BIG_CENTER_Y * s

    tx, ty = compute_center(
        r0 - CENTER_R[bid],
        c0 - CENTER_C[bid],
        w,
    )

    dx = tx - x0
    dy = ty - cy0

    x1 = x0 + dx / AFFINE_X
    y1 = y0 + dy / AFFINE_Y

    sx0, sy0 = win32gui.ClientToScreen(
        hwnd,
        (
            int(round(x0)),
            int(round(y0)),
        ),
    )

    sx1, sy1 = win32gui.ClientToScreen(
        hwnd,
        (
            int(round(x1)),
            int(round(y1)),
        ),
    )

    drag(
        sx0,
        sy0,
        sx1,
        sy1,
    )


# ===========================================================================
# Debug
# ===========================================================================

def board_str(x: int):
    y = bin(x)[2:].zfill(64)

    z = ""

    for i in range(0, 64, 8):
        z += y[i:i + 8] + "\n"

    return z


# ===========================================================================
# Main
# ===========================================================================

def canonicalize_hand(physical_blocks):
    engine_blocks = []
    bmap = []

    for physical_slot, bid in enumerate(physical_blocks):
        if bid != NULL_BLOCK:
            engine_blocks.append(bid)
            bmap.append(physical_slot)

    while len(engine_blocks) < 3:
        engine_blocks.append(NULL_BLOCK)
        bmap.append(3)

    return engine_blocks, bmap

def main() -> int:
    args = parse_args()

    checkpoint = (
        args.checkpoint
        .expanduser()
        .resolve()
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Pass another checkpoint with --checkpoint PATH."
        )

    device = resolve_device(
        args.device
    )

    if args.startup_delay > 0:
        time.sleep(
            args.startup_delay
        )

    wnd, window_title = find_game_window(
        exact_title=args.window_title,
        title_match=args.window_match,
    )

    print(
        f"window     : {window_title}"
    )
    print(
        f"checkpoint : {checkpoint}"
    )
    print(
        f"device     : {device}"
    )

    if device.type == "cuda":
        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(device)}"
        )

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------

    chkpt_dir = str(
        checkpoint.parent
    )
    chkpt_name = checkpoint.name

    cfg = checkpoint_config(
        chkpt_dir,
        chkpt_name,
    )

    P, V = make_models(
        cfg,
        device,
    )

    load_models(
        chkpt_dir,
        chkpt_name,
        P,
        V,
        device,
    )

    P.eval()
    V.eval()

    # -----------------------------------------------------------------------
    # Planner
    # -----------------------------------------------------------------------

    search_cfg = bb.SearchConfig()

    search_cfg.beam_width = (
        cfg.P_beam_width
    )

    search_cfg.per_parent_top_m = (
        cfg.P_per_parent_top_m
    )

    search_cfg.root_eps = (
        cfg.P_root_eps
    )

    search_cfg.max_eval_P = (
        cfg.max_bs_inference_P
    )

    search_cfg.max_eval_V = (
        cfg.max_bs_inference_V
    )

    search_cfg.gamma = (
        cfg.V_gamma
    )

    search_cfg.teacher_tau = (
        cfg.P_teacher_tau
    )

    plan = bb.BeamSearch(
        search_cfg,
        cfg.seed,
    )

    @torch.inference_mode()
    def policy_logits_fn(
        boards_u64: Tensor,
        metas_u64: Tensor,
    ) -> Tensor:
        b, i, k = unpack_obs(
            boards_u64,
            metas_u64,
            device,
        )

        return (
            P(b, i, k)
            .cpu()
        )

    @torch.inference_mode()
    def value_fn(
        boards_u64: Tensor,
        metas_u64: Tensor,
    ) -> Tensor:
        b, i, k = unpack_obs(
            boards_u64,
            metas_u64,
            device,
        )

        return (
            vscale_inv(
                V(b, i, k)
            )
            .cpu()
        )

    # -----------------------------------------------------------------------
    # Read current real-game state
    # -----------------------------------------------------------------------

    frame, w = get_screen(wnd)

    centers = compute_board_centers(w)

    board0 = get_board(
        frame,
        centers,
    )

    blocks = get_blocks(
        frame,
        w,
    )

    # Canonicalize a partially-used initial hand.
    blocks = get_blocks(frame, w) # ALWAYS physical tray order

    engine_blocks, bmap = canonicalize_hand(blocks)

    env = bb.Env(board0, cfg.seed)

    env.set_blocks(
        engine_blocks[0],
        engine_blocks[1],
        engine_blocks[2],
    )

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------

    register_hotkey()

    print(
        "autoplay started "
        "(press Ctrl+Q to stop)"
    )

    try:
        running = True

        while running:
            if hotkey_pressed():
                print(
                    "Ctrl+Q pressed; terminating"
                )
                break

            if env.done:
                print(
                    "game over"
                )
                break

            # ---------------------------------------------------------------
            # Read a fresh hand when all previous blocks have been consumed.
            # ---------------------------------------------------------------

            if env.need_blocks:
                print(
                    board_str(env.board)
                )

                # Allow the game's tray animation to finish.
                time.sleep(1.0)
                blocks, _ = get_blocks_retry(wnd)

                env.set_blocks(
                    blocks[0],
                    blocks[1],
                    blocks[2],
                )

                bmap = [0, 1, 2]

            # ---------------------------------------------------------------
            # Plan
            # ---------------------------------------------------------------

            # IMPORTANT:
            # BeamSearch/native engine state remains on CPU. Neural callbacks
            # independently move observations to the selected inference device.
            board = torch.tensor(
                [env.board],
                dtype=torch.uint64,
            )

            meta = torch.tensor(
                [env.meta],
                dtype=torch.uint64,
            )

            pi_beh = plan.search_batch(
                board,
                meta,
                policy_logits_fn,
                value_fn,
                use_noise=False,
            ).squeeze(0)

            action = int(
                pi_beh.argmax().item()
            )

            # ---------------------------------------------------------------
            # Update internal engine state
            # ---------------------------------------------------------------

            env.step(action)

            # ---------------------------------------------------------------
            # Execute move in the real game
            # ---------------------------------------------------------------

            send_input(
                wnd,
                action,
                blocks,
                bmap,
                w,
            )

            # ---------------------------------------------------------------
            # Update mapping between canonical engine slots and actual tray
            # positions.
            # ---------------------------------------------------------------

            slot_engine = action >> 6
            slot = bmap[slot_engine]

            if slot_engine == 0:
                bmap[0] = bmap[1]
                bmap[1] = bmap[2]

            elif slot_engine == 1:
                bmap[1] = bmap[2]

            bmap[2] = 3

            blocks[slot] = NULL_BLOCK

            time.sleep(0.2)

    finally:
        unregister_hotkey()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
