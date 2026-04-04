from typing import Dict
import torch
from torch import Tensor

class MemoryBuffer(object):
    def __init__(self, capacity: int, n_envs: int, device: torch.device | str = "cpu"):
        self.capacity = capacity
        self.n_envs = n_envs
        self.device = torch.device(device)

        # state / action
        self.board = self._buf(64) # (T, N, 64)
        self.info = self._buf(2, dtype=torch.int64) # (T, N, 2)
        self.block = self._buf(3, dtype=torch.int64) # (T, N, 3)
        self.legal_mask = self._buf(192, dtype=torch.bool) # (T, N, 192)

        # policy
        self.pi_behavior = self._buf(192) # (T, N, 192)
        self.action = self._buf(dtype=torch.int64) # (T, N)

        # mdp
        self.reward = self._buf() # (T, N)
        self.done = self._buf(dtype=torch.bool) # (T, N)
        self.value = self._buf() # (T, N)

        self.returns = self._buf()
        self.v_targ_after = self._buf()

        # afterstate
        self.board_after = self._buf(64) # (T, N, 64)
        self.info_after = self._buf(2, dtype=torch.int64) # (T, N, 2)
        self.block_after = self._buf(3, dtype=torch.int64) # (T, N, 3)
        self.need_blocks = self._buf(dtype=torch.bool) # (T, N)

        self.ptr = 0
        self.size = 0

    def _buf(self, d: int = 1, dtype: torch.dtype = torch.float) -> Tensor:
        if d == 1:
            shape = (self.capacity, self.n_envs)
        else:
            shape = (self.capacity, self.n_envs, d)
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def to(self, device: torch.device | str) -> "MemoryBuffer":
        device = torch.device(device)
        self.device = device
        for name, tensor in list(self.__dict__.items()):
            if isinstance(tensor, Tensor):
                setattr(self, name, tensor.to(device))
        return self

    def clear(self) -> None:
        self.ptr = 0
        self.size = 0
        for _, tensor in self.__dict__.items():
            if isinstance(tensor, Tensor):
                tensor.zero_()

    def __len__(self) -> int:
        return self.size

    @property
    def is_full(self) -> bool:
        return self.size == self.capacity

    def _phys_idx(self, idx: int) -> int:
        if idx < 0:
            idx += self.size
        return (self.ptr - self.size + idx) % self.capacity

    @torch.no_grad()
    def append_step(self,
        boards_t: Tensor, # (N, 64) float
        infos_t: Tensor, # (N, 2) int64
        blocks_t: Tensor, # (N, 3) int64
        legal_mask_t: Tensor, # (N, 192) bool
        pi_behavior_t: Tensor, # (N, 192) float
        action_t: Tensor, # (N,) int64
        reward_t: Tensor, # (N,) float
        done_t: Tensor, # (N,) bool
        value_t: Tensor, # (N,) float
        boards_tp1: Tensor, # (N, 64) float
        infos_tp1: Tensor, # (N, 2) int64
        blocks_tp1: Tensor, # (N, 3) int64
        need_blocks_t: Tensor, # (N,) bool
    ) -> int:
        row = self.ptr

        self.board[row] = boards_t.to(self.device)
        self.info[row] = infos_t.to(self.device)
        self.block[row] = blocks_t.to(self.device)
        self.legal_mask[row] = legal_mask_t.to(self.device)

        self.pi_behavior[row] = pi_behavior_t.to(self.device)
        self.action[row] = action_t.to(self.device)
        self.reward[row] = reward_t.to(self.device)
        self.done[row] = done_t.to(self.device)
        self.value[row] = value_t.to(self.device)

        self.board_after[row] = boards_tp1.to(self.device)
        self.info_after[row] = infos_tp1.to(self.device)
        self.block_after[row] = blocks_tp1.to(self.device)
        self.need_blocks[row] = need_blocks_t.to(self.device)

        self.ptr = (row + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        return row

    def write_adv_segment(
        self, returns: Tensor, returns_after: Tensor, t0: int, T: int
    ) -> None:
        t0_phys = self._phys_idx(t0)
        rows_T = (t0_phys + torch.arange(T, device=self.device)) % self.capacity
        self.returns[rows_T] = returns.to(self.device)
        self.v_targ_after[rows_T] = returns_after.to(self.device)

    # samplers
    def sample_policy_rollout(self, t_idx: Tensor) -> Dict[str, Tensor]:
        t_idx = t_idx.to(self.device)
        phys = (self.ptr - self.size + t_idx) % self.capacity

        return {
            "board": self.board[phys], # (T, N, 64)
            "info": self.info[phys], # (T, N, 2)
            "block": self.block[phys], # (T, N, 3)
            "legal_mask": self.legal_mask[phys], # (T, N, 192)
            "pi_behavior": self.pi_behavior[phys], # (T, N, 192)
            "return": self.returns[phys], # (T, N)
        }

    def sample_value_rollout(self, t_idx: Tensor) -> Dict[str, Tensor]:
        t_idx = t_idx.to(self.device)
        phys = (self.ptr - self.size + t_idx) % self.capacity

        return {
            "board": self.board[phys], # (T, N, 64)
            "info": self.info[phys], # (T, N, 2)
            "block": self.block[phys], # (T, N, 3)

            "board_after": self.board_after[phys], # (T, N, 64)
            "info_after": self.info_after[phys], # (T, N, 2)
            "block_after": self.block_after[phys], # (T, N, 3)
            "need_blocks": self.need_blocks[phys], # (T, N)

            "reward": self.reward[phys], # (T, N)
            "returns": self.returns[phys], # (T, N)
            "v_targ_after": self.v_targ_after[phys], # (T, N)
        }

    # sample a contiguous rollout segment for generator targets (td / vtrace)
    def sample_rollout_segment(self, t0: int, T: int) -> Dict[str, Tensor]:
        t0_phys = self._phys_idx(t0)
        rows_T = (t0_phys + torch.arange(T, device=self.device)) % self.capacity

        return {
            "board": self.board[rows_T], # (T, N, 64)
            "info": self.info[rows_T], # (T, N, 2)
            "block": self.block[rows_T], # (T, N, 3)

            "board_after": self.board_after[rows_T], # (T, N, 64)
            "info_after": self.info_after[rows_T], # (T, N, 2)
            "reward": self.reward[rows_T], # (T, N)
            "value": self.value[rows_T], # (T, N)
            "done": self.done[rows_T], # (T, N)
            "need_blocks": self.need_blocks[rows_T], # (T, N)
        }