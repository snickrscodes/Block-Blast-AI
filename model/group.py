import torch
from torch import Tensor

class Group(object):
    def __init__(self, n: int, order: int):
        self.n = n
        self.identity = 0
        self.order = order
        self.mats = self.build()
        self.product_table = self._get_product_table()
        self.inverse_map = self._get_inverse_map()

    def build(self) -> Tensor:
        raise NotImplementedError()

    def _get_product_table(self) -> Tensor: # left actions
        raise NotImplementedError()

    def _get_inverse_map(self) -> Tensor:
        raise NotImplementedError()
    
class DihedralGroup(Group):
    def __init__(self, n: int):
        super().__init__(n, 2 * n)

    def build(self) -> Tensor:
        self.identity = 0
        thetas = 2 * torch.pi / self.n * torch.arange(self.n)

        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        rot_mats = torch.zeros((self.n, 2, 2))
        rot_mats[:, 0, 0] = cos_t
        rot_mats[:, 0, 1] = -sin_t
        rot_mats[:, 1, 0] = sin_t
        rot_mats[:, 1, 1] = cos_t

        refl_mats = torch.zeros_like(rot_mats)
        refl_mats[:, 0, 0] = cos_t
        refl_mats[:, 0, 1] = sin_t
        refl_mats[:, 1, 0] = sin_t
        refl_mats[:, 1, 1] = -cos_t

        mats = torch.cat([rot_mats, refl_mats], dim=0) # (2N, 2, 2)
        return mats

    def _get_product_table(self) -> Tensor: # left action convention
        a = torch.arange(self.n, dtype=torch.int64)
        add = (a[:, None] + a[None, :]) % self.n
        sub = (a[:, None] - a[None, :]) % self.n
        top = torch.cat([add, add + self.n], dim=1)
        bottom = torch.cat([sub + self.n, sub], dim=1)
        return torch.cat([top, bottom], dim=0) # (2N, 2N)

    def _get_inverse_map(self) -> Tensor:
        inv = torch.zeros((self.order,), dtype=torch.int64)
        inv[:self.n] = (self.n - torch.arange(self.n)) % self.n
        inv[self.n:] = torch.arange(self.n, self.order)
        return inv # (2N,)
    
class CyclicGroup(Group):
    def __init__(self, n):
        super().__init__(n, n)

    def build(self) -> Tensor:
        thetas = 2 * torch.pi / self.n * torch.arange(self.n)
        cos_t = torch.cos(thetas)
        sin_t = torch.sin(thetas)

        mats = torch.zeros((self.n, 2, 2))
        mats[:, 0, 0] = cos_t
        mats[:, 0, 1] = -sin_t
        mats[:, 1, 0] = sin_t
        mats[:, 1, 1] = cos_t

        return mats

    def _get_product_table(self) -> Tensor:
        idx = torch.arange(self.n)
        return (idx[:, None] + idx[None, :]) % self.n # (n, n)
    
    def _get_inverse_map(self) -> Tensor:
        return (self.n - torch.arange(self.n)) % self.n

def orbit_idx_d4(k: int) -> Tensor:
    c2 = k - 1
    idx = torch.empty(k, k, dtype=torch.int64)
    key2id = {}
    next_id = 0

    for i in range(k):
        x = 2 * i - c2
        for j in range(k):
            y = 2 * j - c2
            a, b = sorted((abs(x), abs(y))) # orbit invariant
            key = (a, b)
            if key not in key2id:
                key2id[key] = next_id
                next_id += 1
            idx[i, j] = key2id[key]
    return idx
    
def orbit_idx_c4(k: int) -> Tensor:
    c2 = k - 1
    idx = torch.empty(k, k, dtype=torch.int64)
    rep2id = {}
    next_id = 0

    for i in range(k):
        x = 2 * i - c2
        for j in range(k):
            y = 2 * j - c2
            rot = ((x, y), (-y, x), (-x, -y), (y, -x)) # 4 rotations
            a, b = max(rot, key=lambda t: (t[1], t[0])) # canonical rep
            key = (a, b)
            if key not in rep2id:
                rep2id[key] = next_id
                next_id += 1
            idx[i, j] = rep2id[key]
    return idx

def calc_spatial_idx(mats: Tensor, h: int, w: int) -> Tensor:
    # mats: (G, 2, 2)
    center = 0.5 * (torch.tensor([h, w]) - 1)
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coords = torch.stack([y.flatten(), x.flatten()], dim=1).float() - center # (HW, 2)
    coords_c = coords @ mats + center # (G, HW, 2)
    coords_c = coords_c.clamp(torch.tensor(0), 2 * center).round()
    return (coords_c[..., 0] * w + coords_c[..., 1]).long() # (G, HW)

def invert_perm(perm: Tensor) -> Tensor:
    inv = torch.empty_like(perm)
    ar = torch.arange(perm.shape[-1], device=perm.device, dtype=perm.dtype).expand_as(perm)
    inv.scatter_(-1, perm, ar)
    return inv
