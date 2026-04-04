#include "tables.h"
#include <algorithm>
#include <bit>
#include <exception> // std::terminate

namespace bb {

namespace {

// bitboard symmetries (8x8 packed into u64; bit i = (r*8+c))
inline u64 vflip(u64 x) noexcept {
  x = ((x & 0x5555555555555555ULL) << 1) | ((x >> 1) & 0x5555555555555555ULL);
  x = ((x & 0x3333333333333333ULL) << 2) | ((x >> 2) & 0x3333333333333333ULL);
  x = ((x & 0x0F0F0F0F0F0F0F0FULL) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL);
  return x;
}

inline u64 hflip(u64 x) noexcept {
  x = ((x & 0x00FF00FF00FF00FFULL) << 8)  | ((x >> 8)  & 0x00FF00FF00FF00FFULL);
  x = ((x & 0x0000FFFF0000FFFFULL) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
  x = ((x & 0x00000000FFFFFFFFULL) << 32) | (x >> 32);
  return x;
}

inline u64 dflip(u64 x) noexcept {
  u64 t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AAULL;
  x ^= t ^ (t << 7);
  t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCULL;
  x ^= t ^ (t << 14);
  t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0ULL;
  x ^= t ^ (t << 28);
  return x;
}

inline u64 adflip(u64 x) noexcept {
  u64 t = x ^ (x << 36);
  x ^= (t ^ (x >> 36)) & 0xF0F0F0F00F0F0F0FULL;
  t = (x ^ (x << 18)) & 0xCCCC0000CCCC0000ULL;
  x ^= t ^ (t >> 18);
  t = (x ^ (x << 9)) & 0xAA00AA00AA00AA00ULL;
  x ^= t ^ (t >> 9);
  return x;
}

inline u64 rot180(u64 x) noexcept { return vflip(hflip(x)); }
inline u64 rot90(u64 x)  noexcept { return dflip(vflip(x)); }
inline u64 rot270(u64 x) noexcept { return dflip(hflip(x)); }

// (e, r, r2, r3, s, rs, r2s, r3s) with r=90ccw, s=vflip
inline u64 apply_g(u64 x, int g) noexcept {
  switch (g) {
    case 0: return x;
    case 1: return rot90(x);
    case 2: return rot180(x);
    case 3: return rot270(x);
    case 4: return vflip(x);
    case 5: return dflip(x);
    case 6: return hflip(x);
    default: return adflip(x);
  }
}

// shift a shape so its min row/col becomes 0,0
inline u64 normalize_lr(u64 x) noexcept {
  int min_r = 7, min_c = 7;
  u64 y = x;
  while (y) {
    u64 lsb = y & (~y + 1); // lowbit
    int p = std::countr_zero(lsb);
    int r = p >> 3;
    int c = p & 7;
    min_r = std::min(min_r, r);
    min_c = std::min(min_c, c);
    y ^= lsb;
  }
  int sh = (min_r << 3) + min_c;
  return x >> sh;
}

[[noreturn]] inline void die() noexcept {
  std::terminate();
}

} // namespace

const Tables& Tables::get() {
  static const Tables T{};
  return T;
}

Tables::Tables() {
  // arrays are value-initialized to 0 via {} in the header; no need to fill() here

  // gmap[i][g]
  std::array<std::array<u8, 8>, 64> gmap{}; // how anchor index i transforms under g
  for (int i = 0; i < 64; ++i) {
    u64 x = (1ULL << i);
    for (int g = 0; g < 8; ++g) {
      u64 y = apply_g(x, g);
      gmap[i][g] = static_cast<u8>(std::countr_zero(y));
    }
  }

  // Dedup within each canonical: build pose_off, g2pose, blocks, and rep-g lists.
  std::array<std::array<int, 8>, N_CANON> reps_g{}; // reps_g[b][pi] = representative group element
  std::array<u8, N_CANON> reps_len{};               // number of unique poses for canonical b

  pose_off[0] = 0;
  int global_pid = 0;

  for (int b = 0; b < N_CANON; ++b) {
    std::array<u64, 8> seen_shapes{};
    int seen_n = 0;

    std::array<int, 8> reps{};
    int reps_n = 0;

    for (int g = 0; g < 8; ++g) {
      u64 bg = normalize_lr(apply_g(CANONICAL[b], g));

      int p = -1;
      for (int i = 0; i < seen_n; ++i) {
        if (seen_shapes[i] == bg) { p = i; break; }
      }

      if (p < 0) {
        p = seen_n++;
        seen_shapes[p] = bg;
        reps[reps_n++] = g;

        if (global_pid >= N_BLOCKS) die();
        blocks[global_pid++] = bg; // global pose order matches python
      }

      g2pose[b * 8 + g] = static_cast<u8>(p);
    }

    reps_len[b] = static_cast<u8>(reps_n);
    for (int i = 0; i < reps_n; ++i) reps_g[b][i] = reps[i];

    pose_off[b + 1] = static_cast<u8>(pose_off[b] + reps_n);
  }

  for (int g = 0; g < 8; ++g) {
    g2pose[N_CANON * 8 + g] = static_cast<u8>(0); // pad with zeros for null block
  }

  // Hard-check that the fixed-size assumption is true (even in release builds)
  if (pose_off[N_CANON] != N_BLOCKS) die();
  if (global_pid != N_BLOCKS) die();

  // pid2b / pid2g
  for (int b = 0; b < N_CANON; ++b) {
    int base = pose_off[b];
    int n = reps_len[b];
    for (int pi = 0; pi < n; ++pi) {
      int pid = base + pi;
      pid2b[pid] = static_cast<u8>(b);
      pid2g[pid] = static_cast<u8>(reps_g[b][pi]);
    }
  }
  pid2b[N_BLOCKS] = static_cast<u8>(N_CANON); // NULL_ID marker
  pid2g[N_BLOCKS] = 0;

  // Build per-canonical derived tables (placement/touch/deltas/cand_mask/pop)
  for (int b = 0; b < N_CANON; ++b) {
    u64 block = CANONICAL[b];
    const int pop = std::popcount(block);

    // offs = bit indices of occupied cells (max 9)
    int offs[9];
    int n_offs = 0;
    {
      u64 x = block;
      while (x) {
        u64 lsb = x & (~x + 1);
        int p = std::countr_zero(lsb);
        offs[n_offs++] = p;
        x ^= lsb;
      }
    }
    if (n_offs != pop) die();

    // Candidate anchor mask on empty board (canonical orientation)
    int maxr = 0, maxc = 0;
    for (int i = 0; i < n_offs; ++i) {
      int p = offs[i];
      int r = p >> 3, c = p & 7;
      maxr = std::max(maxr, r);
      maxc = std::max(maxc, c);
    }
    const int rows = 8 - maxr;
    const int cols = 8 - maxc;           // 1..8 here
    const u64 rowwin = (1ULL << cols) - 1ULL;

    u64 m = 0;
    for (int r = 0; r < rows; ++r) m |= (rowwin << (r * 8));

    const int base = pose_off[b];
    const int nposes = reps_len[b];

    // touched lines
    for (int j = 0; j < 64; ++j) {
      if (((m >> j) & 1ULL) == 0ULL) continue;
      const u64 x = (block << j);

      for (int pi = 0; pi < nposes; ++pi) {
        const int g = reps_g[b][pi];
        const int bid = base + pi;

        const int pos = gmap[j][g];
        const u64 xg = apply_g(x, g);

        u16 t = 0;
        for (int li = 0; li < 16; ++li) {
          if (xg & LINE[li]) t |= static_cast<u16>(1u << li);
        }
        touch_lines[(std::size_t)bid * 64 + (std::size_t)pos] = t;
      }
    }

    // deltas + cand_mask + pop
    for (int pi = 0; pi < nposes; ++pi) {
      const int g = reps_g[b][pi];
      const int bid = base + pi;

      const int p0 = gmap[0][g]; // anchor after transform
      const int r0 = p0 >> 3;
      const int c0 = p0 & 7;

      for (int i = 0; i < n_offs; ++i) {
        const int pg = gmap[offs[i]][g];
        const int r = pg >> 3, c = pg & 7;
        const int d = (r - r0) * 8 + (c - c0);
        deltas[(std::size_t)bid * 9 + (std::size_t)i] = static_cast<i8>(d);
      }
      
      const u64 mg = apply_g(m, g);
      cand_mask[bid] = mg;
      ref_pos[bid] = std::countr_zero(mg & (~mg + 1));
      block_pop[bid] = static_cast<u8>(pop);
    }
  }
}

} // namespace bb