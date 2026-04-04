#pragma once
#include <cstdint>
#include "tables.h"

namespace bb {

inline constexpr u64 COMBO_MASK = (1ULL << 37) - 1ULL; // bits 0-36
inline constexpr u64 CTR_MASK   = (1ULL << 3)  - 1ULL; // bits 37-39
inline constexpr u64 BLOCK_MASK = (1ULL << 8)  - 1ULL; // block ids in 3 slots
inline constexpr int COMBO_SHIFT = 0;
inline constexpr int CTR_SHIFT   = 37;
inline constexpr int B0_SHIFT    = 40;
inline constexpr int B1_SHIFT    = 48;
inline constexpr int B2_SHIFT    = 56;
inline constexpr u64 ZERO_META   = ((3ull & CTR_MASK) << CTR_SHIFT);

inline constexpr std::array<int, 16> LINE_BONUS = []{
  std::array<int, 16> a{};
  for (int i = 0; i < 16; ++i) {
    a[i] = i * ((i > 1) ? (i - 1) : 1) * 10;
  }
  return a;
}();

inline u64 get_combo(u64 m) noexcept { return (m >> COMBO_SHIFT) & COMBO_MASK; }
inline u64 get_ctr(u64 m) noexcept { return (m >> CTR_SHIFT) & CTR_MASK; }

inline u64 get_b0(u64 m) noexcept { return (m >> B0_SHIFT) & BLOCK_MASK; }
inline u64 get_b1(u64 m) noexcept { return (m >> B1_SHIFT) & BLOCK_MASK; }
inline u64 get_b2(u64 m) noexcept { return (m >> B2_SHIFT) & BLOCK_MASK; }

inline u64 pack_meta(u64 combo, u64 ctr, u64 b0, u64 b1, u64 b2) noexcept {
  return (combo & COMBO_MASK)
      | ((ctr & CTR_MASK) << CTR_SHIFT)
      | ((b0 & BLOCK_MASK) << B0_SHIFT)
      | ((b1 & BLOCK_MASK) << B1_SHIFT)
      | ((b2 & BLOCK_MASK) << B2_SHIFT);
}

struct ProjectOut {
  u64 board;
  u64 meta;
  int reward;
  bool done;
  u64 m0, m1, m2;
};

// E = ~board (empty squares). Returns 64-bit mask of legal anchor positions for pose `bid`.
u64 legal_mask_for_block_E(u64 E, int bid, const Tables& T) noexcept;
ProjectOut project(u64 board, u64 meta, int a, const Tables& T) noexcept;
u64 project_placement(u64 board, int bid, int pos, const Tables& T) noexcept;
}