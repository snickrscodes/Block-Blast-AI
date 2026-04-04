#pragma once
#include <array>
#include <cstdint>
#include <cstddef>

namespace bb {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u16 = std::uint16_t;
using u8  = std::uint8_t;
using i8  = std::int8_t;

struct Tables {
  static constexpr int N_CANON = 15;
  static constexpr int N_BLOCKS = 41; // should ALWAYS come out to 41 after dedup
  static constexpr int NULL_BLOCK = N_BLOCKS;

  static constexpr std::array<u64, N_CANON> CANONICAL = {
    0x3ull, 0x7ull, 0xFull, 0x1Full, 0x70707ull, 0x303ull, 0x1ull, 0x707ull,
    0x103ull, 0x107ull, 0x10107ull, 0x102ull, 0x10204ull, 0x306ull, 0x207ull
  };

  static constexpr std::array<u64, 16> LINE = {
    // rows
    0x00000000000000FFULL,
    0x000000000000FF00ULL,
    0x0000000000FF0000ULL,
    0x00000000FF000000ULL,
    0x000000FF00000000ULL,
    0x0000FF0000000000ULL,
    0x00FF000000000000ULL,
    0xFF00000000000000ULL,

    // cols
    0x0101010101010101ULL,
    0x0202020202020202ULL,
    0x0404040404040404ULL,
    0x0808080808080808ULL,
    0x1010101010101010ULL,
    0x2020202020202020ULL,
    0x4040404040404040ULL,
    0x8080808080808080ULL
  };

  std::array<u8,  N_CANON + 1> pose_off{}; // where each block type lies in the full block list
  std::array<u8,  (N_CANON + 1) * 8> g2pose{}; // (canonical, g) -> pose within canonical
  std::array<u8,  N_BLOCKS + 1> pid2b{}; // (pose id -> canonical id), last = NULL_ID
  std::array<u8,  N_BLOCKS + 1> pid2g{}; // (pose id -> representative g)
  std::array<u8,  N_BLOCKS> block_pop{}; // size of each block
  std::array<u64, N_BLOCKS> cand_mask{}; // legal placements on an empty board (transforms under g)
  std::array<u64, N_BLOCKS> blocks{}; // normalized pose shapes
  std::array<i8,  N_BLOCKS * 9> deltas{}; // signed relative offsets from an anchor, aligns occupancy bits and test legality via shifts
  std::array<u16, N_BLOCKS * 64> touch_lines{}; // lines touched by placement for [bid * 64 + pos]
  std::array<u8, N_BLOCKS> ref_pos{};

  static const Tables& get();

private:
  Tables();
};

} // namespace bb