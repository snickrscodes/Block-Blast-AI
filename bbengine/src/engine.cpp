#include "engine.h"
#include <bit>

namespace bb {

u64 legal_mask_for_block_E(u64 E, int bid, const Tables& T) noexcept {
  u64 mask = T.cand_mask[bid];
  const u8 n = T.block_pop[bid];

  const std::size_t base = (std::size_t)bid * 9;
  for (u8 i = 0; i < n; ++i) {
    const int d = (int)T.deltas[base + i];
    mask &= (d >= 0) ? (E >> d) : (E << (-d));
    if (!mask) break;
  }
  return mask;
}

ProjectOut project(u64 board, u64 meta, int a, const Tables& T) noexcept {
  const int slot = a >> 6;
  const int pos  = a & 63;

  u64 b0 = (meta >> B0_SHIFT) & BLOCK_MASK;
  u64 b1 = (meta >> B1_SHIFT) & BLOCK_MASK;
  u64 b2 = (meta >> B2_SHIFT) & BLOCK_MASK;
  u64 combo = (meta >> COMBO_SHIFT) & COMBO_MASK;
  u64 ctr = (meta >> CTR_SHIFT) & CTR_MASK;

  const int bid = (slot == 0) ? (int)b0 : (slot == 1) ? (int)b1 : (int)b2;

  const u64 move_bits = T.blocks[bid] << (pos - T.ref_pos[bid]);
  u64 new_board = board | move_bits;

  // clear full lines (only check touched lines)
  u64 full_mask = 0ull;
  int lines = 0;

  u16 touched = T.touch_lines[(std::size_t)bid * 64 + pos];
  while (touched) {
    const unsigned li = std::countr_zero(touched);
    touched &= (u16)(touched - 1);

    const u64 m = T.LINE[li];
    if ((new_board & m) == m) {
      ++lines;
      full_mask |= m;
    }
  }
  new_board &= ~full_mask;
  const u64 E = ~new_board; // empty squares

  // consume the used slot (and shift blocks)
  if (slot == 0) {
    b0 = b1;
    b1 = b2;
  } else if (slot == 1) b1 = b2;
  b2 = (u64)Tables::NULL_BLOCK;

  u64 new_ctr = ctr - 1;
  u64 new_combo = combo;
  int reward = (int)T.block_pop[bid];

  if (lines > 0) {
    new_combo++;
    new_ctr = 3 + (b0 != b2) + (b1 != b2);
    reward += (int)new_combo * LINE_BONUS[lines];
  } else if (ctr == 1) {
    new_ctr = 3;
    new_combo = 0;
  }

  const u64 m0 = (b0 != b2) ? legal_mask_for_block_E(E, (int)b0, T) : 0ull;
  const u64 m1 = (b1 != b2) ? legal_mask_for_block_E(E, (int)b1, T) : 0ull;

  const bool done = !((b0 == b2) || (m0 | m1));
  if (new_board == 0ull) reward += 300;
  const u64 new_meta = new_combo|(new_ctr<<CTR_SHIFT)|(b0<<B0_SHIFT)|(b1<<B1_SHIFT)|(b2<<B2_SHIFT);
  return ProjectOut{ new_board, new_meta, reward, done, m0, m1, 0ull };
}

u64 project_placement(u64 board, int bid, int pos, const bb::Tables& T) noexcept {
  const u64 move_bits = T.blocks[bid] << (pos - T.ref_pos[bid]);
  u64 new_board = board | move_bits;

  u64 full_mask = 0ull;
  u16 touched = T.touch_lines[(std::size_t)bid * 64 + (std::size_t)pos];
  while (touched) {
    const unsigned li = std::countr_zero(touched);
    touched &= (u16)(touched - 1);

    const bb::u64 m = T.LINE[li];
    if ((new_board & m) == m) full_mask |= m;
  }
  return new_board & ~full_mask;
}

}