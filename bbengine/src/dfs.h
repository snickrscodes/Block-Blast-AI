#pragma once
#include <bit>
#include <cstdint>
#include "tables.h"
#include "engine.h"

namespace bb {

static inline bool all_null(u64 m) noexcept {
  return get_b0(m) == (u64)Tables::NULL_BLOCK; 
}

static inline bool can_finish1(u64 board, u8 b0, const Tables& T) noexcept {
  return legal_mask_for_block_E(~board, (int)b0, T) != 0ull;
}

// exact depth-2 existence check (order matters, so we try both unless duplicate)
static inline bool can_finish2(u64 board, u8 b0, u8 b1, const Tables& T) noexcept {
  const u64 E  = ~board;
  const u64 m0 = legal_mask_for_block_E(E, (int)b0, T);
  const u64 m1 = legal_mask_for_block_E(E, (int)b1, T);

  if (!(m0 | m1)) return false;

  // b0 first
  if (m0) {
    u64 mm = m0;
    while (mm) {
      const int pos = std::countr_zero(mm); 
      mm &= (mm - 1);
      const u64 nb = project_placement(board, (int)b0, pos, T);
      if (can_finish1(nb, b1, T)) return true;
    }
  }

  // b1 first
  if (b1 != b0 && m1) {
    u64 mm = m1;
    while (mm) {
      const int pos = std::countr_zero(mm); 
      mm &= (mm - 1);
      const u64 nb = project_placement(board, (int)b1, pos, T);
      if (can_finish1(nb, b0, T)) return true;
    }
  }

  return false;
}

// exact depth-3 existence check; tries each distinct block as first (if it has a move)
static inline bool can_finish3(u64 board, u8 b0, u8 b1, u8 b2, const Tables& T) noexcept {
  const u64 E  = ~board;
  const u64 m0 = legal_mask_for_block_E(E, (int)b0, T);
  const u64 m1 = legal_mask_for_block_E(E, (int)b1, T);
  const u64 m2 = legal_mask_for_block_E(E, (int)b2, T);

  if (!(m0 | m1 | m2)) return false;

  // b0 first
  if (m0) {
    u64 mm = m0;
    while (mm) {
      const int pos = std::countr_zero(mm); 
      mm &= (mm - 1);
      const u64 nb = project_placement(board, (int)b0, pos, T);
      if (can_finish2(nb, b1, b2, T)) return true;
    }
  }

  // b1 first (skip if same id as b0)
  if (b1 != b0 && m1) {
    u64 mm = m1;
    while (mm) {
      const int pos = std::countr_zero(mm); 
      mm &= (mm - 1);
      const u64 nb = project_placement(board, (int)b1, pos, T);
      if (can_finish2(nb, b0, b2, T)) return true;
    }
  }

  // b2 first (skip if same id as b0 or b1)
  if (b2 != b0 && b2 != b1 && m2) {
    u64 mm = m2;
    while (mm) {
      const int pos = std::countr_zero(mm); 
      mm &= (mm - 1);
      const u64 nb = project_placement(board, (int)b2, pos, T);
      if (can_finish2(nb, b0, b1, T)) return true;
    }
  }

  return false;
}

// Max discounted in-hand reward achievable in <=depth moves.
static inline float quiescence_max(u64 board, u64 meta, const Tables& T, float gamma, int depth) noexcept {
  if (depth <= 0) return 0.0f;
  if (all_null(meta)) return 0.0f;

  const u64 E = ~board;

  const u64 b0 = get_b0(meta);
  const u64 b1 = get_b1(meta);
  const u64 b2 = get_b2(meta);

  u64 m0 = (b0 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)b0, T) : 0ull;
  u64 m1 = (b1 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)b1, T) : 0ull;
  u64 m2 = (b2 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)b2, T) : 0ull;

  if (!(m0 | m1 | m2)) return 0.0f;
  float best = 0.0f;

  // slot 0
  while (m0) {
    const int pos = std::countr_zero(m0); 
    m0 &= (m0 - 1);
    const ProjectOut o = project(board, meta, pos, T);
    const float r = (float)o.reward;

    float cont = 0.0f;
    if (depth > 1 && !o.done && !all_null(o.meta)) {
      cont = gamma * quiescence_max(o.board, o.meta, T, gamma, depth - 1);
    }
    const float v = r + cont;
    if (v > best) best = v;
  }

  // slot 1
  while (m1) {
    const int pos = std::countr_zero(m1); 
    m1 &= (m1 - 1);
    const ProjectOut o = project(board, meta, 64 + pos, T);
    const float r = (float)o.reward;

    float cont = 0.0f;
    if (depth > 1 && !o.done && !all_null(o.meta)) {
      cont = gamma * quiescence_max(o.board, o.meta, T, gamma, depth - 1);
    }
    const float v = r + cont;
    if (v > best) best = v;
  }

  // slot 2
  while (m2) {
    const int pos = std::countr_zero(m2);
    m2 &= (m2 - 1);
    const ProjectOut o = project(board, meta, 128 + pos, T);
    const float r = (float)o.reward;

    float cont = 0.0f;
    if (depth > 1 && !o.done && !all_null(o.meta)) {
      cont = gamma * quiescence_max(o.board, o.meta, T, gamma, depth - 1);
    }
    const float v = r + cont;
    if (v > best) best = v;
  }

  return best;
}

} // namespace bb