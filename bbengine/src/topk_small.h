#pragma once
#include <bit>
#include <span>
#include <algorithm>
#include <cstddef>
#include <cmath>      // log
#include <cstdint>

#include "engine.h"
#include "tables.h"

namespace bb {

// Action space: 3 slots * 64 anchors
inline constexpr int kSlotSize = 64;
inline constexpr int kActions  = 3 * kSlotSize; // 192

inline void canonicalize_slot_masks(u64 meta, u64& m0, u64& m1, u64& m2) noexcept {
  const u64 b0 = get_b0(meta);
  const u64 b1 = get_b1(meta);
  const u64 b2 = get_b2(meta);
  const u64 nullb = static_cast<u64>(Tables::NULL_BLOCK);

  const bool b0_ok = (b0 != nullb);
  const bool b2_ok = (b2 != nullb);

  // earliest wins
  if (b0_ok && (b0 == b1)) m1 = 0;
  if (b2_ok && (b0 == b2 || b1 == b2)) m2 = 0;
}

template<int MAX_M = 32>
struct TopM {
  int   act[MAX_M];
  float sc [MAX_M];
  int n = 0;
  int M = 0;

  inline void reset(int M_) noexcept {
    n = 0;
    M = (M_ <= 0) ? 0 : (M_ > MAX_M ? MAX_M : M_);
  }

  inline void push(int a, float s) noexcept {
    if (M <= 0) return;

    if (n < M) [[likely]] {
      act[n] = a; sc[n] = s; ++n;
      // keep ascending so sc[0] is current min
      for (int i = n - 1; i > 0; --i) {
        if (sc[i] >= sc[i - 1]) break;
        std::swap(sc[i], sc[i - 1]);
        std::swap(act[i], act[i - 1]);
      }
      return;
    }

    if (s <= sc[0]) return;  // not better than current min
    sc[0]  = s;
    act[0] = a;

    // bubble up to restore ascending order
    for (int i = 0; i + 1 < n; ++i) {
      if (sc[i] <= sc[i + 1]) break;
      std::swap(sc[i], sc[i + 1]);
      std::swap(act[i], act[i + 1]);
    }
  }

  inline int write_desc(std::span<int> out) const noexcept {
    const int cnt = n;
    const int cap = (int)out.size();
    const int w   = (cnt <= cap) ? cnt : cap;
    for (int i = 0; i < w; ++i) out[i] = act[cnt - 1 - i];
    return w;
  }
};

// ---- tiny RNG + gumbel ----
// Stateful splitmix64 variant (fast, deterministic, header-only).
inline std::uint64_t splitmix64_next(std::uint64_t& x) noexcept {
  x += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

inline float uniform01(std::uint64_t& rng) noexcept {
  // 24 random bits -> (0,1); never 0, never 1.
  const std::uint64_t r = splitmix64_next(rng);
  const std::uint32_t x = (std::uint32_t)(r >> 40); // top 24 bits
  return (float)(((double)x + 0.5) * (1.0 / 16777216.0)); // 2^24
}

inline float gumbel01(std::uint64_t& rng) noexcept {
  const float u = uniform01(rng); // strictly in (0,1)
  return -std::log(-std::log(u));
}

template<int MAX_M>
inline void push_mask_actions_scores(
    u64 mask, int slot_off,
    std::span<const float> scores,
    TopM<MAX_M>& top) noexcept
{
  while (mask) {
    const int bit = std::countr_zero(mask);
    mask &= (mask - 1);
    const int a = slot_off + bit;
    top.push(a, scores[(std::size_t)a]);
  }
}

template<int MAX_M>
inline void push_mask_actions_gumbel(
    u64 mask, int slot_off,
    std::span<const float> logits,
    float tau,
    std::uint64_t& rng,
    TopM<MAX_M>& top) noexcept
{
  while (mask) {
    const int bit = std::countr_zero(mask);
    mask &= (mask - 1);
    const int a = slot_off + bit;
    const float key = logits[(std::size_t)a] + tau * gumbel01(rng);
    top.push(a, key);
  }
}

// Deterministic top-M on arbitrary "scores" (logits are fine).
template<int MAX_M = 32>
inline int select_top_m_from_masks_scores(
    u64 m0, u64 m1, u64 m2,
    u64 meta,
    std::span<const float> scores,
    int M,
    std::span<int> out_actions) noexcept
{
  canonicalize_slot_masks(meta, m0, m1, m2);

  TopM<MAX_M> top;
  top.reset(M);

  if (m0) push_mask_actions_scores<MAX_M>(m0, 0,             scores, top);
  if (m1) push_mask_actions_scores<MAX_M>(m1, kSlotSize,     scores, top);
  if (m2) push_mask_actions_scores<MAX_M>(m2, 2 * kSlotSize, scores, top);

  return top.write_desc(out_actions);
}

// Gumbel-top-M on logits (tau=0 => deterministic path).
template<int MAX_M = 32>
inline int select_top_m_from_masks_gumbel(
    u64 m0, u64 m1, u64 m2,
    u64 meta,
    std::span<const float> logits,
    int M,
    float tau,
    std::uint64_t& rng,
    std::span<int> out_actions) noexcept
{
  if (tau <= 0.0f) {
    return select_top_m_from_masks_scores<MAX_M>(m0, m1, m2, meta, logits, M, out_actions);
  }

  canonicalize_slot_masks(meta, m0, m1, m2);

  TopM<MAX_M> top;
  top.reset(M);

  if (m0) push_mask_actions_gumbel<MAX_M>(m0, 0,             logits, tau, rng, top);
  if (m1) push_mask_actions_gumbel<MAX_M>(m1, kSlotSize,     logits, tau, rng, top);
  if (m2) push_mask_actions_gumbel<MAX_M>(m2, 2 * kSlotSize, logits, tau, rng, top);

  return top.write_desc(out_actions);
}

} // namespace bb
