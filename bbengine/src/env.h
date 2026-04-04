#pragma once
#include <vector>
#include <cstdint>
#include "tables.h"
#include "engine.h"

namespace bb {

using u64 = std::uint64_t;
using u8  = std::uint8_t;


struct SplitMix64 {
  u64 state;
  explicit SplitMix64(u64 seed=0) : state(seed) {}

  inline u64 next_u64() noexcept {
    state += 0x9E3779B97F4A7C15ULL;
    u64 z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }
};

// returns a pose-id in [0, Tables::N_BLOCKS)
inline int rand_block_pose(SplitMix64& rng, const Tables& T) noexcept {
  const u64 r0 = rng.next_u64();
  const int canon = (int)(r0 % Tables::N_CANON);

  const int base = (int)T.pose_off[canon];
  const int npose = (int)T.pose_off[canon + 1] - base;

  if (npose <= 1) return base;

  const u64 r1 = rng.next_u64();
  return base + (int)(r1 % (u64)npose);
}

struct Env {
  explicit Env(u64 seed=0);
  explicit Env(u64 board, u64 seed);

  void reset() noexcept;

  void set_blocks(int b0, int b1, int b2) noexcept; // pose ids or -1 for NULL
  void rand_blocks() noexcept;

  int step(int action) noexcept;

  void legal_actions(std::vector<int>& out) const;

  // accessors
  u64 board() const noexcept { return board_; }
  u64 meta()  const noexcept { return meta_; }
  u64 m0()    const noexcept { return m0_; }
  u64 m1()    const noexcept { return m1_; }
  u64 m2()    const noexcept { return m2_; }
  u8  done()  const noexcept { return done_; }
  int32_t score() const noexcept { return score_; }

private:
  SplitMix64 rng_;
  const Tables& T_;

  u64 board_ = 0ull;
  u64 meta_  = ZERO_META; // sets initial counter to 3
  int32_t score_ = 0;

  u64 m0_ = 0ull, m1_ = 0ull, m2_ = 0ull;
  u8 done_ = 0;
};

struct BatchEnv {
  explicit BatchEnv(int n_envs, u64 seed=0);

  int size() const noexcept { return n_; }

  void reset(int i) noexcept;
  void reset_all() noexcept;
  void reset_indices(const int* idx, int m) noexcept;

  void set_blocks(int i, int b0, int b1, int b2) noexcept;
  void rand_blocks(int i) noexcept;
  void rand_blocks_indices(const int* idx, int m) noexcept;

  int step(int i, int action) noexcept;
  void step_indices(const int* idx, const int* act, int m, int* rewards_out) noexcept;

  const std::vector<u64>& boards() const noexcept { return board_; }
  const std::vector<u64>& metas()  const noexcept { return meta_; }
  const std::vector<u64>& m0()     const noexcept { return m0_; }
  const std::vector<u64>& m1()     const noexcept { return m1_; }
  const std::vector<u64>& m2()     const noexcept { return m2_; }
  const std::vector<u8>&  done()   const noexcept { return done_; }
  const std::vector<int32_t>& score() const noexcept { return score_; }

  void legal_actions(int i, std::vector<int>& out) const;

private:
  int n_;
  SplitMix64 rng_;
  const Tables& T_;

  std::vector<u64> board_;
  std::vector<u64> meta_;
  std::vector<int32_t> score_;
  std::vector<u64> m0_, m1_, m2_;
  std::vector<u8> done_;
};

} // namespace bb
