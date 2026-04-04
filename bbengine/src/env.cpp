#include "env.h"
#include <bit>

namespace bb {

Env::Env(u64 seed) : rng_(seed), T_(Tables::get()) {}
Env::Env(u64 board, u64 seed) : board_(board), rng_(seed), T_(Tables::get()) {}

void Env::reset() noexcept {
  board_ = 0ull;
  meta_  = ZERO_META; // sets initial counter to 3
  score_ = 0;
  m0_ = m1_ = m2_ = 0ull;
  done_ = 0;
}

void Env::set_blocks(int b0, int b1, int b2) noexcept {
  const u64 combo = get_combo(meta_);
  const u64 ctr   = get_ctr(meta_);

  const u64 ub0 = (b0 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b0;
  const u64 ub1 = (b1 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b1;
  const u64 ub2 = (b2 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b2;

  meta_ = pack_meta(combo, ctr, ub0, ub1, ub2);

  const u64 E = ~board_;
  m0_ = (ub0 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub0, T_) : 0ULL;
  m1_ = (ub1 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub1, T_) : 0ULL;
  m2_ = (ub2 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub2, T_) : 0ULL;

  const bool all_null =
    (ub0 == (u64)Tables::NULL_BLOCK) &&
    (ub1 == (u64)Tables::NULL_BLOCK) &&
    (ub2 == (u64)Tables::NULL_BLOCK);

  done_ = (u8)!((m0_ | m1_ | m2_) || all_null);
}

void Env::rand_blocks() noexcept {
  const int b0 = rand_block_pose(rng_, T_);
  const int b1 = rand_block_pose(rng_, T_);
  const int b2 = rand_block_pose(rng_, T_);
  set_blocks(b0, b1, b2);
}

int Env::step(int action) noexcept {
  const ProjectOut out = project(board_, meta_, action, T_);
  board_ = out.board;
  meta_  = out.meta;
  m0_    = out.m0;
  m1_    = out.m1;
  m2_    = out.m2;
  done_  = (u8)out.done;
  score_ += out.reward;
  return out.reward;
}

void Env::legal_actions(std::vector<int>& out) const {
  out.clear();
  u64 m0 = m0_, m1 = m1_, m2 = m2_;

  while (m0) { int pos = std::countr_zero(m0); m0 &= (m0 - 1); out.push_back(pos); }
  while (m1) { int pos = std::countr_zero(m1); m1 &= (m1 - 1); out.push_back(64 + pos); }
  while (m2) { int pos = std::countr_zero(m2); m2 &= (m2 - 1); out.push_back(128 + pos); }
}

BatchEnv::BatchEnv(int n_envs, u64 seed)
: n_(n_envs), rng_(seed), T_(Tables::get()),
  board_(n_envs, 0ULL),
  meta_(n_envs, ZERO_META),
  score_(n_envs, 0),
  m0_(n_envs, 0ULL), m1_(n_envs, 0ULL), m2_(n_envs, 0ULL),
  done_(n_envs, 0)
{}

void BatchEnv::reset(int i) noexcept {
  board_[i] = 0ULL;
  meta_[i]  = ZERO_META;
  score_[i] = 0;
  m0_[i] = m1_[i] = m2_[i] = 0ULL;
  done_[i] = 0;
}

void BatchEnv::reset_all() noexcept {
  for (int i = 0; i < n_; ++i) {
    board_[i] = 0ULL;
    meta_[i]  = ZERO_META;
    score_[i] = 0;
    m0_[i] = m1_[i] = m2_[i] = 0ULL;
    done_[i] = 0;
  }
}

void BatchEnv::reset_indices(const int* idx, int m) noexcept {
  for (int k = 0; k < m; ++k) {
    int i = idx[k];
    board_[i] = 0ULL;
    meta_[i]  = ZERO_META;
    score_[i] = 0;
    m0_[i] = m1_[i] = m2_[i] = 0ULL;
    done_[i] = 0;
  }
}

void BatchEnv::set_blocks(int i, int b0, int b1, int b2) noexcept {
  // preserve combo/ctr, replace b0/b1/b2
  const u64 combo = get_combo(meta_[i]);
  const u64 ctr   = get_ctr(meta_[i]);

  const u64 ub0 = (b0 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b0;
  const u64 ub1 = (b1 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b1;
  const u64 ub2 = (b2 < 0) ? (u64)Tables::NULL_BLOCK : (u64)b2;

  meta_[i] = pack_meta(combo, ctr, ub0, ub1, ub2);

  const u64 E = ~board_[i];

  m0_[i] = (ub0 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub0, T_) : 0ULL;
  m1_[i] = (ub1 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub1, T_) : 0ULL;
  m2_[i] = (ub2 != (u64)Tables::NULL_BLOCK) ? legal_mask_for_block_E(E, (int)ub2, T_) : 0ULL;

  const bool all_null = (ub0 == (u64)Tables::NULL_BLOCK) &&
                        (ub1 == (u64)Tables::NULL_BLOCK) &&
                        (ub2 == (u64)Tables::NULL_BLOCK);

  done_[i] = (u8)!((m0_[i] | m1_[i] | m2_[i]) || all_null);
}

void BatchEnv::rand_blocks(int i) noexcept {
  const int b0 = rand_block_pose(rng_, T_);
  const int b1 = rand_block_pose(rng_, T_);
  const int b2 = rand_block_pose(rng_, T_);
  set_blocks(i, b0, b1, b2);
}

void BatchEnv::rand_blocks_indices(const int* idx, int m) noexcept {
  for (int k = 0; k < m; ++k) rand_blocks(idx[k]);
}

int BatchEnv::step(int i, int action) noexcept {
  // assumes caller doesn't step done envs (or you can allow it; up to you)
  const ProjectOut out = project(board_[i], meta_[i], action, T_);
  board_[i] = out.board;
  meta_[i]  = out.meta;
  m0_[i]    = out.m0;
  m1_[i]    = out.m1;
  m2_[i]    = out.m2;
  done_[i]  = (u8)out.done;
  score_[i] += out.reward;
  return out.reward;
}

void BatchEnv::step_indices(const int* idx, const int* act, int m, int* rewards_out) noexcept {
  for (int k = 0; k < m; ++k) {
    const int i = idx[k];
    rewards_out[k] = step(i, act[k]);
  }
}

void BatchEnv::legal_actions(int i, std::vector<int>& out) const {
  out.clear();
  u64 m0 = m0_[i], m1 = m1_[i], m2 = m2_[i];

  while (m0) {
    u64 lsb = m0 & (~m0 + 1);
    int pos = std::countr_zero(lsb);
    out.push_back(pos);
    m0 ^= lsb;
  }
  while (m1) {
    u64 lsb = m1 & (~m1 + 1);
    int pos = std::countr_zero(lsb);
    out.push_back(64 + pos);
    m1 ^= lsb;
  }
  while (m2) {
    u64 lsb = m2 & (~m2 + 1);
    int pos = std::countr_zero(lsb);
    out.push_back(128 + pos);
    m2 ^= lsb;
  }
}

} // namespace bb
