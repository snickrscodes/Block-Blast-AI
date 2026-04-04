#pragma once
#include <cstdint>
#include <vector>
#include <span>
#include <functional>

#include "tables.h"        // Tables::NULL_BLOCK (and shifts/masks if you prefer)
#include "topk_small.h"    // bb::select_top_m_from_masks(...)
#include "dedup.h"         // bb::BeamState / dedup_topk_per_env

namespace bb {

using u64 = std::uint64_t;

static constexpr int MAX_M = 192;

struct SearchConfig {
  int depth = 3;
  int beam_width = 32;
  int per_parent_top_m = 16;
  float gamma = 0.95f; // discounts bootstrap
  float teacher_tau = 2.0f;
  float root_eps = 0.5f; // root noise
  int max_eval_P = 2048;  // chunk size for policy
  int max_eval_V = 2048;  // chunk size for value
};

struct StateKey {
  u64 board;
  u64 meta;

  friend constexpr bool operator==(const StateKey& a, const StateKey& b) noexcept {
    return a.board == b.board && a.meta == b.meta;
  }
};

struct EvalCallbacks {
  // Fill out_logits with size (N*192).
  // Convention: logits are unmasked; C++ will mask+softmax using legal masks.
  std::function<void(std::span<const u64> boards,
                     std::span<const u64> metas,
                     std::span<float> out_logits)>
    policy_logits;

  // Fill out_values with size N, aligned to keys.
  std::function<void(std::span<const StateKey> keys,
                     std::span<float> out_values)>
    value;
};

class BeamSearch {
public:
  explicit BeamSearch(SearchConfig cfg = {}) : cfg_(cfg) {}

  SearchConfig& cfg() noexcept { return cfg_; }
  const SearchConfig& cfg() const noexcept { return cfg_; }

  // Output: out_pi_behavior is (N*192), row-major.
  void search_batch(std::span<const u64> boards,
                    std::span<const u64> metas,
                    const EvalCallbacks& cb,
                    bool use_noise,
                    std::uint64_t rng_seed,
                    std::vector<float>& out_pi_behavior);

private:
  SearchConfig cfg_;
  std::size_t tt_reserved_states_ = 0;
};

} // namespace bb