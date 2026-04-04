#pragma once
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <limits>
#include <bit>
#include <cassert>

namespace bb {

using u64 = std::uint64_t;
using u32 = std::uint32_t;

struct BeamState {
  int env;
  int root_action;
  u64 board;
  u64 meta;
  u64 m0, m1, m2;   // NEW
  float q;

  BeamState() = default;

  BeamState(int env_, u64 board_, u64 meta_, u64 m0_, u64 m1_, u64 m2_,
            float q_, int root_action_) noexcept
  : env(env_), root_action(root_action_),
    board(board_), meta(meta_), m0(m0_), m1(m1_), m2(m2_),
    q(q_) {}
};

static_assert(std::is_trivially_copyable_v<BeamState>);

// ------------------------------
// Hashing (splitmix64)
// ------------------------------
inline u64 splitmix64(u64 x) noexcept {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline u64 hash_state(u64 board, u64 meta) noexcept {
  // mix meta into board, then mix again
  u64 x = board ^ splitmix64(meta + 0xD6E8FEB86659FD93ULL);
  return splitmix64(x);
}

struct ChildKey {
  int env;
  u64 board;
  u64 meta;

  friend constexpr bool operator==(const ChildKey& a, const ChildKey& b) noexcept {
    return a.env == b.env && a.board == b.board && a.meta == b.meta;
  }
};

static_assert(std::is_trivially_copyable_v<ChildKey>);

inline u64 hash_child_key(const ChildKey& k) noexcept {
  const u64 e = static_cast<u64>(static_cast<u32>(k.env));
  u64 x = k.board ^ splitmix64(k.meta + 0xD6E8FEB86659FD93ULL);
  x ^= splitmix64(e + 0xA5A5A5A5C3C3C3C3ULL);
  return splitmix64(x);
}

// ------------------------------
// FlatChildMap (same as before)
// ------------------------------
class FlatChildMap {
public:
  void clear() noexcept {
    ++gen_;
    size_ = 0;
    if (gen_ == 0) {
      std::fill(stamp_.begin(), stamp_.end(), 0u);
      gen_ = 1;
    }
  }

  void reset() {
    std::vector<ChildKey>().swap(keys_);
    std::vector<u64>().swap(hash_);
    std::vector<u32>().swap(val_);
    std::vector<u32>().swap(stamp_);
    size_ = 0;
    gen_ = 1;
  }

  void reserve(std::size_t expected, float max_load = 0.70f) {
    max_load_ = (max_load > 0.10f && max_load < 0.95f) ? max_load : 0.70f;
    const std::size_t need = (std::size_t)(float(expected) / max_load_) + 1;
    const std::size_t cap = std::max<std::size_t>(8, std::bit_ceil(need));
    if (cap > capacity()) rehash_(cap);
  }

  [[nodiscard]] std::size_t capacity() const noexcept { return keys_.size(); }

  u32 find_or_insert(const ChildKey& k, u32 value, bool& inserted) {
    if (keys_.empty()) rehash_(8);

    if ((float)(size_ + 1) > max_load_ * (float)capacity()) {
      rehash_(capacity() * 2);
    }

    const u64 h = hash_child_key(k);
    const std::size_t mask = capacity() - 1;
    std::size_t pos = (std::size_t)h & mask;

    for (;;) {
      if (stamp_[pos] != gen_) {
        stamp_[pos] = gen_;
        keys_[pos]  = k;
        hash_[pos]  = h;
        val_[pos]   = value;
        ++size_;
        inserted = true;
        return value;
      }
      if (hash_[pos] == h && keys_[pos] == k) {
        inserted = false;
        return val_[pos];
      }
      pos = (pos + 1) & mask;
    }
  }

private:
  void rehash_(std::size_t new_cap) {
    new_cap = std::max<std::size_t>(8, std::bit_ceil(new_cap));
    assert((new_cap & (new_cap - 1)) == 0);

    auto old_keys  = std::move(keys_);
    auto old_hash  = std::move(hash_);
    auto old_val   = std::move(val_);
    auto old_stamp = std::move(stamp_);
    const u32 old_gen = gen_;

    keys_.assign(new_cap, ChildKey{0, 0, 0});
    hash_.assign(new_cap, 0ULL);
    val_.assign(new_cap, 0u);
    stamp_.assign(new_cap, 0u);

    size_ = 0;
    gen_ = 1;

    if (old_keys.empty()) return;

    const std::size_t old_cap = old_keys.size();
    for (std::size_t i = 0; i < old_cap; ++i) {
      if (old_stamp[i] != old_gen) continue;

      const ChildKey& k = old_keys[i];
      const u64 h = old_hash[i];
      const u32 v = old_val[i];

      const std::size_t mask = new_cap - 1;
      std::size_t pos = (std::size_t)h & mask;
      while (stamp_[pos] == gen_) pos = (pos + 1) & mask;

      stamp_[pos] = gen_;
      keys_[pos]  = k;
      hash_[pos]  = h;
      val_[pos]   = v;
      ++size_;
    }
  }

private:
  std::vector<ChildKey> keys_;
  std::vector<u64>      hash_;
  std::vector<u32>      val_;
  std::vector<u32>      stamp_;

  std::size_t size_ = 0;
  u32 gen_ = 1;
  float max_load_ = 0.70f;
};

// score-desc comparator with deterministic tie-break
inline bool sc_desc(const BeamState& a, const BeamState& b) noexcept {
  if (a.q  > b.q ) return true;
  if (a.q  < b.q ) return false;
  if (a.board != b.board) return a.board > b.board;
  return a.meta > b.meta;
}

inline void dedup_topk_per_env(std::vector<BeamState>& children,
                            int K,
                            int n_envs, // pass N from search; avoids max_env scan
                            std::vector<BeamState>& kept) {
  kept.clear();
  if (children.empty()) return;

  struct Scratch {
    FlatChildMap map;
    std::vector<BeamState> uniq;

    std::vector<u32> counts;
    std::vector<u32> offsets;
    std::vector<u32> cursor;

    std::vector<u32> by_env_idx; // <- indices into uniq
  };
  thread_local Scratch S;

  // 1) Hash dedup: keep best-(sc, then q) per (env, board, meta)
  S.map.clear();
  S.uniq.clear();
  S.uniq.reserve(children.size());
  S.map.reserve(children.size(), 0.70f);

  for (const auto& cs : children) {
    const ChildKey key{cs.env, cs.board, cs.meta};
    bool inserted = false;
    const u32 next = (u32)S.uniq.size();
    const u32 idx  = S.map.find_or_insert(key, next, inserted);

    if (inserted) {
      S.uniq.push_back(cs);
    } else {
      auto& u = S.uniq[idx];
      if (cs.q > u.q) u = cs;
    }
  }

  if (S.uniq.empty() || K <= 0) return;

  // 2) Bucket uniq INDICES by env
  const std::size_t E = (std::size_t)n_envs; // trust caller
  S.counts.assign(E, 0u);

  for (u32 i = 0; i < (u32)S.uniq.size(); ++i) {
    const int env = S.uniq[i].env;
    // If env always in [0, n_envs), skip check. Otherwise clamp/assert.
    S.counts[(std::size_t)env] += 1u;
  }

  S.offsets.resize(E + 1);
  S.offsets[0] = 0;
  for (std::size_t e = 0; e < E; ++e) S.offsets[e + 1] = S.offsets[e] + S.counts[e];

  S.cursor = S.offsets;
  S.by_env_idx.resize(S.uniq.size());

  for (u32 i = 0; i < (u32)S.uniq.size(); ++i) {
    const std::size_t e = (std::size_t)S.uniq[i].env;
    const u32 pos = S.cursor[e]++;
    S.by_env_idx[(std::size_t)pos] = i;
  }

  // 3) Per-env top-K on indices
  kept.reserve(std::min<std::size_t>(S.uniq.size(), E * (std::size_t)K));

  auto idx_desc = [&](u32 ia, u32 ib) noexcept {
    return sc_desc(S.uniq[ia], S.uniq[ib]);
  };

  for (std::size_t e = 0; e < E; ++e) {
    const std::size_t s = (std::size_t)S.offsets[e];
    const std::size_t t = (std::size_t)S.offsets[e + 1];
    const std::size_t len = t - s;
    if (len == 0) continue;

    const std::size_t take = std::min<std::size_t>(len, (std::size_t)K);

    auto begin = S.by_env_idx.begin() + (std::ptrdiff_t)s;
    auto end   = S.by_env_idx.begin() + (std::ptrdiff_t)t;

    if (len > take) {
      std::nth_element(begin, begin + (std::ptrdiff_t)take, end, idx_desc);
    }
    // kept
    for (std::size_t i = 0; i < take; ++i) {
      const BeamState& cs = S.uniq[(std::size_t)S.by_env_idx[s + i]];
      kept.push_back(BeamState{cs.env, cs.board, cs.meta, cs.m0, cs.m1, cs.m2, cs.q, cs.root_action});
    }
  }
}
} // namespace bb
