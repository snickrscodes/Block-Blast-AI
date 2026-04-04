// search.cpp (hard max backups + no finalize-on-prune anywhere + pure top-k everywhere; root differs only by noise)
//
// Key behavior:
//  - HARD MAX backups per (env, root_action).
//  - NO finalize-on-prune (including at root).
//  - Backups ONLY happen at:
//      (a) terminal leaves (done): q_prefix
//      (b) frontier leaves (no legal moves but not done): q_prefix + gamma*V(state)
//      (c) horizon (remaining beams after depth limit): q_prefix + gamma*V(state)
//  - Selection is uniform across depths:
//      * Per parent: select top-M actions from legality masks using a score row.
//        - At root: score row = (optionally noised) logits for that env
//        - At d>0:  score row = policy logits for that beam state
//      * After expansion: dedup + keep top-K per env by sc=q_prefix (pure value/top-k everywhere).
//
// Notes:
//  - Noise only affects root action selection set (top-M) via logits (gumbel noise)
//  - Teacher still uses root logits + L to build pi_behavior.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

#include "search.h"
#include "engine.h"
#include "tables.h"
#include "topk_small.h"
#include "dedup.h"

namespace bb {

namespace {

inline bool any_legal(u64 m0, u64 m1, u64 m2) noexcept { return (m0 | m1 | m2) != 0ULL; }

inline bool is_full_hand(u64 meta) noexcept {
  const u64 nb = static_cast<u64>(Tables::NULL_BLOCK);
  return get_b2(meta) != nb;
}

inline void compute_legal_masks(u64 board, u64 meta, const Tables& T, u64& m0, u64& m1, u64& m2) noexcept {
  const u64 nb = static_cast<u64>(Tables::NULL_BLOCK);
  const u64 E = ~board;

  const u64 b0 = get_b0(meta);
  const u64 b1 = get_b1(meta);
  const u64 b2 = get_b2(meta);

  m0 = (b0 != nb) ? legal_mask_for_block_E(E, (int)b0, T) : 0ULL;
  m1 = (b1 != nb) ? legal_mask_for_block_E(E, (int)b1, T) : 0ULL;
  m2 = (b2 != nb) ? legal_mask_for_block_E(E, (int)b2, T) : 0ULL;
}

// Masked softmax for teacher logits = logits + eta * L
inline void masked_softmax_teacher(const float* L_row,
                                  u64 m0, u64 m1, u64 m2,
                                  float eta,
                                  float* out_pi) noexcept {
  std::memset(out_pi, 0, kActions * sizeof(float));

  int acts[192];
  int n = 0;
  while (m0) { int bit = std::countr_zero(m0); m0 &= (m0 - 1); acts[n++] = bit; }
  while (m1) { int bit = std::countr_zero(m1); m1 &= (m1 - 1); acts[n++] = 64  + bit; }
  while (m2) { int bit = std::countr_zero(m2); m2 &= (m2 - 1); acts[n++] = 128 + bit; }
  if (n == 0) return;

  float mx = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    const int a = acts[i];
    mx = std::max(mx, eta * L_row[a]);
  }

  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    const int a = acts[i];
    const float v = std::exp((eta * L_row[a]) - mx);
    out_pi[a] = v;
    sum += v;
  }

  if (sum <= 0.0f) {
    const float u = 1.0f / float(n);
    for (int i = 0; i < n; ++i) out_pi[acts[i]] = u;
    return;
  }

  const float inv = 1.0f / sum;
  for (int i = 0; i < n; ++i) out_pi[acts[i]] *= inv;
}

// chunked policy eval: fills logits_out (n*192)
// caller must provide logits_out.size() == n*kActions
inline void eval_policy_chunked(const EvalCallbacks& cb,
                                std::span<const u64> boards,
                                std::span<const u64> metas,
                                int max_batch,
                                std::span<float> logits_out) {
  const int n = (int)boards.size();
  for (int s0 = 0; s0 < n; s0 += max_batch) {
    const int s1 = std::min(n, s0 + max_batch);
    cb.policy_logits(boards.subspan(s0, s1 - s0),
                     metas.subspan(s0, s1 - s0),
                     logits_out.subspan(std::size_t(s0) * kActions,
                                        std::size_t(s1 - s0) * kActions));
  }
}

template <class T>
inline void ensure_size(std::vector<T>& v, std::size_t n) {
  if (v.size() < n) v.resize(n);
}

// HARD MAX backup accumulator (per env, per root_action)
struct MaxBackup {
  std::vector<float> best; // size N*kActions, -inf sentinel means "unset"

  void init(std::size_t n) {
    best.assign(n, 0.0);
  }

  inline void add(std::size_t ix, float q) noexcept {
    float& cur = best[ix];
    cur = (q > cur) ? q : cur;
  }
};

struct FinalizeItem {
  int env;
  int root_action;
  u64 board;
  u64 meta;
  float q_prefix;
  bool done; // terminal => no bootstrap
};


static_assert(std::is_trivially_copyable_v<StateKey>);

struct Scratch {
  // root
  std::vector<u64>   root_m0, root_m1, root_m2;
  std::vector<float> root_logits;

  // per-depth policy logits for current beams
  std::vector<float> logits_buf;

  std::vector<BeamState> beams;      // current depth beams
  std::vector<BeamState> next_beams; // output of dedup_topk_per_env

  // per-depth
  std::vector<BeamState> expandables;
  std::vector<u64> eval_boards;
  std::vector<u64> eval_metas;

  // finalize (flush once per depth)
  std::vector<FinalizeItem> fin;
  std::vector<StateKey>      fin_keys;
  std::vector<int>           fin_key_item;
  std::vector<float>         fin_vals;

  std::vector<int> group_starts;

  MaxBackup Lb;
};

thread_local Scratch TLS;

} // namespace

void BeamSearch::search_batch(std::span<const u64> boards,
                             std::span<const u64> metas,
                             const EvalCallbacks& cb,
                             bool use_noise,
                             std::uint64_t rng_seed,
                             std::vector<float>& out_pi_behavior) {
  const Tables& T = Tables::get();

  const int N = (int)boards.size();
  if ((int)metas.size() != N) throw std::runtime_error("boards/metas size mismatch");
  if (!cb.policy_logits) throw std::runtime_error("policy_logits callback not set");
  if (!cb.value) throw std::runtime_error("value callback not set");

  Scratch& S = TLS;

  // --- root legality masks ---
  ensure_size(S.root_m0, (std::size_t)N);
  ensure_size(S.root_m1, (std::size_t)N);
  ensure_size(S.root_m2, (std::size_t)N);

  auto root_m0 = std::span<u64>(S.root_m0.data(), (std::size_t)N);
  auto root_m1 = std::span<u64>(S.root_m1.data(), (std::size_t)N);
  auto root_m2 = std::span<u64>(S.root_m2.data(), (std::size_t)N);

  for (int i = 0; i < N; ++i) {
    compute_legal_masks(boards[i], metas[i], T, root_m0[i], root_m1[i], root_m2[i]);
  }

  // root policy logits
  const std::size_t SZ = (std::size_t)N * kActions;
  ensure_size(S.root_logits, SZ);

  eval_policy_chunked(cb, boards, metas, cfg_.max_eval_P,
                      std::span<float>(S.root_logits.data(), SZ));

  std::uint64_t rng_state = rng_seed;

  // --- beams init (root states) ---
  S.beams.clear();
  S.beams.reserve(N);

  for (int i = 0; i < N; ++i) {
    if (!any_legal(root_m0[i], root_m1[i], root_m2[i])) continue;
    S.beams.push_back(BeamState{
      i, boards[i], metas[i],
      root_m0[i], root_m1[i], root_m2[i],
      0.0f, -1
    });
  }

  // --- L accumulator: hard max return per (env, action) ---
  S.Lb.init(SZ);
  auto& Lb = S.Lb;

  auto tt_eval = [&](std::span<const StateKey> new_keys, std::span<float> out_vals) {
    const int n = (int)new_keys.size();
    const int maxb = cfg_.max_eval_V;
    for (int s0 = 0; s0 < n; s0 += maxb) {
      const int s1 = std::min(n, s0 + maxb);
      cb.value(new_keys.subspan(s0, s1 - s0), out_vals.subspan(s0, s1 - s0));
    }
  };

  // finalize helper (flush once per depth)
  auto& fin          = S.fin;
  auto& fin_keys     = S.fin_keys;
  auto& fin_key_item = S.fin_key_item;
  auto& fin_vals     = S.fin_vals;
  auto& group_starts = S.group_starts;

  auto finalize_items = [&](std::span<const FinalizeItem> items) {
    if (items.empty()) return;

    // 1) collect bootstrap candidates (non-terminal)
    fin_key_item.clear();  // reuse as "idx list" into items
    fin_key_item.reserve(items.size());

    for (int i = 0; i < (int)items.size(); ++i) {
      const auto& it = items[(size_t)i];
      const size_t ix = size_t(it.env) * kActions + size_t(it.root_action);

      if (it.done) {
        Lb.add(ix, it.q_prefix);
      } else {
        fin_key_item.push_back(i);
      }
    }

    if (fin_key_item.empty()) return;

    // 2) sort indices by (board, meta)
    auto& idx = fin_key_item;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
      const auto& A = items[(size_t)a];
      const auto& B = items[(size_t)b];
      if (A.board != B.board) return A.board < B.board;
      return A.meta < B.meta;
    });

    // 3) build unique key list + group starts
    fin_keys.clear();     // unique keys
    fin_keys.reserve(idx.size());

    // group_starts[g] = starting position in idx for unique key g
    // group_starts.size() == fin_keys.size() + 1 at end (sentinel)
    group_starts.clear();
    group_starts.reserve(idx.size() + 1);

    auto push_unique = [&](const FinalizeItem& it, int start_pos) {
      fin_keys.push_back(StateKey{it.board, it.meta});
      group_starts.push_back(start_pos);
    };

    // first group
    {
      const auto& it0 = items[(size_t)idx[0]];
      push_unique(it0, 0);
    }

    for (int p = 1; p < (int)idx.size(); ++p) {
      const auto& prev = items[(size_t)idx[p - 1]];
      const auto& cur  = items[(size_t)idx[p]];
      if (cur.board != prev.board || cur.meta != prev.meta) {
        push_unique(cur, p);
      }
    }
    group_starts.push_back((int)idx.size()); // sentinel end

    // 4) eval value on unique keys
    fin_vals.resize(fin_keys.size());
    tt_eval(std::span<const StateKey>(fin_keys.data(), fin_keys.size()),
            std::span<float>(fin_vals.data(), fin_vals.size()));

    // 5) scatter: all items in a group share the same V
    for (int g = 0; g < (int)fin_keys.size(); ++g) {
      const float boot = fin_vals[(size_t)g];
      const int s0 = group_starts[(size_t)g];
      const int s1 = group_starts[(size_t)g + 1];

      for (int p = s0; p < s1; ++p) {
        const int i = idx[(size_t)p];
        const auto& it = items[(size_t)i];

        const float q_final = it.q_prefix + cfg_.gamma * boot;
        const size_t out_ix = size_t(it.env) * kActions + size_t(it.root_action);
        Lb.add(out_ix, q_final);
      }
    }
  };


  // scratch aliases
  auto& logits_buf  = S.logits_buf;
  auto& expandables = S.expandables;
  auto& next_beams  = S.next_beams;
  auto& beams  = S.beams;

  // --------------------
  // depth loop
  // --------------------
  for (int d = 0; d < cfg_.depth; ++d) {
    const int B = (int)beams.size();
    if (B == 0) break;

    // For d>0, we need policy logits per beam state for top-M selection.
    if (d > 0) {
      ensure_size(S.eval_boards, (size_t)B);
      ensure_size(S.eval_metas,  (size_t)B);
      for (int j = 0; j < B; ++j) {
        S.eval_boards[(size_t)j] = beams[j].board;
        S.eval_metas [(size_t)j] = beams[j].meta;
      }
      const std::size_t BSZ = (std::size_t)B * kActions;
      ensure_size(logits_buf, BSZ);

      eval_policy_chunked(cb,
        std::span<const u64>(S.eval_boards.data(), (size_t)B),
        std::span<const u64>(S.eval_metas.data(),  (size_t)B),
        cfg_.max_eval_P,
        std::span<float>(logits_buf.data(), BSZ)
      );
    }

    expandables.clear();
    expandables.reserve((size_t)B * (size_t)cfg_.per_parent_top_m);
    fin.clear();

    for (int j = 0; j < B; ++j) {
      const BeamState& b = beams[j];
      if (!any_legal(b.m0, b.m1, b.m2)) continue;

      const int env = b.env;
      const float parentQ = b.q;
      const int parent_root = b.root_action;

      // score row for selecting top-M actions at this node
      const float* score_row =
        (d == 0)
          ? (S.root_logits.data() + std::size_t(env) * kActions)
          : (logits_buf.data()    + std::size_t(j)   * kActions);

      // root-only gumbel temperature
      int sel_buf[MAX_M];
      int nsel;
      if (d == 0 && use_noise && cfg_.root_eps > 0.0f && is_full_hand(b.meta)) {
        nsel = select_top_m_from_masks_gumbel<MAX_M>(
            b.m0, b.m1, b.m2,
            b.meta,
            std::span<const float>(score_row, kActions),
            cfg_.per_parent_top_m,
            cfg_.root_eps,
            rng_state,
            std::span<int>(sel_buf, MAX_M)
        );
      } else {
        nsel = select_top_m_from_masks_scores<MAX_M>(
            b.m0, b.m1, b.m2,
            b.meta,
            std::span<const float>(score_row, kActions),
            cfg_.per_parent_top_m,
            std::span<int>(sel_buf, MAX_M)
        );
      }

      if (nsel <= 0) continue;

      for (int t = 0; t < nsel; ++t) {
        const int a = sel_buf[t];

        const ProjectOut s = project(b.board, b.meta, a, T);
        const float q_prefix = parentQ + float(s.reward);

        const bool expandable = any_legal(s.m0, s.m1, s.m2);

        // root_action carried through the whole rollout:
        // - from root, root_action is the chosen root move a
        // - deeper, inherit parent_root
        const int root_action = (parent_root < 0) ? a : parent_root;

        if (s.done) {
          fin.push_back(FinalizeItem{env, root_action, s.board, s.meta, q_prefix, true});
          continue;
        }
        if (!expandable) {
          // frontier leaf (need-deal etc): bootstrap
          fin.push_back(FinalizeItem{env, root_action, s.board, s.meta, q_prefix, false});
          continue;
        }
        // candidate beam; pure top-k everywhere => sc=q_prefix
        expandables.push_back(BeamState{
          env, s.board, s.meta, s.m0, s.m1, s.m2,
          q_prefix, root_action
        });
      }
    }

    // dedup + keep top-K per env; pruned are ignored
    next_beams.clear();
    if (!expandables.empty()) {
      dedup_topk_per_env(expandables, cfg_.beam_width, N, next_beams);
    }

    // flush once per depth: frontier bootstraps + terminal leaves
    finalize_items(std::span<const FinalizeItem>(fin.data(), fin.size()));
    S.beams.swap(S.next_beams);
  }

  // --------------------
  // horizon bootstrap
  // --------------------
  if (!beams.empty()) {
    fin.clear();
    fin.reserve(beams.size());
    for (const auto& b : beams) {
      const int ra = b.root_action;
      if (ra < 0) continue;
      fin.push_back(FinalizeItem{b.env, ra, b.board, b.meta, b.q, false});
    }
    finalize_items(std::span<const FinalizeItem>(fin.data(), fin.size()));
  }
  // teacher mixing -> pi_behavior
  out_pi_behavior.resize(SZ);
  for (int i = 0; i < N; ++i) {
    masked_softmax_teacher(
      Lb.best.data()         + (std::size_t)i * kActions,
      root_m0[i], root_m1[i], root_m2[i], 1.0 / cfg_.teacher_tau,
      out_pi_behavior.data() + (std::size_t)i * kActions);
  }
}

} // namespace bb
