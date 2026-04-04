#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <numeric>
#include <span>

#include "tables.h"
#include "engine.h"
#include "env.h"
#include "search.h"
#include "dfs.h"

namespace py = pybind11;

namespace {
  thread_local std::uint64_t rng_seed = 0x9e3779b97f4a7c15ULL;
}

template <class T>
static py::array np_view_1d(const T* data, std::size_t n) {
  // zero-copy view; lifetime is tied to Tables singleton (static)
  return py::array(py::dtype::of<T>(),
                   {py::ssize_t(n)},
                   {py::ssize_t(sizeof(T))},
                   (const void*)data);
}

template <class T>
static py::array np_view_2d(const T* data, std::size_t r, std::size_t c) {
  return py::array(py::dtype::of<T>(),
                   {py::ssize_t(r), py::ssize_t(c)},
                   {py::ssize_t(sizeof(T) * c), py::ssize_t(sizeof(T))},
                   (const void*)data);
}

static constexpr int kActions = 192;

static inline void set_bits_slot(uint8_t* row, std::uint64_t mask, int off) noexcept {
  while (mask) {
    const int bit = std::countr_zero(mask);
    mask &= (mask - 1);
    row[off + bit] = 1;
  }
}

// ---------------------------
// small helpers
// ---------------------------
static inline torch::Tensor to_cpu_contig(torch::Tensor t, c10::ScalarType dt) {
  if (t.device().is_cuda()) t = t.to(torch::kCPU);
  if (t.scalar_type() != dt) t = t.to(dt);
  return t.contiguous();
}

// zero-copy tensor view over std::vector<T>, lifetime tied to shared_ptr owner
template <class T>
static torch::Tensor vec_view_1d(std::shared_ptr<void> owner,
                                 const std::vector<T>& v,
                                 c10::ScalarType dtype) {
  auto opt = torch::TensorOptions().dtype(dtype).device(torch::kCPU);

  // capture owner to keep it alive as long as tensor lives
  auto deleter = [owner](void*) mutable { owner.reset(); };

  return torch::from_blob((void*)v.data(),
                          {(long)v.size()},
                          deleter,
                          opt);
}

static inline void compute_masks_one(bb::u64 board, bb::u64 meta,
                                     const bb::Tables& T,
                                     bb::u64& m0, bb::u64& m1, bb::u64& m2) noexcept {
  const bb::u64 nb = static_cast<bb::u64>(bb::Tables::NULL_BLOCK);
  const bb::u64 E  = ~board;

  const bb::u64 b0 = bb::get_b0(meta);
  const bb::u64 b1 = bb::get_b1(meta);
  const bb::u64 b2 = bb::get_b2(meta);

  m0 = (b0 != nb) ? bb::legal_mask_for_block_E(E, (int)b0, T) : 0ULL;
  m1 = (b1 != nb) ? bb::legal_mask_for_block_E(E, (int)b1, T) : 0ULL;
  m2 = (b2 != nb) ? bb::legal_mask_for_block_E(E, (int)b2, T) : 0ULL;
}

// Convenience: (N,) u64 boards/metas -> (N,192) bool
static torch::Tensor legal_mask_batch(torch::Tensor boards_u64, torch::Tensor metas_u64) {
  boards_u64 = to_cpu_contig(std::move(boards_u64), torch::kUInt64);
  metas_u64  = to_cpu_contig(std::move(metas_u64),  torch::kUInt64);

  TORCH_CHECK(boards_u64.dim() == 1, "boards must be shape (N,)");
  TORCH_CHECK(metas_u64.dim()  == 1, "metas must be shape (N,)");
  TORCH_CHECK(boards_u64.size(0) == metas_u64.size(0), "boards/metas size mismatch");

  const auto N = (int64_t)boards_u64.size(0);
  auto* bptr = (bb::u64*)boards_u64.data_ptr<std::uint64_t>();
  auto* mptr = (bb::u64*)metas_u64.data_ptr<std::uint64_t>();

  auto opt_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor out = torch::zeros({N, kActions}, opt_b);
  bool* optr = out.data_ptr<bool>();

  const auto& T = bb::Tables::get();

  for (int64_t i = 0; i < N; ++i) {
    bb::u64 m0, m1, m2;
    compute_masks_one(bptr[i], mptr[i], T, m0, m1, m2);

    // slot 0
    while (m0) {
      int bit = std::countr_zero(m0);
      m0 &= (m0 - 1);
      optr[i * kActions + bit] = true;
    }
    // slot 1
    while (m1) {
      int bit = std::countr_zero(m1);
      m1 &= (m1 - 1);
      optr[i * kActions + 64 + bit] = true;
    }
    // slot 2
    while (m2) {
      int bit = std::countr_zero(m2);
      m2 &= (m2 - 1);
      optr[i * kActions + 128 + bit] = true;
    }
  }

  return out;
}

static torch::Tensor done_batch(torch::Tensor boards_u64, torch::Tensor metas_u64) {
  boards_u64 = to_cpu_contig(std::move(boards_u64), torch::kUInt64);
  metas_u64  = to_cpu_contig(std::move(metas_u64),  torch::kUInt64);

  TORCH_CHECK(boards_u64.dim() == 1, "boards must be shape (N,)");
  TORCH_CHECK(metas_u64.dim()  == 1, "metas must be shape (N,)");
  TORCH_CHECK(boards_u64.size(0) == metas_u64.size(0), "boards/metas size mismatch");

  const auto N = (int64_t)boards_u64.size(0);
  auto* bptr = (bb::u64*)boards_u64.data_ptr<std::uint64_t>();
  auto* mptr = (bb::u64*)metas_u64.data_ptr<std::uint64_t>();

  auto opt_b = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
  torch::Tensor out = torch::zeros({N}, opt_b);
  bool* optr = out.data_ptr<bool>();

  const auto& T = bb::Tables::get();

  for (int64_t i = 0; i < N; ++i) {
    bb::u64 m0, m1, m2;
    compute_masks_one(bptr[i], mptr[i], T, m0, m1, m2);
    optr[i] = !(m0 | m1 | m2);
  }

  return out;
}

static torch::Tensor sample_hands_crn(torch::Tensor boards_u64,
                                  torch::Tensor metas_u64,
                                  int64_t K) {
  boards_u64 = to_cpu_contig(std::move(boards_u64), torch::kUInt64);
  metas_u64  = to_cpu_contig(std::move(metas_u64),  torch::kUInt64);

  TORCH_CHECK(boards_u64.dtype() == torch::kUInt64, "boards_u64 must be uint64");
  TORCH_CHECK(metas_u64.dtype()  == torch::kUInt64, "metas_u64 must be uint64");
  TORCH_CHECK(boards_u64.size(0) == metas_u64.size(0), "boards/metas size mismatch");

  const auto& T = bb::Tables::get();
  const int64_t N = boards_u64.size(0);

  auto out = torch::empty({N, K, 3},
                          torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  auto* bptr = boards_u64.data_ptr<uint64_t>();
  auto* mptr = metas_u64.data_ptr<uint64_t>();
  auto* optr = out.data_ptr<int64_t>();

  py::gil_scoped_release release;

  for (int64_t n = 0; n < N; ++n) {
    // deterministic per-afterstate seed
    uint64_t h = bb::hash_state(bptr[n], mptr[n]);
    bb::SplitMix64 local_rng(h ^ rng_seed);

    // fill K hands
    // layout: out[n, k, j] contiguous in last dim
    int64_t base = n * (K * 3);
    for (int64_t k = 0; k < K; ++k) {
      optr[base + 3*k + 0] = (int64_t)bb::rand_block_pose(local_rng, T);
      optr[base + 3*k + 1] = (int64_t)bb::rand_block_pose(local_rng, T);
      optr[base + 3*k + 2] = (int64_t)bb::rand_block_pose(local_rng, T);
    }
  }

  return out;
}

static torch::Tensor can_complete_hand_batch(torch::Tensor boards_u64, torch::Tensor hands) {
  boards_u64 = to_cpu_contig(std::move(boards_u64), torch::kUInt64);
  hands  = to_cpu_contig(std::move(hands),  torch::kUInt8);
  // boards_u64: (N,) u64
  // hands: (N, 3) u8

  TORCH_CHECK(boards_u64.size(0) == hands.size(0), "boards/hands size mismatch");

  const auto& T = bb::Tables::get();
  const int64_t N = boards_u64.size(0);

  auto out = torch::empty({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

  uint64_t* bptr = boards_u64.data_ptr<uint64_t>();
  uint8_t* hptr = hands.data_ptr<uint8_t>();
  bool* optr = out.data_ptr<bool>();

  py::gil_scoped_release release;

  for (int64_t n = 0; n < N; ++n) {
      uint64_t board = bptr[n];
      uint8_t b0 = hptr[n * 3 + 0];
      uint8_t b1 = hptr[n * 3 + 1];
      uint8_t b2 = hptr[n * 3 + 2];
      optr[n] = bb::can_finish3(board, b0, b1, b2, T);
  }

  return out;
}

// ---------------------------
// Py wrappers for envs
// ---------------------------
struct PyEnv {
  std::shared_ptr<bb::Env> env;
  explicit PyEnv(std::uint64_t seed) : env(std::make_shared<bb::Env>((bb::u64)seed)) {}
  explicit PyEnv(std::uint64_t board, std::uint64_t seed) : env(std::make_shared<bb::Env>((bb::u64)board, (bb::u64)seed)) {}

  void reset() { env->reset(); }
  void rand_blocks() { env->rand_blocks(); }
  void set_blocks(int b0, int b1, int b2) { env->set_blocks(b0, b1, b2); }
  int step(int action) { return env->step(action); }

  py::list legal_actions() {
    std::vector<int> out;
    env->legal_actions(out);
    py::list L;
    for (int a : out) L.append(a);
    return L;
  }

  std::uint64_t board() const { return (std::uint64_t)env->board(); }
  std::uint64_t meta()  const { return (std::uint64_t)env->meta(); }
  std::uint64_t m0()    const { return (std::uint64_t)env->m0(); }
  std::uint64_t m1()    const { return (std::uint64_t)env->m1(); }
  std::uint64_t m2()    const { return (std::uint64_t)env->m2(); }

  bool need_blocks() const { 
    const std::uint64_t meta = (std::uint64_t)env->meta();
    const bb::u64 nb = static_cast<bb::u64>(bb::Tables::NULL_BLOCK);
    return (env->done() == 0) && 
    (bb::get_b0(meta) == nb) && 
    (bb::get_b1(meta) == nb) && 
    (bb::get_b2(meta) == nb); 
  }
  bool done() const { return env->done() != 0; }
  int score() const { return (int)env->score(); }
};

struct PyBatchEnv {
  std::shared_ptr<bb::BatchEnv> env;
  explicit PyBatchEnv(int n_envs, std::uint64_t seed)
    : env(std::make_shared<bb::BatchEnv>(n_envs, (bb::u64)seed)) {}

  int size() const { return env->size(); }
  void reset_all() { env->reset_all(); }

  void reset_indices(torch::Tensor idx) {
    idx = to_cpu_contig(std::move(idx), torch::kInt64);
    TORCH_CHECK(idx.dim() == 1, "idx must be shape (m,)");
    const int m = (int)idx.size(0);
    std::vector<int> tmp(m);
    auto* p = idx.data_ptr<std::int64_t>();
    for (int i = 0; i < m; ++i) tmp[i] = (int)p[i];
    env->reset_indices(tmp.data(), m);
  }

  void rand_blocks_indices(torch::Tensor idx) {
    idx = to_cpu_contig(std::move(idx), torch::kInt64);
    TORCH_CHECK(idx.dim() == 1, "idx must be shape (m,)");
    const int m = (int)idx.size(0);
    std::vector<int> tmp(m);
    auto* p = idx.data_ptr<std::int64_t>();
    for (int i = 0; i < m; ++i) tmp[i] = (int)p[i];
    env->rand_blocks_indices(tmp.data(), m);
  }

  torch::Tensor need_blocks() const {
    const int n = env->size();

    auto opt = torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU);
    torch::Tensor out = torch::empty({n}, opt);

    bool* ptr = out.data_ptr<bool>();

    const auto& metas = env->metas();
    const auto& done = env->done();

    const bb::u64 nb = static_cast<bb::u64>(bb::Tables::NULL_BLOCK);

    for (int i = 0; i < n; ++i) {
      const bb::u64 meta = metas[i];
      ptr[i] = (
        (!done[i]) && 
        (bb::get_b0(meta) == nb) &&
        (bb::get_b1(meta) == nb) &&
        (bb::get_b2(meta) == nb));
    }
    return out;
  }

  torch::Tensor step_indices(torch::Tensor idx, torch::Tensor act) {
    idx = to_cpu_contig(std::move(idx), torch::kInt64);
    act = to_cpu_contig(std::move(act), torch::kInt64);
    TORCH_CHECK(idx.dim() == 1 && act.dim() == 1, "idx and act must be 1D");
    TORCH_CHECK(idx.size(0) == act.size(0), "idx/act length mismatch");
    const int m = (int)idx.size(0);

    std::vector<int> itmp(m), atmp(m), rtmp(m);
    auto* ip = idx.data_ptr<std::int64_t>();
    auto* ap = act.data_ptr<std::int64_t>();
    for (int k = 0; k < m; ++k) { itmp[k] = (int)ip[k]; atmp[k] = (int)ap[k]; }

    env->step_indices(itmp.data(), atmp.data(), m, rtmp.data());

    auto opt = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor out = torch::empty({m}, opt);
    std::memcpy(out.data_ptr<int32_t>(), rtmp.data(), sizeof(int32_t) * (std::size_t)m);
    return out.to(torch::kFloat32);
  }

  // zero-copy views
  torch::Tensor boards() const { return vec_view_1d<std::uint64_t>(env, env->boards(), torch::kUInt64); }
  torch::Tensor metas()  const { return vec_view_1d<std::uint64_t>(env, env->metas(),  torch::kUInt64); }
  torch::Tensor m0()     const { return vec_view_1d<std::uint64_t>(env, env->m0(),     torch::kUInt64); }
  torch::Tensor m1()     const { return vec_view_1d<std::uint64_t>(env, env->m1(),     torch::kUInt64); }
  torch::Tensor m2()     const { return vec_view_1d<std::uint64_t>(env, env->m2(),     torch::kUInt64); }
  torch::Tensor done()   const { return vec_view_1d<std::uint8_t>(env, env->done(),    torch::kBool);   }
  torch::Tensor score()  const { return vec_view_1d<std::int32_t>(env, env->score(),  torch::kInt32);  }
};

// ---------------------------
// Py wrapper for search
// ---------------------------
class PyBeamSearch {
public:
  explicit PyBeamSearch(bb::SearchConfig cfg = {}, std::uint64_t base_seed = 0)
  : search_(cfg), base_seed_(base_seed), call_counter_(0) {}

  // --- config access (optional) ---
  bb::SearchConfig& cfg() { return search_.cfg(); }

  // --- RNG seed control (separate from cfg) ---
  void set_base_seed(std::uint64_t s) { base_seed_ = s; call_counter_ = 0; }
  std::uint64_t base_seed() const { return base_seed_; }

  void reset_rng_counter() { call_counter_ = 0; }
  std::uint64_t rng_counter() const { return call_counter_; }

  torch::Tensor search_batch(torch::Tensor boards_u64,
                            torch::Tensor metas_u64,
                            py::function policy_logits_fn,
                            py::function value_fn,
                            bool use_noise) {
    boards_u64 = to_cpu_contig(std::move(boards_u64), torch::kUInt64);
    metas_u64  = to_cpu_contig(std::move(metas_u64),  torch::kUInt64);

    TORCH_CHECK(boards_u64.dim() == 1, "boards must be shape (N,)");
    TORCH_CHECK(metas_u64.dim()  == 1, "metas must be shape (N,)");
    TORCH_CHECK(boards_u64.size(0) == metas_u64.size(0), "boards/metas size mismatch");

    const int N = (int)boards_u64.size(0);
    auto* bptr = (bb::u64*)boards_u64.data_ptr<std::uint64_t>();
    auto* mptr = (bb::u64*)metas_u64.data_ptr<std::uint64_t>();

    bb::EvalCallbacks cb;

    cb.policy_logits = [policy_logits_fn](std::span<const bb::u64> b,
                                          std::span<const bb::u64> m,
                                          std::span<float> out_logits) {
      py::gil_scoped_acquire gil;

      auto opt_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);
      torch::Tensor tb = torch::from_blob((void*)b.data(), {(long)b.size()}, opt_u64);
      torch::Tensor tm = torch::from_blob((void*)m.data(), {(long)m.size()}, opt_u64);

      torch::Tensor res = policy_logits_fn(tb, tm).cast<torch::Tensor>();
      res = to_cpu_contig(std::move(res), torch::kFloat32);

      TORCH_CHECK(res.dim() == 2 && res.size(0) == (long)b.size() && res.size(1) == kActions,
                  "policy_logits must return float32 tensor of shape (N, 192)");

      std::memcpy(out_logits.data(), res.data_ptr<float>(),
                  sizeof(float) * (std::size_t)out_logits.size());
    };

    cb.value = [value_fn](std::span<const bb::StateKey> keys,
                          std::span<float> out_values) {
      py::gil_scoped_acquire gil;

      const long n = (long)keys.size();
      auto opt_u64 = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCPU);

      // View StateKey as (n,2) [board, meta] uint64
      torch::Tensor t2 = torch::from_blob((void*)keys.data(), {n, 2}, opt_u64);
      torch::Tensor kb = t2.select(1, 0);
      torch::Tensor km = t2.select(1, 1);

      torch::Tensor res = value_fn(kb, km).cast<torch::Tensor>();
      res = to_cpu_contig(std::move(res), torch::kFloat32).view({n});

      TORCH_CHECK(res.numel() == n, "value must return float32 tensor of shape (N,)");

      std::memcpy(out_values.data(), res.data_ptr<float>(),
                  sizeof(float) * (std::size_t)out_values.size());
    };

    std::vector<float> pi_behavior;

    // deterministic per-call seed derived from base_seed_
    const std::uint64_t seed = base_seed_ + call_counter_;
    ++call_counter_;

    {
      py::gil_scoped_release release;
      search_.search_batch(std::span<const bb::u64>(bptr, (std::size_t)N),
                           std::span<const bb::u64>(mptr, (std::size_t)N),
                           cb, use_noise, seed,
                           pi_behavior);
    }

    auto opt_f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    torch::Tensor out_pi = torch::empty({N, kActions}, opt_f);
    std::memcpy(out_pi.data_ptr<float>(), pi_behavior.data(),
                sizeof(float) * pi_behavior.size());
    return out_pi;
  }

private:
  bb::BeamSearch search_;
  std::uint64_t base_seed_;
  std::uint64_t call_counter_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // constants
  m.attr("ACTIONS") = py::int_(kActions);
  m.attr("NULL_BLOCK") = py::int_(bb::Tables::NULL_BLOCK);
  m.attr("N_CANON") = py::int_(bb::Tables::N_CANON);
  m.attr("N_BLOCKS") = py::int_(bb::Tables::N_BLOCKS);
  m.attr("COMBO_MASK") = py::int_(bb::COMBO_MASK);
  m.attr("CTR_MASK") = py::int_(bb::CTR_MASK);
  m.attr("BLOCK_MASK") = py::int_(bb::BLOCK_MASK);
  m.attr("COMBO_SHIFT") = py::int_(bb::COMBO_SHIFT);
  m.attr("CTR_SHIFT") = py::int_(bb::CTR_SHIFT);
  m.attr("B0_SHIFT") = py::int_(bb::B0_SHIFT);
  m.attr("B1_SHIFT") = py::int_(bb::B1_SHIFT);
  m.attr("B2_SHIFT") = py::int_(bb::B2_SHIFT);
  m.attr("ZERO_META") = py::int_(bb::ZERO_META);

  // tables
  const auto& T = bb::Tables::get();

  m.attr("pose_off") = np_view_1d(T.pose_off.data(), T.pose_off.size()); // (N_CANON+1,)
  m.attr("pid2b") = np_view_1d(T.pid2b.data(), T.pid2b.size()); // (N_BLOCKS+1,)
  m.attr("pid2g") = np_view_1d(T.pid2g.data(), T.pid2g.size()); // (N_BLOCKS+1,)
  m.attr("block_pop") = np_view_1d(T.block_pop.data(), T.block_pop.size()); // (N_BLOCKS,)
  m.attr("cand_mask") = np_view_1d(T.cand_mask.data(), T.cand_mask.size()); // (N_BLOCKS,)
  m.attr("blocks") = np_view_1d(T.blocks.data(), T.blocks.size()); // (N_BLOCKS,)
  m.attr("ref_pos") = np_view_1d(T.ref_pos.data(), T.ref_pos.size()); // (N_BLOCKS,)
  m.attr("g2pose") = np_view_2d(T.g2pose.data(), T.pose_off.size(), 8); // (N_CANON+1, 8)
  m.attr("deltas") = np_view_2d(T.deltas.data(), T.blocks.size(), 9); // (N_BLOCKS, 9)
  m.attr("touch_lines") = np_view_2d(T.touch_lines.data(), T.blocks.size(), 64); // (N_BLOCKS, 64)

  // engine primitives
  m.def("legal_mask_for_block",
    [](std::uint64_t board, int bid) {
      const auto& T = bb::Tables::get();
      return (std::uint64_t)bb::legal_mask_for_block_E(~(bb::u64)board, bid, T);
    },
    py::arg("board"), py::arg("bid"));

  m.def("project",
    [](std::uint64_t board, std::uint64_t meta, int action) {
      const auto& T = bb::Tables::get();
      bb::ProjectOut out = bb::project((bb::u64)board, (bb::u64)meta, action, T);
      return py::make_tuple(
        (std::uint64_t)out.board,
        (std::uint64_t)out.meta,
        out.reward,
        out.done,
        (std::uint64_t)out.m0,
        (std::uint64_t)out.m1,
        (std::uint64_t)out.m2
      );
    },
    py::arg("board"), py::arg("meta"), py::arg("action"));
    
    m.def("legal_mask_batch", &legal_mask_batch, py::arg("boards"), py::arg("metas"));
    m.def("done_batch", &done_batch, py::arg("boards"), py::arg("metas"));
    m.def("can_complete_hand_batch", &can_complete_hand_batch, py::arg("boards"), py::arg("hands"));
    m.def("sample_hands_crn", &sample_hands_crn, py::arg("boards"), py::arg("metas"), py::arg("k"));

    m.def("seed_rng",
      [](std::uint64_t seed) {
        rng_seed = bb::splitmix64(seed);
      },
      py::arg("seed")
    );

  // single env
  py::class_<PyEnv>(m, "Env")
    .def(py::init<std::uint64_t>(), py::arg("seed")=0)
    .def(py::init<std::uint64_t, std::uint64_t>(), py::arg("board")=0, py::arg("seed")=0)
    .def("reset", &PyEnv::reset)
    .def("rand_blocks", &PyEnv::rand_blocks)
    .def("set_blocks", &PyEnv::set_blocks, py::arg("b0"), py::arg("b1"), py::arg("b2"))
    .def("step", &PyEnv::step, py::arg("action"))
    .def("legal_actions", &PyEnv::legal_actions)
    .def_property_readonly("board", &PyEnv::board)
    .def_property_readonly("meta",  &PyEnv::meta)
    .def_property_readonly("m0",    &PyEnv::m0)
    .def_property_readonly("m1",    &PyEnv::m1)
    .def_property_readonly("m2",    &PyEnv::m2)
    .def_property_readonly("done",  &PyEnv::done)
    .def_property_readonly("score", &PyEnv::score)
    .def_property_readonly("need_blocks", &PyEnv::need_blocks);

  // batch env
  py::class_<PyBatchEnv>(m, "BatchEnv")
    .def(py::init<int, std::uint64_t>(), py::arg("n_envs"), py::arg("seed")=0)
    .def("size", &PyBatchEnv::size)
    .def("reset_all", &PyBatchEnv::reset_all)
    .def("reset_indices", &PyBatchEnv::reset_indices, py::arg("idx"))
    .def("rand_blocks_indices", &PyBatchEnv::rand_blocks_indices, py::arg("idx"))
    .def("need_blocks", &PyBatchEnv::need_blocks)
    .def("step_indices", &PyBatchEnv::step_indices, py::arg("idx"), py::arg("act"))
    .def("boards", &PyBatchEnv::boards)
    .def("metas",  &PyBatchEnv::metas)
    .def("m0",     &PyBatchEnv::m0)
    .def("m1",     &PyBatchEnv::m1)
    .def("m2",     &PyBatchEnv::m2)
    .def("done",   &PyBatchEnv::done)
    .def("score",  &PyBatchEnv::score);

  // search config
  py::class_<bb::SearchConfig>(m, "SearchConfig")
    .def(py::init<>())
    .def_readwrite("depth", &bb::SearchConfig::depth)
    .def_readwrite("beam_width", &bb::SearchConfig::beam_width)
    .def_readwrite("per_parent_top_m", &bb::SearchConfig::per_parent_top_m)
    .def_readwrite("gamma", &bb::SearchConfig::gamma)
    .def_readwrite("teacher_tau", &bb::SearchConfig::teacher_tau)
    .def_readwrite("root_eps", &bb::SearchConfig::root_eps)
    .def_readwrite("max_eval_P", &bb::SearchConfig::max_eval_P)
    .def_readwrite("max_eval_V", &bb::SearchConfig::max_eval_V);

  // beam search
  py::class_<PyBeamSearch>(m, "BeamSearch")
    .def(py::init<bb::SearchConfig, std::uint64_t>(),
        py::arg("cfg") = bb::SearchConfig{},
        py::arg("base_seed") = 0)

    // core API
    .def("search_batch", &PyBeamSearch::search_batch,
        py::arg("boards_u64"),
        py::arg("metas_u64"),
        py::arg("policy_logits_fn"),
        py::arg("value_fn"),
        py::arg("use_noise") = true)

    // seed controls
    .def_property("base_seed", &PyBeamSearch::base_seed, &PyBeamSearch::set_base_seed)
    .def("reset_rng_counter", &PyBeamSearch::reset_rng_counter)
    .def_property_readonly("rng_counter", &PyBeamSearch::rng_counter)
    .def("cfg", &PyBeamSearch::cfg, py::return_value_policy::reference_internal);
}
