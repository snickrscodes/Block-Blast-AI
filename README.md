# Block Blast AI

A reinforcement-learning and policy-guided search system for **Block Blast**, built with **PyTorch, C++, and pybind11**.

The final agent does not attempt to learn the game end-to-end. Instead, it learns a **policy prior** for proposing promising placements and a **value function** for evaluating unresolved frontier states, then performs explicit short-horizon planning with a high-throughput native beam-search engine.

> **Learn what is expensive to calculate; search over what is cheap to enumerate.**

The released system combines:

* separate policy and value neural networks;
* an afterstate formulation for separating player decisions from future random hands;
* policy-guided depth-3 beam search;
* hard-max search backups and frontier value bootstrapping;
* explicit Monte Carlo evaluation of hand-boundary chance states;
* common-random-number sampling for reproducible value targets;
* a C++ bitboard simulator and search engine;
* a Pygame simulator renderer;
* and an experimental Windows real-game controller using screen segmentation and calibrated mouse input.

<!-- TODO: Add a short simulator/autoplay GIF or video thumbnail here. -->

---

## Contents

* [Results](#results)
* [Why Block Blast?](#why-block-blast)
* [Final Algorithm](#final-algorithm)

  * [Afterstates and stochasticity](#afterstates-and-stochasticity)
  * [Policy-guided beam search](#policy-guided-beam-search)
  * [Search backups and policy targets](#search-backups-and-policy-targets)
  * [Value learning](#value-learning)
* [Model Architecture](#model-architecture)

  * [Inputs](#model-inputs)
  * [Policy network](#policy-network)
  * [Value network](#value-network)
* [Environment and Benchmark Definition](#environment-and-benchmark-definition)

  * [State representation](#state-representation)
  * [Action encoding](#action-encoding)
  * [Block distribution](#block-distribution)
  * [Scoring](#scoring)
  * [Evaluation protocol](#evaluation-protocol)
  * [Simulator benchmark vs. the commercial game](#simulator-benchmark-vs-the-commercial-game)
* [Native C++ Engine](#native-c-engine)
* [Training and Reproducibility](#training-and-reproducibility)

  * [Three-stage training curriculum](#three-stage-training-curriculum)
  * [Important hyperparameters](#important-hyperparameters)
  * [Hardware and runtime](#hardware-and-runtime)
  * [Training commands](#training-commands)
  * [Seeds and common random numbers](#seeds-and-common-random-numbers)
  * [Checkpoints and logs](#checkpoints-and-logs)
  * [Distributed training](#distributed-training)
* [Running the Pretrained Agent](#running-the-pretrained-agent)

  * [Native extension](#native-extension)
  * [Pygame renderer](#pygame-renderer)
* [Real-Game Autoplay](#real-game-autoplay)

  * [Screen capture and segmentation](#screen-capture-and-segmentation)
  * [Block recognition](#block-recognition)
  * [Action-to-mouse translation](#action-to-mouse-translation)
  * [Emulator calibration](#emulator-calibration)
* [Experimental P4M / D4-Equivariant Model](#experimental-p4m--d4-equivariant-model)
* [Research History](#research-history)
* [Repository Structure](#repository-structure)
* [Limitations](#limitations)
* [References](#references)

---

# Results

The released checkpoint is:

```text
chkpts/chkpt3
```

It is the checkpoint produced after zero-indexed training phase 3 of the final fine-tuning run and evaluated in the log as **phase 4**.

Using the fixed simulator, fixed random-block distribution, deterministic evaluation seed, and the checkpoint's full search configuration:

| Metric              |          Result |
| ------------------- | --------------: |
| Mean score          | **1,631,955.7** |
| Median score        | **1,153,812.5** |
| Q25                 |       218,686.3 |
| Q75                 |     2,505,886.0 |
| Minimum             |              71 |
| Maximum             |  **10,657,435** |
| Mean episode length |     **1,395.6** |

These statistics are over **256 simulator games** with:

```text
evaluation seed      = 100003
evaluation envs      = 256
episodes / env       = 1
maximum placements   = 3000
search depth         = 3
beam width           = 1024
top actions / parent = 32
search noise         = disabled
root selection       = argmax
```

The subsequent phase-5 checkpoint regressed to a mean score of approximately 1.33M and a median of approximately 492K, so `chkpt3` was deliberately retained as the production checkpoint.

Earlier experimental systems also produced substantially larger raw scores, including a historical real-game run reaching roughly **141M**. Those results used different experimental configurations and are **not directly comparable** to the fixed released benchmark above.

The reproducible simulator benchmark is therefore the primary quantitative result for this repository.

---

# Why Block Blast?

Block Blast is a compact but surprisingly difficult stochastic planning problem.

The board contains only \(8\times8=64\) cells, yet each hand contains three randomly generated pieces and can expose up to:

$$
3\times64=192
$$

candidate action indices before legality masking.

A placement has several interacting consequences:

* immediate cells placed;
* rows or columns cleared;
* combo reward;
* remaining geometry;
* compatibility with the other pieces in the hand;
* survival probability under the next random hand;
* and the long-term ability to preserve usable board structure.

A move that scores immediately can leave a strategically poor board. Conversely, a move with little immediate reward may create a much stronger position for the remaining pieces.

This makes pure one-step policy prediction difficult: the network must learn both **spatial heuristics** and **short-horizon combinatorial planning**.

The main question that shaped the project became:

> Should the neural network learn to solve the entire decision problem implicitly, or should it learn the quantities that make explicit search effective?

The project ultimately converged on the second approach.

---

# Final Algorithm

The released agent is best described as a **learned heuristic search system**.

The policy network proposes useful branches. The C++ simulator resolves game mechanics exactly. Beam search explicitly compares short placement sequences. The value network estimates what lies beyond the search horizon and across future-hand chance boundaries.

## Afterstates and stochasticity

The stochasticity in Block Blast has a useful structure.

The player controls the placement of the current pieces, but does **not** control the next hand.

For current board \(s\), hand \(h\), and placement \(a\),

$$
s' = T(s,h,a)
$$

is deterministic.

The engine can calculate exactly:

* whether \(a\) is legal;
* which cells are occupied afterward;
* which lines clear;
* the immediate reward;
* and which pieces remain.

The randomness enters when a completed hand is replaced by another randomly generated hand.

This motivates an **afterstate representation**.

At the end of a hand, the value network can be trained to answer:

> How good is this board before I know what the next three pieces will be?

Formally, an afterstate value is approximately

$$
V_{\text{after}}(s')
\approx
\mathbb{E}_{h'\sim p(h)}
\left[V(s',h')\right].
$$

This separates the player's deterministic decision consequences from environmental randomness.

Rather than forcing the network or search tree to enumerate arbitrary sequences of future hands, the current hand is searched explicitly and the uncertainty beyond the hand boundary is summarized by the learned value function.

---

## Policy-guided beam search

The native search has a default depth of three placements, matching the maximum number of pieces in a hand.

For each search node:

1. the C++ engine computes the exact legality masks;
2. the policy network produces logits over the 192 fixed action indices;
3. only the top \(M\) legal actions are expanded;
4. the engine applies each placement exactly;
5. equivalent successor states are deduplicated;
6. the strongest \(K\) states are retained for the next search depth.

For the released checkpoint:

$$
D=3,\qquad M=32,\qquad K=1024.
$$

This changes what the policy network needs to learn.

A direct policy ideally needs:

$$
\arg\max_a P_\theta(a\mid s,h)=a^\*.
$$

A search prior only needs something closer to:

$$
a^\*
\in
\operatorname{TopM}
\left(P_\theta(\cdot\mid s,h)\right).
$$

If the best move is ranked fourth rather than first, explicit search can still recover it.

This proved to be a much easier and more useful learning objective.

### Beam retention

The beam itself is deliberately simple.

After the policy has restricted each parent's branching factor, surviving states are ranked using the **exact accumulated in-hand reward**. The value model is not repeatedly used to sort every intermediate beam node.

This keeps the inner search loop cheap.

The value model is instead used when search reaches a state whose continuation is unresolved.

---

## Search backups and policy targets

A branch is finalized when it:

* reaches a terminal state;
* reaches a non-terminal hand boundary;
* or survives to the depth horizon.

For terminal branches, the exact accumulated reward is used.

For a non-terminal frontier state:

$$
Q_{\text{leaf}}
=
R_{\text{prefix}}
+
\gamma V_\phi(s_{\text{leaf}}),
$$

with:

$$
\gamma=0.997.
$$

Search then performs a **hard-max backup** for each root action:

$$
L(a)
=
\max_{\text{searched continuations beginning with }a}
Q.
$$

Beam pruning has no special terminal meaning.

A candidate that falls outside the beam is simply ignored; it is not treated as solved, terminal, or assigned an artificial final value. Removing this earlier "finalize-on-prune" behavior was an important simplification in the search design.

The backed-up root values are transformed into a teacher distribution:

$$
\pi_{\text{search}}(a)
\propto
\exp
\left(
\frac{L(a)}{\tau}
\right),
$$

over legal root actions.

The released training configuration uses:

$$
\tau=2.0.
$$

The policy is then trained by masked cross-entropy against this search-generated distribution.

In effect:

> **search repeatedly generates a stronger policy target, and the policy learns to imitate the search.**

During training, actions are sampled from this teacher distribution. During evaluation, the agent takes its argmax.

### Root exploration

Training search also perturbs the root proposal set with Gumbel noise.

If \(z_a\) is a policy logit,

$$
z'_a=z_a+\epsilon G_a,
\qquad
G_a\sim\operatorname{Gumbel}(0,1),
$$

before root top-\(M\) selection.

The final training run used:

```text
P_root_eps = 0.5
```

Noise is applied only to root proposal selection during training. Evaluation search is deterministic.

---

## Value learning

The value network is trained using both ordinary trajectory states and explicit hand-boundary afterstates.

For regular trajectory states, returns are constructed using GAE with:

$$
\gamma=0.997,
\qquad
\lambda=0.99.
$$

The difficult case is a completed-hand afterstate.

The next three pieces have not been observed yet, so the code explicitly samples \(K\) possible future hands:

$$
h_1,\ldots,h_K\sim p(h).
$$

For each sampled hand, the engine first tests whether the hand can still be completed from the board.

Playable hands are evaluated by the value network; immediately losing deals contribute zero continuation value.

The afterstate target is then approximated by:

$$
\hat V_{\text{after}}(s)
=
\frac1K
\sum_{k=1}^K
V(s,h_k).
$$

The final fine-tuning stage uses:

$$
K=1024
$$

future hands per afterstate.

This is one of the main ways the system handles stochasticity: **search resolves the current hand explicitly while Monte Carlo value targets teach the network what to expect beyond the chance boundary.**

### Heavy-tailed returns

Block Blast scores are extremely heavy-tailed.

The value network therefore operates on transformed targets:

$$
\tilde V=\operatorname{asinh}(V).
$$

When search requires values in the original reward scale, the inverse transform is:

$$
V=\sinh(\tilde V).
$$

`asinh` behaves approximately linearly near zero while strongly compressing extreme returns, making value regression considerably easier numerically.

---

# Model Architecture

The released system uses **separate policy and value networks** rather than a single shared dual-head network.

They use closely related architectures but have independent parameters.

The released CNN checkpoint contains approximately:

```text
Policy network: 1,956,512 trainable parameters
Value network:  1,975,792 trainable parameters
```

## Model inputs

Each network receives three components:

### Board

The \(8\times8\) board is provided as a flattened binary vector:

$$
x_{\text{board}}\in\{0,1\}^{64}.
$$

Inside the network it is reshaped to:

$$
1\times8\times8.
$$

Three fixed CoordConv-style channels are concatenated:

* normalized \(x\);
* normalized \(y\);
* radius \(r=\sqrt{x^2+y^2}\).

The initial spatial input therefore has four channels:

```text
occupancy + x + y + radius
```

These coordinates provide a lightweight absolute spatial prior while leaving the CNN to learn the relevant geometric features.

### State information

Two scalar state variables are supplied:

```text
combo
counter
```

The combo is transformed with `log1p`, while the counter receives a learned embedding.

### Current hand

The hand is represented by three oriented block IDs:

$$
(b_0,b_1,b_2).
$$

There are 41 non-null oriented block poses plus a null sentinel used after a piece has been consumed.

Thus the conceptual input signature is:

$$
f(
\text{board}_{64},
\text{state}_{2},
\text{blocks}_{3}
).
$$

---

## Policy network

Each block receives a learned 128-dimensional embedding.

The three current blocks are then treated as a complete \(K_3\) graph.

A small message-passing module (`TriadMP`) repeatedly exchanges information between each block and the other two:

```text
block 0 ←→ block 1
   ↖        ↗
      block 2
```

This lets the representation of each individual piece depend on the rest of the current hand.

For example, a piece that is harmless by itself may become strategically important when the other two pieces compete for the same remaining space.

### Spatial backbone

The board passes through:

* an initial convolutional stem;
* residual convolutional blocks;
* GroupNorm;
* SiLU nonlinearities;
* FiLM-style conditioning from the hand/state context;
* spatial self-attention;
* repeated block-to-board cross-attention.

The cross-attention layers use block embeddings as queries and spatial board features as keys/values.

This allows the model to ask, in effect:

> Which regions of this board matter for this particular piece?

Block embeddings are then updated using the attended board information and another round of message passing.

This creates a repeated interaction between:

```text
board geometry
    ↕
individual blocks
    ↕
relationships among the three blocks
```

rather than treating the hand as a flat auxiliary vector.

### Policy output

The final spatial features are combined with:

* per-piece embeddings;
* state embeddings;
* each of the 64 board positions.

The policy outputs:

$$
3\times64=192
$$

placement logits.

Legality itself is **not learned**. Invalid actions are masked using the exact C++ environment.

---

## Value network

The value model uses the same broad design:

* CoordConv-style board input;
* residual convolutional backbone;
* hand embeddings;
* \(K_3\) message passing;
* state conditioning;
* spatial self-attention;
* block-to-board cross-attention.

Instead of producing 192 action logits, the final board-position features are pooled and passed through an MLP to produce a scalar transformed value:

$$
V_\phi(s,h)\in\mathbb R.
$$

The policy and value networks are architecturally related but separately parameterized.

---

# Environment and Benchmark Definition

A major goal of the release is to distinguish the **fixed research environment** from the current commercial Block Blast application.

The simulator below is the authoritative benchmark definition.

## State representation

The board occupancy itself fits in a single 64-bit integer:

$$
\texttt{board}\in\{0,\ldots,2^{64}-1\}.
$$

Each bit corresponds to one cell of the \(8\times8\) grid.

The native engine stores the remaining state information in another 64-bit metadata word containing:

* combo;
* combo counter;
* block slot 0;
* block slot 1;
* block slot 2.

A search state can therefore be represented using approximately two machine words:

```text
(board_uint64, meta_uint64)
```

This makes copying, hashing, comparing, and expanding large numbers of candidate states extremely inexpensive.

---

## Action encoding

The action space contains 192 fixed indices:

$$
a\in[0,192).
$$

The upper portion chooses one of three hand slots:

$$
\text{slot}=a\gg6,
$$

while the lower six bits specify an \(8\times8\) anchor position:

$$
\text{position}=a\ \&\ 63.
$$

Therefore:

```text
0–63      → hand slot 0
64–127    → hand slot 1
128–191   → hand slot 2
```

Each range contains 64 possible board anchors.

Many of these action indices are illegal for a particular board/piece combination. The native engine computes exact 64-bit legality masks and the policy/search code masks invalid actions.

---

## Block distribution

The simulator contains:

* **15 canonical piece types**;
* **41 distinct oriented poses** after deduplicating rotational/reflection symmetries.

Piece generation is **not uniform over the 41 pose IDs**.

For each generated piece:

1. one of the 15 canonical block types is selected uniformly;
2. one of that block's distinct \(D_4\) orientations is selected uniformly.

Thus:

$$
P(B=b)=\frac1{15},
$$

and for a canonical block \(b\) with \(n_b\) distinct poses:

$$
P(P=p\mid B=b)=\frac1{n_b}.
$$

Three independent draws form a hand.

The native implementation uses `SplitMix64` for its deterministic pseudorandom streams.

This distribution is part of the benchmark definition and should be preserved when reproducing the reported scores.

---

## Scoring

The simulator's scoring rule was reverse-engineered from the scoring behavior used during the original development period.

Let:

* \(n(b)\) be the number of occupied cells in the placed block;
* \(\ell\) be the number of rows/columns cleared simultaneously;
* \(c\) be the combo value before the placement.

The line-clear base bonus is:

$$
B(\ell)=
\begin{cases}
0,&\ell=0,\\
10,&\ell=1,\\
10\ell(\ell-1),&\ell\ge2.
\end{cases}
$$

A line clear first increments the combo:

$$
c'=c+1.
$$

The placement reward is then:

$$
r=
n(b)
+
c' B(\ell)
+
300\cdot
\mathbf 1[\text{board is empty after clearing}].
$$

If no line is cleared:

$$
r=n(b),
$$

and the combo may expire according to the simulator's combo counter.

After a successful clear, the combo counter is reset to:

$$
3+
(\text{number of pieces remaining in the current hand}).
$$

On each placement the counter decreases. If it expires on a non-clearing move, the combo is reset.

The C++ simulator is the authoritative implementation of these rules.

---

## Evaluation protocol

The published `chkpt3` benchmark uses the checkpoint's saved hyperparameters.

Evaluation creates:

```text
256 parallel environments
1 episode per environment
eval_seed = 100003
maximum placement loop = 3000
```

The search planner is separately initialized from:

```text
eval_seed + 1
```

and runs with exploration disabled.

The policy/value callbacks use the same neural networks and search settings stored in the checkpoint:

```text
depth                 = 3
beam width            = 1024
per-parent top-M      = 32
gamma                 = 0.997
teacher temperature   = 2.0
```

The selected root action is:

```python
pi.argmax(-1)
```

rather than sampled.

The 256 fixed-seed games produced:

```text
phase 4
mean episode length = 1395.6094
mean score          = 1631955.7188

min =       71
q25 =   218686.25
q50 =  1153812.50
q75 =  2505886.00
max = 10657435
```

The `max_placements_test=3000` safety cap should be kept in mind when interpreting exceptionally long games.

---

## Simulator benchmark vs. the commercial game

The simulator is deliberately treated as the reproducible benchmark.

The real commercial game is not.

During the original January/February 2026 experiments, Block Blast **v9.1.4 on Google Play Games** exhibited a scoring regime that matched the implemented simulator scoring rule very closely.

More recent testing has shown substantially different scoring behavior:

* different multipliers across nearby app versions;
* different behavior after restarting games;
* and apparent changes even during some individual games.

The underlying cause has not been established and should not be assumed to be purely version-dependent.

Consequently:

> **Current real-game scores are not treated as directly comparable to the fixed simulator benchmark.**

The distinction throughout this repository is:

| Reproducible benchmark         | Commercial-game integration      |
| ------------------------------ | -------------------------------- |
| Fixed C++ environment          | External Block Blast application |
| Fixed block distribution       | Game-controlled behavior         |
| Fixed scoring function         | Scoring can vary                 |
| Explicit random seeds          | External state/configuration     |
| Repeatable evaluation protocol | Emulator/UI dependent            |
| Primary quantitative result    | Integration demonstration        |

Historical real-game footage is useful evidence that the controller and agent genuinely operated the commercial game, but the simulator benchmark is the authoritative performance measurement.

---

# Native C++ Engine

The first search implementations spent a large fraction of their runtime manipulating states and expanding trees in Python.

The final system moves the combinatorial inner loop into a C++20 extension exposed through PyTorch/pybind11.

The native engine handles:

* compact bitboard states;
* block geometry;
* move legality;
* state transitions;
* simultaneous line clearing;
* scoring;
* random hand generation;
* batched environments;
* top-\(M\) action selection;
* beam management;
* duplicate-state elimination;
* search backups;
* common-random-number hand generation.

### Precomputed geometry

Every distinct piece pose is precomputed.

For each block, the engine stores information including:

* bit representation;
* reference position;
* occupied-cell count;
* legal empty-board anchor mask;
* touched rows/columns;
* transformed orientations.

A legality test can therefore be expressed mostly using shifts and bitwise operations instead of repeatedly reconstructing piece geometry.

### Hybrid Python/C++ execution

The architecture is roughly:

```text
Python / PyTorch
        │
        │ batched policy/value inference
        ▼
     callbacks
        │
        ▼
C++ beam-search engine
        │
        ├── legality
        ├── top-M selection
        ├── transitions
        ├── scoring
        ├── deduplication
        ├── beam pruning
        └── backups
        │
        ▼
candidate board/meta states
```

Neural inference remains in PyTorch while the much larger number of cheap combinatorial operations remains native.

One major search-optimization pass during development produced approximately a **25× reduction** in search/training time on the workload being profiled at the time, taking an earlier roughly 50-minute iteration down to around two minutes.

That improvement was one of the clearest lessons from the project: optimizing the search representation and inner loop often mattered more than making the neural network larger.

---

# Training and Reproducibility

The released CNN was trained in three stages.

Each new stage loaded the policy/value weights from a previous checkpoint and started a new optimizer/scheduler configuration with a more expensive search and value-target setup.

## Three-stage training curriculum

| Parameter                   |    Stage 1 — coarse | Stage 2 — fine-tune | Stage 3 — large-search fine-tune |
| --------------------------- | ------------------: | ------------------: | -------------------------------: |
| Starting model              |              random |   Stage-1 `chkpt35` |                 Stage-2 `chkpt2` |
| Worker environments         |                 256 |                 256 |                              256 |
| Evaluation environments     |                 256 |                 256 |                              256 |
| Horizon                     |                 128 |                 512 |                              512 |
| Rollouts / phase            |                  32 |                  16 |                               32 |
| Configured phases           |                  40 |                  15 |                                5 |
| Future hands / chance state |                  64 |                 256 |                             1024 |
| Per-parent top-\(M\)        |                  16 |                  24 |                               32 |
| Beam width                  |                 128 |                 256 |                             1024 |
| Max train batch             |                8192 |                8192 |                             8192 |
| Policy inference batch cap  |              40,000 |              35,000 |                           35,000 |
| Value inference batch cap   |             100,000 |             100,000 |                          100,000 |
| Policy LR                   | \(5e{-4}\to5e{-5}\) | \(1e{-4}\to1e{-5}\) |                       \(3e{-4}\) |
| Value LR                    | \(3e{-4}\to3e{-5}\) | \(5e{-5}\to5e{-6}\) |                     \(1.5e{-4}\) |
| Policy warmup               |                   0 |         128 updates |                      128 updates |
| Value warmup                |                   0 |         192 updates |                      192 updates |
| \(\gamma\)                  |                .997 |                .997 |                             .997 |
| GAE \(\lambda\)             |                 .99 |                 .99 |                              .99 |
| Search teacher \(\tau\)     |                 2.0 |                 2.0 |                              2.0 |
| Root Gumbel scale           |                  .5 |                  .5 |                               .5 |
| Approx. runtime / phase     |                ~1 h |              ~5–6 h |                      15.7–18.4 h |

The second fine-tuning run was configured for 15 phases but was **manually stopped early after approximately nine phases**. The exact earlier log was not retained.

The final stage did not use the last Stage-2 checkpoint. It was initialized specifically from:

```text
chkpts_finetune/chkpt2
```

Stage 3 then ran all five configured phases.

Its best balanced checkpoint was the model evaluated as phase 4:

```text
chkpts_finetune2/chkpt3
```

which is published in this repository simply as:

```text
chkpts/chkpt3
```

The curriculum intentionally moved compute from cheaper early learning toward much stronger planning and stochastic-value targets:

```text
future hands:    64  →  256  → 1024
beam width:     128  →  256  → 1024
top-M:           16  →   24  →   32
horizon:        128  →  512  →  512
```

---

## Important hyperparameters

The authoritative configurations are preserved in `main.py` and `config.py`.

The most important knobs are:

### `P_beam_width`

Maximum number of surviving search states retained per environment after an expansion depth.

Larger values preserve more candidate continuations but increase native search and neural inference cost.

### `P_per_parent_top_m`

Maximum number of policy-ranked legal actions expanded from each parent.

This controls the effective branching factor before global beam pruning.

### `n_hands_V`

Number of common-random-number future hands sampled when constructing a hand-boundary afterstate target.

This is one of the most expensive training parameters.

### `horizon_len`

Number of environment transitions collected in each rollout segment.

This is unrelated to the search depth of three; it controls training trajectory length.

### `n_rollouts_phase`

Number of rollout/update cycles performed before the phase-level value epoch and evaluation.

### `V_gamma`

Discount factor used for search frontier bootstrapping and value-target construction.

Released value:

```text
0.997
```

### `V_lmbda`

GAE parameter.

Released value:

```text
0.99
```

### `P_teacher_tau`

Temperature applied to hard-max root search returns when constructing the policy target.

Released value:

```text
2.0
```

### `P_root_eps`

Scale of root Gumbel exploration during rollout search.

Released value:

```text
0.5
```

### Batch-size caps

The code separately controls:

* neural training batch size;
* policy inference batch size;
* value inference batch size.

This lets the native search create very large candidate populations without requiring them to fit through the neural networks simultaneously.

---

## Hardware and runtime

The final large-search fine-tuning run was performed on a **rented single-GPU server** using:

```text
GPU:               NVIDIA GeForce RTX 5090
reported VRAM:     33.67 GB
compute capability: 12.0
PyTorch:           2.13.0+cu130
CUDA runtime:      13.0
cuDNN:             92000
```

The code enables TF32 for supported CUDA matrix operations.

The final run recorded the following approximate phase durations, including evaluation:

| Logged phase               | Approx. elapsed time |
| -------------------------- | -------------------: |
| Initial phase-0 evaluation |               54 min |
| Phase 1                    |               17.5 h |
| Phase 2                    |               15.8 h |
| Phase 3                    |               15.7 h |
| Phase 4                    |               18.4 h |
| Phase 5                    |               17.3 h |

The earlier stages were substantially cheaper:

```text
Stage 1: roughly 1 hour / phase
Stage 2: roughly 5–6 hours / phase
Stage 3: roughly 16–18 hours / phase
```

Only the Stage-3 full timing log was retained, so Stage-1/2 timings should be interpreted as approximate observed runtimes.

---

## Training commands

The original single-GPU server setup was launched from `/workspace/block_blast`.

PyTorch was installed with:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

`torchvision` happened to be installed in the training environment but is not required by the current project code.

The native extension was then compiled in-place:

```bash
cd /workspace/block_blast/bbengine
pip install -v -e . --no-build-isolation
```

Finally:

```bash
cd /workspace/block_blast
python3 main.py
```

`bbengine/setup.py` imports PyTorch's C++ extension infrastructure, so PyTorch should be installed before building the extension. `--no-build-isolation` allows the build to use the already-installed PyTorch environment.

The three historical training configurations remain in `main.py`. Only the desired stage should be active when reproducing a particular run.

---

## Seeds and common random numbers

The training harness maintains separate deterministic seed streams for:

```text
PyTorch
environment generation
search exploration
common-random-number hand sampling
```

For each phase/rank/stream, a deterministic phase seed is derived from:

```python
base_seed
phase
rank
stream_id
```

This prevents unrelated stochastic components from implicitly sharing one global RNG trajectory.

### Common random numbers

Hand-boundary value estimation uses **common random numbers (CRN)**.

For each afterstate, the native extension computes a hash of:

```text
(board, metadata)
```

and combines it with the current CRN seed.

A local `SplitMix64` stream then deterministically generates the \(K\) candidate future hands for that afterstate.

Conceptually:

$$
\text{seed}(s)
=
\operatorname{hash}(s)
\oplus
\text{CRN seed}.
$$

This means the same afterstate under the same training phase receives the same sampled future-hand set, reducing unnecessary Monte Carlo variation when states are revisited or compared.

The final training stage used:

$$
K=1024.
$$

The evaluation environment uses a separate fixed seed:

```text
eval_seed = 100003
```

Exact bitwise identity across arbitrary GPUs, PyTorch versions, compilers, and CUDA versions is not guaranteed, particularly because CUDA/TF32 is enabled. The seed design instead aims to make the stochastic structure of a given software/hardware configuration controlled and repeatable.

---

## Checkpoints and logs

A normal training run writes:

```text
logs/
└── train.log

chkpts/
├── chkpt0
├── chkpt1
├── chkpt2
└── ...
```

Every checkpoint stores:

```text
checkpoint version
completed phase
policy state_dict
value state_dict
policy optimizer state
value optimizer state
policy scheduler state
value scheduler state
full Config hyperparameters
```

Writes use a temporary file followed by `os.replace` to avoid leaving a partially written checkpoint.

### Checkpoint numbering

The initial untrained/loaded model is evaluated as:

```text
phase 0
```

After training zero-indexed phase `0`, the code evaluates:

```text
phase 1
```

and saves:

```text
chkpt0
```

Therefore:

```text
chkpt0 → completed_phase=1 → phase-1 evaluation
chkpt1 → completed_phase=2 → phase-2 evaluation
chkpt2 → completed_phase=3 → phase-3 evaluation
chkpt3 → completed_phase=4 → phase-4 evaluation
```

The released `chkpts/chkpt3` therefore corresponds to the reported phase-4 result.

### Starting a new fine-tuning stage

`start_path` loads only:

```text
policy weights
value weights
```

from another checkpoint.

It intentionally starts with:

```text
fresh optimizer state
fresh scheduler state
new Config
phase counter = 0
```

This is the mechanism used between the three training stages.

### Exact continuation

`resume_name` is different.

It restores:

```text
model weights
optimizer states
scheduler states
completed phase
saved hyperparameters
```

and verifies that the requested configuration exactly matches the checkpoint configuration.

This is intended for resuming an interrupted run rather than beginning a new fine-tuning schedule.

A fresh run truncates `train.log`; an exact resume appends to the existing log.

---

## Distributed training

The code also contains a single-machine multi-GPU path through `DistributedRunner`.

It uses:

* `torch.multiprocessing.spawn`;
* one process per GPU;
* `DistributedDataParallel`;
* the NCCL backend.

The released model was **not** trained with distributed training.

All reported production training was performed on a single RTX 5090.

---

# Running the Pretrained Agent

The default released checkpoint is located at:

```text
chkpts/chkpt3
```

Both `render.py` and `autoplay.py` resolve this path relative to the repository root by default.

A different checkpoint can be supplied explicitly.

## Native extension

The repository includes the native source as well as prebuilt CPython 3.12 x86-64 binaries for Windows and Linux.

The native code requires C++20.

For unsupported Python/platform/PyTorch combinations, rebuild from source:

```bash
cd bbengine
pip install -v -e . --no-build-isolation
```

Because `bbengine` is built with PyTorch's `CppExtension`, binary compatibility can depend on the local Python and PyTorch installation.

The source tree is therefore the authoritative portable implementation; the bundled `.pyd` and `.so` files are convenience builds for the tested release environment.

---

## Pygame renderer

`render.py` runs the trained agent entirely against the local simulator and renders it using Pygame.

Additional dependency:

```bash
pip install pygame
```

Run:

```bash
python render.py
```

The renderer defaults to:

```text
chkpts/chkpt3
```

and automatically uses CUDA when available, otherwise CPU.

A different device or checkpoint can be selected with:

```bash
python render.py --device cuda:0 --checkpoint path/to/checkpoint
```

The bundled font is expected at:

```text
assets/LeagueSpartan-Bold.otf
```

but the renderer falls back to a system font when it is unavailable.

<!-- TODO: Add a screenshot of render.py here. -->

---

# Real-Game Autoplay

`autoplay.py` is an experimental integration layer that controls the commercial Windows/emulator game.

It is separate from the reproducible simulator benchmark.

Current additional dependencies include:

```bash
pip install numpy mss pywin32
```

The controller was tested with:

* **Google Play Games Beta**
* **BlueStacks 5**

on Windows.

The implementation is intentionally simple: it does not use an Android API or game-specific internal hooks.

It observes the game visually and produces ordinary mouse drags.

The pipeline is:

```text
emulator window
      │
      ▼
MSS screen capture
      │
      ▼
grayscale conversion
      │
      ├── 8×8 board segmentation
      │
      └── three tray ROI segmentations
      │
      ▼
(board bitboard, block IDs)
      │
      ▼
native search + neural agent
      │
      ▼
action index [0, 192)
      │
      ▼
engine slot → physical tray slot
      │
      ▼
calibrated Win32 mouse drag
```

---

## Screen capture and segmentation

The game client area is located using `win32gui`.

Pixels are captured using MSS.

MSS returns BGRA pixels, which are converted to grayscale luminance using:

$$
Y=
0.2126R+
0.7152G+
0.0722B.
$$

The calibrated client-area crop is then applied using:

```python
AREA_TOP
AREA_LEFT
```

### Board recognition

The expected center of each of the 64 board cells is calculated from the current client width and a reference calibration width.

For every cell, a small image patch is sampled and its median luminance is compared to the calibrated empty-cell value:

```python
EMPTY_COL
```

The resulting occupancy matrix is converted directly to the same 64-bit bitboard representation used by the native agent.

The currently calibrated value is approximately:

```text
EMPTY_COL = 41.5442
```

An older observed calibration value:

```text
35.6148
```

is retained in the source as a reference because the game's visual colors have changed across versions.

These values should not be assumed portable across arbitrary game versions or display configurations.

---

## Block recognition

Each of the three physical tray positions has a calibrated region of interest.

For each ROI, the controller:

1. separates foreground pixels from the tray background;
2. finds the top-left extent of the rendered piece;
3. samples a \(5\times5\) mini-grid;
4. converts that grid to a bit representation;
5. normalizes the shape;
6. matches it against the engine's 41 known oriented block poses.

Animations can temporarily cover enough of the screen to produce invalid block patterns.

These failures are treated as recoverable events: the controller catches block-recognition failures, waits briefly, captures a fresh frame, and retries rather than immediately terminating the run.

### Physical vs. engine hand order

When the controller attaches in the middle of a partially consumed hand, the physical tray may look like:

```text
[NULL, NULL, block]
```

while the native environment compacts remaining pieces into canonical engine slots.

The controller therefore separately tracks:

```text
physical tray order
engine/canonical block order
engine slot → physical tray slot mapping
```

so an action chosen for canonical engine slot 0 can still pick up the correct visible piece from physical tray slot 2.

---

## Action-to-mouse translation

The agent returns the same fixed action encoding used by the simulator:

$$
a\in[0,192).
$$

From this the controller extracts:

```text
engine hand slot = action >> 6
anchor position  = action & 63
```

The engine slot is translated back into the corresponding physical tray slot.

Using:

* the block's reference position;
* its geometric center;
* the target board cell;
* calibrated pickup coordinates;
* and affine drag factors;

the desired board placement is translated into a physical mouse drag.

Windows `SendInput` / cursor APIs then perform a multi-step drag from the tray to the board.

This lets the real-game controller use exactly the same action space as simulator inference.

---

## Emulator calibration

Autoplay calibration depends on:

* emulator;
* window scaling;
* game version;
* side panels/advertisements;
* UI layout.

The values below are examples from the development machine rather than universal defaults.

| Environment                        | `AREA_TOP` | `AREA_LEFT` |
| ---------------------------------- | ---------: | ----------: |
| Google Play Games Beta             |        ~58 |           0 |
| BlueStacks 5 without side ad panel |        ~48 |           0 |
| BlueStacks 5 with side ad panel    |        ~48 |        ~348 |

The remaining board/tray/pickup constants in `autoplay.py` were measured against a reference window width:

```text
MEASUREMENT_WIDTH = 576
```

and scaled with the current client width.

Anyone using a different screen layout should expect to recalibrate these constants.

The real-game integration should therefore be viewed as a proof-of-functionality controller rather than a portable computer-vision package.

---

# Experimental P4M / D4-Equivariant Model

The repository also includes a custom symmetry-aware version of the CNN architecture in:

```text
model/p4mcnn.py
```

Block Blast has strong geometric \(D_4\) symmetry:

* four rotations;
* four reflected orientations.

The order of the three blocks in a hand is also arbitrary from a strategic perspective, giving an additional \(S_3\) permutation symmetry.

The experimental model was designed so that:

* the **policy is equivariant** under \(D_4\times S_3\);
* the **value is invariant** under \(D_4\times S_3\).

The implementation follows the broad group-convolution framework of Cohen & Welling's **Group Equivariant Convolutional Networks (2016)**, while the finite-group machinery used here was implemented specifically for this project.

### Architecture

The P4M model mirrors the conventional CNN design where practical:

* board convolutions;
* residual blocks;
* block embeddings;
* hand message passing;
* self-attention;
* block-to-board cross-attention;
* CoordConv-style positional information.

Regular spatial features additionally carry an explicit eight-element \(D_4\) group axis throughout the group-convolution backbone.

This greatly increases activation volume and computation compared with the ordinary CNN.

### Verification

Run:

```bash
python test_equivariance.py
```

The test enumerates all:

$$
|D_4|\times|S_3|
=
8\times6
=
48
$$

combined transformations.

Using double-precision networks, it checks:

$$
P(g\cdot x)
=
\rho(g)P(x)
$$

for the policy and:

$$
V(g\cdot x)=V(x)
$$

for the value.

The test asserts maximum absolute error below:

$$
10^{-5}.
$$

### Why it was not trained

The equivariant implementation was mathematically useful and verified, but substantially more computationally expensive.

Given a limited training budget, that compute was more valuable when allocated to:

* larger beam search;
* more policy proposals;
* more future-hand samples;
* and longer training trajectories.

The released checkpoint therefore uses the conventional CNN.

The P4M implementation remains in the repository as an experimental research branch, not as a trained production model.

---

# Research History

The final system emerged from a much larger set of experiments.

This section is intentionally secondary to the released algorithm; these approaches were useful in understanding the problem but are not all present in the final training objective.

## Model-free RL

Early versions treated the problem as conventional:

$$
s_t\to a_t\to r_t\to s_{t+1}.
$$

Experiments included variations of:

* policy gradients;
* actor-critic;
* PPO;
* A2C-style training;
* n-step returns;
* GAE;
* value-based methods;
* distributional critics.

These agents learned useful placement heuristics but struggled with the short combinatorial interactions among the pieces in a hand.

**Lesson:** better return estimation did not remove the underlying planning problem.

---

## Off-policy return estimation

Retrace and V-trace-style methods were investigated to make exploratory/off-policy trajectories more useful.

They improved the flexibility of the training machinery but did not fundamentally change the decision-quality bottleneck.

**Lesson:** improving sample reuse is not the same as improving planning.

---

## Counterfactual evaluation

The environment makes it cheap to construct every legal consequence of a decision:

$$
a_1,\ldots,a_n
\rightarrow
s'_1,\ldots,s'_n.
$$

Evaluating alternative afterstates proved substantially more informative than relying only on the action actually sampled in a trajectory.

This was one of the key conceptual bridges from model-free RL toward explicit search.

**Lesson:** the ability to simulate alternative actions should be used directly rather than forcing a policy gradient to infer their consequences indirectly.

---

## Difficult-hand generation

A separate block-generation model was also explored as a way to expose the primary agent to difficult hands and board conditions.

This helped illustrate that difficult states are often produced by the interaction between unfavorable hands and earlier geometric mistakes.

However, the generator introduced additional model complexity while counterfactual search attacked the decision problem more directly.

---

## Search evolution

Once afterstate learning worked, explicit lookahead became the natural extension.

Search experiments evolved through several forms involving:

* broader tree expansion;
* learned backups;
* counterfactual continuation values;
* beam search;
* policy-guided top-\(k\);
* hard-max returns;
* frontier bootstrapping;
* root exploration;
* different pruning/finalization behavior.

The final version became intentionally simpler:

```text
policy proposes
→ native engine expands
→ exact reward ranks the beam
→ value bootstraps unresolved leaves
→ hard max backs up root actions
```

This simplicity improved both performance and interpretability.

---

## Symmetry experiments

The project also explored:

* cyclic groups;
* dihedral groups;
* D4 pose orbits;
* lifted convolutions;
* P4M group convolutions;
* equivariant attention;
* explicit \(D_4\times S_3\) testing.

The symmetry assumptions were real and the implementation passed its numerical equivariance tests.

However, the computational cost was not justified by the available training budget relative to simply spending more compute on explicit search.

---

## Overall conclusion

The strongest design principle to emerge was:

> **For a small environment that can be simulated extremely cheaply, explicit planning can be a better use of computation than endlessly increasing the expressive power of the policy network.**

The development progression was approximately:

```text
model-free RL
      │
      ├── PPO / actor-critic
      ├── GAE / n-step
      ├── distributional critics
      └── Retrace / V-trace
      │
      ▼
afterstate formulation
      │
      ├── counterfactual evaluation
      └── difficult-hand generation
      │
      ▼
policy + value heuristics
      │
      ▼
explicit lookahead
      │
      ▼
policy-guided beam search
      │
      ├── top-M proposals
      ├── hard-max backups
      ├── frontier bootstrapping
      ├── root Gumbel exploration
      └── no finalize-on-prune
      │
      ▼
native C++ bitboard search
      │
      ▼
released system
```

The final architecture is neither a pure neural policy nor a conventional hand-designed solver.

It is a **learned heuristic search system**.

---

# Repository Structure

A simplified view of the repository:

```text
Block-Blast-AI/
│
├── main.py
│   └── training configurations / entry point
│
├── config.py
│   └── experiment hyperparameters
│
├── core.py
│   ├── model construction
│   ├── local / DDP training
│   ├── checkpointing workflow
│   └── deterministic phase seeds
│
├── worker.py
│   ├── rollout collection
│   ├── search policy targets
│   ├── GAE / afterstate targets
│   └── evaluation
│
├── memory.py
│   └── rollout buffer
│
├── utils.py
│   ├── checkpoints
│   ├── logging
│   ├── LR schedulers
│   └── state packing/unpacking
│
├── render.py
│   └── Pygame simulator visualization
│
├── autoplay.py
│   └── Windows real-game screen/input controller
│
├── test_equivariance.py
│   └── exhaustive D4 × S3 policy/value symmetry test
│
├── model/
│   ├── cnn.py
│   │   └── released CNN policy/value networks
│   ├── p4mcnn.py
│   │   └── experimental equivariant networks
│   └── group.py
│       └── finite-group / coordinate utilities
│
├── policy_math/
│   ├── returns.py
│   ├── value_scale.py
│   └── mirror.py
│
├── bbengine/
│   ├── bindings/
│   │   └── pybind.cpp
│   ├── src/
│   │   ├── engine.cpp
│   │   ├── env.cpp
│   │   ├── search.cpp
│   │   ├── tables.cpp
│   │   └── ...
│   ├── bbengine/
│   │   ├── __init__.py
│   │   ├── _bbengine.cp312-win_amd64.pyd
│   │   └── _bbengine.cpython-312-x86_64-linux-gnu.so
│   ├── setup.py
│   └── pyproject.toml
│
├── chkpts/
│   └── chkpt3
│       └── released policy/value checkpoint
│
└── assets/
    └── LeagueSpartan-Bold.otf
```

---

# Limitations

### Commercial-game reproducibility

The commercial Block Blast application has exhibited inconsistent scoring behavior across versions and sessions. Real-game scores should therefore not be treated as a stable benchmark.

### Autoplay portability

`autoplay.py` is Windows- and calibration-specific. It depends on fixed visual geometry, Win32 input APIs, emulator layout, and game colors.

### Compute cost

The final training stage is expensive. On one RTX 5090, a single phase required roughly 16–18 hours.

### Search cost

The released checkpoint uses a beam width of 1024 and up to 32 policy proposals per parent. This is intentionally compute-heavy inference compared with direct policy execution.

### Evaluation cap

Evaluation terminates after at most 3000 placement iterations, so extremely long games can be administratively truncated.

### Exact hardware determinism

The code explicitly controls major RNG streams, but exact floating-point identity is not guaranteed across different GPU architectures, compiler versions, PyTorch builds, or CUDA implementations.

### P4M model

The equivariant implementation is tested but not accompanied by a trained production checkpoint.

---

# References

### Group-equivariant CNNs

Taco S. Cohen and Max Welling.
**Group Equivariant Convolutional Networks.**
ICML 2016.

https://proceedings.mlr.press/v48/cohenc16.html

---

## Disclaimer

This is an independent research/engineering project and is not affiliated with or endorsed by the developers of Block Blast.

The local simulator and fixed benchmark environment are provided for experimentation and reproducibility; the commercial-game controller is included only as an external integration demonstration.
