## 🔧 Problem Statement & Motivation

### 1. Main Problem

Long-term time series forecasting is inherently difficult due to:

- Complex, multi-scale temporal patterns (e.g., daily, weekly, seasonal)
- Non-stationarity and context-specific behavior
- Inflexibility of standard positional encoding methods (e.g., sinusoidal, learnable)

While transformer-based models have shown promise, most suffer from:

- Rigid or handcrafted positional encodings
- Uniform attention across all time steps
- Inefficiency when modeling long-term dependencies
- Lack of adaptation to dynamic local/global temporal features

---

### 2. Specific Issues Tackled

TAPFormer addresses several key limitations in prior models:

| Challenge                            | TAPFormer Solution                                                                 |
|--------------------------------------|-------------------------------------------------------------------------------------|
| **Rigid positional encodings**       | ✅ *Temporal-Proj*: Learnable projection of raw time features (e.g. hour, day) that generalizes across sequences and tasks. |
| **Static attention behavior**        | ✅ *Local/Global Gating*: A learnable sigmoid gate blends local and global attention. In multi-head settings, this is applied per head. With a single head, the gate acts as a **global trade-off** controller. |
| **Limited locality bias**            | ✅ *Soft Local Bias*: Learnable or scheduled attention mask injected into attention logits to guide locality. |
| **Overhead from full encoder-decoder** | ✅ *Decoder-Only Design*: Lightweight and efficient model relying on autoregressive decoding with attention refinement. |

---

### 3. Purpose & Vision

The goal of TAPFormer is to **rethink how transformers understand time**:

- Move beyond handcrafted or static temporal priors
- Capture both fine-grained (local) and long-range (global) temporal dependencies
- Improve forecasting accuracy while reducing model complexity
- Enable explainability via explicit temporal-aware components

---

### 🧠 Summary

> **TAPFormer** introduces a decoder-only transformer for long-term time series forecasting. It redefines positional
encoding through **Temporal-Proj**, a normalized projection of raw temporal features, and enhances attention via **gated
local/global mixing** and **soft locality bias**.
>
> **Note:** In single-head configurations, the local/global gating becomes a shared controller across the entire
attention layer, retaining the adaptivity at lower computational cost.

## Motivation & Problem Formulation

Time series forecasting — especially over long horizons — remains a difficult task due to the multi-scale, dynamic
nature of real-world temporal patterns. Standard transformer-based models have made progress, but they still suffer from
two major limitations:

**First**, most models use fixed positional encodings like sinusoidal functions or simple learnable embeddings. These
approaches are static, lack explicit temporal semantics (such as hour, day, or season), and struggle to generalize
across datasets with different time structures. They treat time as a sequence index, not as a contextual signal.

**Second**, many models are limited to either local or global attention mechanisms, with no dynamic blending of the two.
Forecasting is often done recursively (step-by-step) or in a rigid parallel form, which restricts accuracy and
expressiveness. These models lack the flexibility to handle varying temporal scopes in a unified way.

TAPFormer addresses both of these challenges:

- It introduces **Temporal-Proj**, a learnable projection of raw temporal features that captures seasonality and context
  directly, offering better generalization and richer encodings than fixed methods.
- It adopts **Dual-Scale Attention via Head Splitting**, where attention heads are explicitly divided across local and
  global scopes — combined with **gating** and **soft locality bias** — without increasing compute cost.
- Forecasting is performed in a single step, rather than recursively, allowing the model to refine predictions across
  the entire horizon in parallel.

> In multi-head setups, each head learns its own balance of local vs. global focus. With a single head, the mechanism
still operates globally — maintaining dynamic control with minimal overhead.

By rethinking both positional encoding and attention structure, TAPFormer delivers a compact, efficient, and
interpretable architecture that better captures temporal dependencies at multiple scales.