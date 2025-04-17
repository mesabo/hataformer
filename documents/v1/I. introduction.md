# 📘 Introduction

The rapid proliferation of Internet of Things (IoT) devices across domains such as smart grids, traffic systems,
healthcare, and industrial monitoring has led to an explosion of multivariate time series (MTS) data. These data
streams, characterized by complex temporal dependencies, variable periodicities, and dynamic noise patterns, demand
accurate and robust long-term forecasting models. However, capturing long-range temporal correlations in such diverse
environments remains a formidable challenge. While Transformer-based models have shown promise in modeling long-term
dependencies in NLP and vision, their adaptation to time series forecasting, especially in IoT scenarios, continues to
face fundamental limitations in both positional encoding and attention mechanism design.

Recent works have attempted to address these limitations from different angles. Models like PatchTST and Autoformer
reformulated input representations to capture local trends and seasonality, while iTransformer and TFEformer revisited
multivariate tokenization and variate-wise dynamics. DifFormer, MTST, and Crossformer explored multi-scale temporal
modeling through differencing and patch-based mechanisms. Despite these advances, two critical issues remain open.

**First**, most models adopt fixed positional encodings — sinusoidal, learnable vectors, or implicit indexes — which
fail to capture explicit temporal semantics such as time-of-day, day-of-week, or month-of-year. This static view of time
reduces model generalization across heterogeneous datasets, especially when periodic structures vary. **Second**, the
attention structures are either fully global, leading to high computation and diluted local signals, or purely local,
which limits the model’s ability to capture global patterns such as long-term seasonality or regime shifts. Furthermore,
most recent models perform recursive prediction or rigid horizon-level forecasting, without allowing interaction across
predicted time steps.

To overcome these shortcomings, we propose **TAPFormer**, a decoder-only Transformer architecture designed explicitly
for efficient and expressive long-term time series forecasting in IoT environments. Our model reimagines both positional
encoding and attention design through lightweight, learnable, and dynamic components, ensuring both **generality and
scalability** across domains.

---

## 🎯 Contributions

This paper makes the following key contributions:

1. **Temporal-Proj: A Learnable Temporal Encoding**  
   We propose a novel positional encoding scheme — **Temporal-Proj** — which embeds raw temporal features (e.g., hour,
   weekday, season) via a shared MLP. This allows the model to generalize better across datasets with different time
   structures and preserves meaningful temporal semantics, outperforming traditional sinusoidal and learned embeddings.

2. **Dual-Scale Attention With Gating and Bias**  
   TAPFormer introduces **Dual-Scale Attention** via head-splitting: a gated blend of local and global self-attention
   within each attention layer. A **soft locality bias** further enhances locality modeling without hard masking. This
   flexible framework enables the model to dynamically adjust its temporal scope at **no additional computational cost
   **.

3. **Fully Decoder-Only, Single-Step Forecasting**  
   Unlike encoder-decoder architectures or autoregressive pipelines, TAPFormer predicts the entire output horizon in one
   step using a **streamlined decoder-only design**. This design improves efficiency, avoids error accumulation, and
   allows holistic refinement across the forecast window.

4. **State-of-the-Art Accuracy With Lower Complexity**  
   TAPFormer achieves **state-of-the-art performance** across 8 real-world benchmarks, surpassing recent models like
   iTransformer, MCformer, and TFEformer. Notably, it improves forecasting accuracy by **+5.7% on average** (MSE
   reduction) over strong baselines, while reducing inference cost by up to **40%** through its decoder-only structure
   and localized attention.
