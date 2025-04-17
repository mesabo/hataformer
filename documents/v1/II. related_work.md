# 📚 Related Work

Recent progress in multivariate time series forecasting (MTSF) has been largely driven by the introduction of
Transformer-based architectures, which enable long-range dependency modeling through self-attention. However, adapting
these architectures to the specific demands of temporal data — especially in the IoT context — has revealed several core
challenges. Below, we review related research in four major categories relevant to TAPFormer: **positional encoding**, *
*multi-scale and long-range attention**, **channel dependency strategies**, and **decoder-based forecasting models**.

---

### 🔢 Positional Encoding in Time Series Transformers

Most Transformer models originally designed for NLP employ **sinusoidal or learnable positional encodings**, which do
not generalize well to the structured and often periodic nature of time series. Models like **PatchTST** and *
*Autoformer** rely on time-point indexes or hand-crafted sinusoidal embeddings, which are insufficient to capture rich
temporal semantics such as time-of-day, seasonal cycles, or weekday/weekend patterns.  
**TFEformer** and **iTransformer** made important progress by introducing series-wise tokenization and shallow variate
representations, but they still suffer from limited depth in semantic encoding.  
In contrast, **TAPFormer** introduces **Temporal-Proj**, a learnable MLP that projects raw time features (e.g., hour,
day, season) into the latent space, significantly enhancing temporal awareness and improving generalization across
datasets with different temporal structures.

---

### 🔭 Multi-Scale and Long-Range Temporal Attention

Modeling temporal dependencies across multiple scales is crucial for long-term forecasting. Traditional Transformer
attention is quadratic in time, making it inefficient for long sequences. Several models have attempted to address this:

- **MTST**, **MTPNet**, and **DifFormer** explicitly model multi-resolution patterns through branch splitting or
  patch-based modules.
- **Autoformer** and **FEDformer** leverage series decomposition and Fourier transforms, respectively, to represent
  seasonalities.
- **MCformer** and **MR-Transformer** address temporal mixing using patching and channel-wise convolutions.

While these methods improve scalability and accuracy, they often rely on static structures or heuristic scale
selections.  
**TAPFormer** contributes a novel approach by **splitting attention heads** into local and global components, using *
*learnable gating** and a **soft locality bias**. This allows the model to dynamically adjust its receptive field based
on content — without modifying architectural depth or increasing attention cost.

---

### 🔁 Channel-Independent vs. Dependent Modeling

In multivariate forecasting, the trade-off between **channel independence (CI)** and **channel dependency (CD)** has
gained attention.  
CI models such as **DLinear** and **PatchTST** generalize well but ignore inter-series correlations.  
CD models like **GTformer**, **SageFormer**, and **PCDformer** use graph networks or shared encoders to capture
cross-series dynamics but are prone to overfitting or increased complexity.  
**MCformer** attempts a hybrid strategy by mixing channels at set intervals.

TAPFormer circumvents this dichotomy by employing **variational attention across heads** — the **Dual-Scale Attention
mechanism** allows selective dependency modeling without overmixing or rigid structure. In single-head configurations,
the global-local gating becomes a shared controller, keeping the design lightweight and interpretable.

---

### 🧠 Decoder-Based Forecasting Architectures

While most state-of-the-art models use encoder-decoder architectures (e.g., **Informer**, **Crossformer**), recent
trends show promise in **simplified decoder-only pipelines**.  
**PETformer** introduces placeholder-based future tokens, and **TiDE** uses lightweight MLP decoders.  
These approaches reduce complexity and improve speed but often lack expressiveness or depth in temporal modeling.

TAPFormer builds on this trend by adopting a **pure decoder-only architecture**, augmented with **Temporal-Proj**, *
*adaptive attention**, and **a wide receptive field**. Forecasting is performed in a **single step**, avoiding recursive
prediction and reducing cumulative error — a key advantage over autoregressive and step-wise models.

---

## 🧩 Summary

TAPFormer integrates key ideas from positional encoding, attention design, and architectural efficiency — but uniquely
reconfigures them through **learnable temporal projection**, **dual-scale gated attention**, and **decoder-only
forecasting**. It builds upon, but also clearly diverges from, recent Transformer variants such as TFEformer,
iTransformer, MCformer, and PETformer, offering a new solution for robust, scalable, and interpretable long-term
forecasting in IoT and other real-world time series environments.
