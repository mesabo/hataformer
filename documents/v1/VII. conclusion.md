# ✅ Conclusion

In this paper, we presented **TAPFormer**, a novel decoder-only Transformer architecture for long-term multivariate time
series forecasting, with a focus on Internet of Things (IoT) scenarios. By rethinking two foundational aspects of
Transformer models — **positional encoding** and **attention structure** — TAPFormer addresses key challenges that limit
the performance and generalizability of existing methods.

We introduced **Temporal-Proj**, a learnable projection of raw temporal features, which captures fine-grained semantic
context such as hour, day, and season. This replaces static or learned index-based encodings with temporally meaningful
representations, enhancing adaptability across datasets with diverse periodic structures. We also proposed a *
*Dual-Scale Attention mechanism** that explicitly splits attention heads into local and global scopes, combined through
a learnable **gating controller** and **soft locality bias**. This design enables dynamic adjustment of temporal context
within each attention layer, without additional computational cost.

TAPFormer is trained to perform **parallel, non-autoregressive and non-recursive forecasting**, enabling horizon-wide prediction in a
single step. Extensive experiments across eight real-world datasets demonstrate that TAPFormer achieves *
*state-of-the-art accuracy**, outperforming recent models such as iTransformer, MCformer, and TFEformer. Our approach
delivers an average **5.7% reduction in MSE**, while reducing inference overhead by up to **40%**, making it both
effective and efficient for deployment in practical IoT systems.

In summary, TAPFormer offers a compact, interpretable, and scalable solution for long-range forecasting, advancing the
frontier of Transformer-based time series modeling. We believe its design principles can inform future architectures
that aim to unify temporal semantics with efficient, multi-scale attention.
