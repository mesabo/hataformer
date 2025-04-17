# Paper Title

| Section        | Details                                                                 |
|----------------|-------------------------------------------------------------------------|
| **Motivation** | Why this paper exists                                                   |
| **Problem**    | What specific challenge it tackles                                      |
| **Solution**   | The core method or idea introduced                                      |
| **Approach**   | How they implemented/tested the solution                                |
| **Pros**       | What works well / key strengths                                         |
| **Cons**       | Weaknesses / trade-offs                                                 |
| **Limitations**| What the paper doesn't cover / doesn't solve                           |
| **Future Work**| What authors suggest or hint at for follow-up                          |



## 1. A Transformer-Based Industrial Time Series Prediction Model With Multivariate Dynamic Embedding

| Section        | Details                                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| A Transformer-Based Industrial Time Series Prediction Model With Multivariate Dynamic Embedding (2025)                                              |
| **Motivation** | Address challenges in industrial time series prediction due to distribution drift in dynamic industrial environments.                               |
| **Problem**    | Existing models fail to adapt to dynamic changes in multivariate data distribution, leading to poor prediction in real scenarios.                   |
| **Solution**   | MDEformer: introduces multivariate dynamic embedding and a generative pretraining strategy.                                                          |
| **Approach**   | Combines patching, dynamic embedding (DMD, attention), and bidirectional residual connection with Transformer and generative pretraining.           |
| **Pros**       | Effectively handles distribution drift; leverages dynamic mode structure; improves robustness via CI strategy; strong performance on real datasets. |
| **Cons**       | Potential computational complexity; model structure and training require tuning; domain-specific validation.                                         |
| **Limitations**| Focuses only on zinc smelting data; generalization to other industrial domains not verified.                                                        |
| **Future Work**| Extend to broader industrial datasets; explore lightweight variants; integrate with online learning frameworks.                                     |

## 2. DEformer: Dual Embedded Transformer for Multivariate Time Series Forecasting​

| Section        | Details                                                                                                                                             |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| DEformer: Dual Embedded Transformer for Multivariate Time Series Forecasting (2024)                                                                |
| **Motivation** | Enhance modeling of both temporal and spatial dependencies in MTS forecasting, which are often embedded together, leading to loss of specificity.  |
| **Problem**    | Existing models merge temporal and spatial dependencies early, hampering representation clarity and learning effectiveness.                        |
| **Solution**   | A dual-encoder architecture where temporal and spatial embeddings are learned independently before being fused.                                     |
| **Approach**   | Two linear projections produce variate and temporal tokens separately; each encoder processes one, outputs are fused for forecasting.               |
| **Pros**       | Captures independent dynamics more effectively; outperforms SOTA models; empirical validation across real datasets.                                |
| **Cons**       | Increases model complexity; requires more memory and processing; lacks interpretability for some domains.                                          |
| **Limitations**| Embedding strategy not validated across all time series domains; impact of dual attention untested under noisy data.                              |
| **Future Work**| Improve efficiency, explore cross-attention between encoders, and apply to online/streaming MTS settings.                                          |

## 3. Dynamic Long-Term Time-Series Forecasting via Meta Transformer Networks (MANTRA)

| Section        | Details                                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Dynamic Long-Term Time-Series Forecasting via Meta Transformer Networks (2024)                                                                              |
| **Motivation** | Improve adaptability and scalability in long-term forecasting under dynamic and nonstationary conditions.                                                   |
| **Problem**    | Existing models fail to adapt quickly to concept drifts in long sequences; deep learning is computationally intensive and slow in nonstationary contexts.  |
| **Solution**   | MANTRA: a meta-transformer architecture combining fast (task-specific) and slow (representation-focused) learners with URT attention-based routing.        |
| **Approach**   | Combines autoformer base with ensemble of fast learners and a slow learner; fast adaptation via attention-based URT to fuse predictions.                   |
| **Pros**       | Adaptable to distribution drift; handles dynamic environments; outperforms baselines by 3%+ on multiple datasets.                                           |
| **Cons**       | Training is resource-intensive; ensemble structure may overfit; complexity of meta-learning pipeline.                                                        |
| **Limitations**| Limited real-world datasets tested; effectiveness relies on URT tuning; masking strategy sensitive to hyperparameters.                                     |
| **Future Work**| Explore broader tasks (e.g. anomaly detection), multi-modal forecasting, online URT adaptation, hybrid URT strategies.                                     |

## 4. Evaluation of Different Deep Learning Methods for Meteorological Element Forecasting

| Section        | Details                                                                                                                                      |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Evaluation of Different Deep Learning Methods for Meteorological Element Forecasting (2024)                                                 |
| **Motivation** | Assess effectiveness of DL models across multiple meteorological variables and conditions in one consistent benchmark.                      |
| **Problem**    | Lack of comprehensive comparative evaluation across DL architectures for different weather forecasting targets.                             |
| **Solution**   | Benchmarks LSTM, GRU, Bi-LSTM, MTGNN, Informer, Autoformer, and Deep Transformer on key meteorological tasks.                              |
| **Approach**   | Tests on ERA5 dataset; evaluates model training efficiency and prediction accuracy with various sliding windows and time steps.             |
| **Pros**       | Clear comparison framework; identifies model strengths per task (e.g., Informer best for ET/SSR, MTGNN good for T); easy reproducibility. |
| **Cons**       | Focused only on one region (Hebei farmland); generalizability unclear; less architectural innovation.                                       |
| **Limitations**| Excludes ensemble or hybrid models; lacks treatment for extreme events or sparse data.                                                      |
| **Future Work**| Extend to other climate zones; incorporate transfer learning; explore spatiotemporal ensemble methods.                                      |


## 5. Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis

| Section        | Details                                                                                                                                                  |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis (2025)                              |
| **Motivation** | Resolve contradictions and variability in reported performance of MTS models across studies.                                                            |
| **Problem**    | Lack of unified evaluation framework and failure to consider dataset heterogeneity (temporal/spatial patterns).                                         |
| **Solution**   | BasicTS+: a benchmarking suite with unified training, evaluation, and heterogeneity-aware dataset categorization.                                       |
| **Approach**   | Benchmarks 30+ models on 20 datasets, analyzes how spatial/temporal heterogeneity impacts model performance (e.g., GCNs vs. Linear).                   |
| **Pros**       | Addresses reproducibility issues; explains contradictory results; aids in proper model selection per dataset type.                                     |
| **Cons**       | Does not propose a new model; computationally heavy; limited insights on newer cross-modal setups.                                                     |
| **Limitations**| Benchmark excludes non-deep learning baselines; not real-time; interpretability not addressed.                                                          |
| **Future Work**| Incorporate hybrid and online learning models, expand to more heterogeneous datasets (e.g., financial + sensor mix).                                   |

## 6. Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis

| Section        | Details                                                                                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis (2025)                                 |
| **Motivation** | Resolve inconsistent performance findings and benchmark fragmentation across MTS forecasting literature.                                                   |
| **Problem**    | Unreliable performance comparisons due to varying training pipelines and ignoring dataset heterogeneity.                                                   |
| **Solution**   | BasicTS+: a unified benchmarking framework with standardized pipelines and classification of MTS heterogeneity.                                            |
| **Approach**   | Benchmarks 30+ models across 20 datasets; includes a heterogeneity-aware classification (temporal/spatial patterns); introduces new evaluation criteria.   |
| **Pros**       | Reveals contradictions in prior studies; unifies training/evaluation; demonstrates dataset-specific model effectiveness.                                   |
| **Cons**       | Does not propose a new model; only focuses on benchmarking; high compute cost.                                                                            |
| **Limitations**| No interpretability techniques or real-time benchmarking; limited domain-specific customization.                                                           |
| **Future Work**| Expand to more dataset types; include hybrid/online learning models; enhance interpretability modules.                                                     |

## 7. Long-Term Multivariate Time-Series Forecasting Model Based on Gaussian Fuzzy Information Granules

| Section        | Details                                                                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Long-Term Multivariate Time-Series Forecasting Model Based on Gaussian Fuzzy Information Granules (2024)                                                 |
| **Motivation** | Extend fuzzy information granules (Fuzzy IG) from univariate to multivariate long-term forecasting for improved interpretability and robustness.         |
| **Problem**    | Lack of modeling frameworks using GFIGs for multivariate settings; existing methods are limited to univariate or overly simplified GLFIGs.              |
| **Solution**   | A hybrid model combining BPNN, LSTM, Transformer, and a novel polynomial-core GFIG segmentation and representation strategy.                            |
| **Approach**   | Introduces optimized GFIG construction, adaptive segmentation using membership functions, and a joint neural network for final predictions.              |
| **Pros**       | Strong interpretability; robust to uncertainty and noise; outperforms standard methods on multiple benchmarks.                                           |
| **Cons**       | Heavy preprocessing; relies on manual tuning of polynomial core orders and segmentation parameters.                                                      |
| **Limitations**| Limited to GFIG-centric granulation; generalization to non-polynomial trends or real-time systems unproven.                                              |
| **Future Work**| Generalize to other types of fuzzy granules; integrate with streaming forecasting or real-time processing.                                                |

## 8. LTScoder: Long-Term Time Series Forecasting Based on a Linear Autoencoder Architecture

| Section        | Details                                                                                                                                              |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| LTScoder: Long-Term Time Series Forecasting Based on a Linear Autoencoder Architecture (2024)                                                      |
| **Motivation** | Reduce complexity and memory cost of Transformer-based LTSF methods for edge environments, while preserving long-term dependencies.                |
| **Problem**    | Transformer models are too resource-intensive for low-power devices; linear models lack feature representation power.                               |
| **Solution**   | A lightweight autoencoder using linear projection to extract features and conduct future prediction with reduced complexity.                        |
| **Approach**   | Encodes input into latent vectors using linear AE, then decodes for future forecasting; comparison with PatchTST and DLinear.                       |
| **Pros**       | 17x faster than PatchTST; uses only 2% of parameters; maintains competitive accuracy; edge-computing ready.                                         |
| **Cons**       | Lacks model expressiveness for highly nonlinear systems; some loss in prediction accuracy on certain datasets.                                     |
| **Limitations**| Does not explore hybrid models; no frequency or attention mechanisms; mainly benchmarked in edge scenarios.                                        |
| **Future Work**| Explore hybrid encoder-decoder design with attention; extend to streaming TS; fuse with memory-efficient Transformers.                             |

## 9. MCformer: Multivariate Time Series Forecasting With Mixed-Channels Transformer

| Section        | Details                                                                                                                                                 |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| MCformer: Multivariate Time Series Forecasting With Mixed-Channels Transformer (2024)                                                                 |
| **Motivation** | Balance between channel-independence and interchannel dependencies in multivariate TS to prevent correlation forgetting.                              |
| **Problem**    | CI models improve generalization but ignore inter-channel dependencies; CD models overfit or disrupt long-term learning.                             |
| **Solution**   | A mixed-channel strategy fusing strengths of both CI and CD approaches using a customized patch-based Transformer.                                    |
| **Approach**   | Channel mixing → patch extraction → transformer encoder; applies RevIN, shared parameter blocks, and selective fusion of temporal/variable patterns.  |
| **Pros**       | Outperforms CI/CD on large datasets; preserves long-term dependencies; shows scalability with # of channels.                                         |
| **Cons**       | Requires tuning channel-mix ratios; complexity increases over pure CI.                                                                                 |
| **Limitations**| Needs evaluation on ultra-high frequency/sparse TS; lacks interpretability mechanisms.                                                                |
| **Future Work**| Adapt to dynamically varying channels; integrate graph learning for topological relations among channels.                                             |

## 10. MR-Transformer: Multiresolution Transformer for Multivariate Time Series Prediction

| Section        | Details                                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| MR-Transformer: Multiresolution Transformer for Multivariate Time Series Prediction (2025)                                                                  |
| **Motivation** | Model both long-term and short-term temporal dependencies and disentangle variable-consistent and variable-specific patterns.                               |
| **Problem**    | Existing Transformer models ignore local (short-term) and variable-specific signals critical for nuanced MTS modeling.                                       |
| **Solution**   | Introduces Long Short-term Attention and Variable-specific Temporal Convolution modules within a unified Transformer encoder.                              |
| **Approach**   | Adaptive DTW-based segmentation for short-term blocks + long-range attention + channel-wise convolution to extract multiresolution features.                |
| **Pros**       | Captures both fine-grained and coarse patterns; excels in high-variance, noisy MTS; visualization shows clear pattern learning.                            |
| **Cons**       | DTW alignment incurs compute overhead; may require significant training data to converge.                                                                  |
| **Limitations**| Lacks frequency-domain or hybrid fusion (e.g., wavelet); does not support real-time adaptation.                                                            |
| **Future Work**| Extend to multimodal TS (e.g., text + sensor); combine with Fourier/decomposition modules; investigate real-time segmentation.                             |

## 11. Multi-Resolution Expansion of Analysis in Time-Frequency Domain for Time Series Forecasting

| Section        | Details                                                                                                                                                     |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multi-Resolution Expansion of Analysis in Time-Frequency Domain for Time Series Forecasting (2024, IEEE TKDE)                                              |
| **Motivation** | Capture missing and redundant information in both time and frequency domains to enhance deep models' understanding of long-term patterns.                 |
| **Problem**    | Existing models downsample for multi-resolution info but miss valuable data due to low resolution and discrete sampling (“Fence Effect”).                  |
| **Solution**   | TFMRN: a Time-Frequency Multi-Resolution Expansion Network that extends both high and low resolution in time and frequency domains.                        |
| **Approach**   | Combines time interpolation + downsampling + FFT with top-k component selection + information gating unit to filter expanded data for robust learning.     |
| **Pros**       | Captures richer global/local patterns; robust to noisy inputs; generalizable to both univariate and multivariate TS.                                       |
| **Cons**       | Interpolation and FFT increase computational burden; masking needs careful hyperparameter tuning.                                                          |
| **Limitations**| Focused on synthetic interpolation; lacks interpretability or real-time extension.                                                                         |
| **Future Work**| Extend to real-time augmentation; integrate into online learning; investigate dynamic resolution strategies.                                                |

## 12. Multi-Scale Attention Flow for Probabilistic Time Series Forecasting

| Section        | Details                                                                                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multi-Scale Attention Flow for Probabilistic Time Series Forecasting (2024, IEEE TKDE)                                                                        |
| **Motivation** | Improve probabilistic forecasting by modeling cross-series correlations and local/global time dependencies without autoregressive drawbacks.                 |
| **Problem**    | Vanilla attention lacks relative location awareness; autoregressive models suffer from cumulative errors and inefficiency.                                  |
| **Solution**   | MANF: a Multi-scale Attention Normalizing Flow model combining relative positional attention with normalizing flows for non-autoregressive distribution.    |
| **Approach**   | Hierarchical multi-scale attention with dynamic position embedding in encoder, followed by RealNVP-based conditional flows in decoder for uncertainty modeling. |
| **Pros**       | Avoids autoregressive error accumulation; models rich temporal hierarchy and uncertainty; state-of-the-art results across datasets.                          |
| **Cons**       | Increased architectural complexity; requires large data for reliable flow modeling.                                                                          |
| **Limitations**| Not optimized for sparse or irregular time series; lacks integration with exogenous variable modeling.                                                        |
| **Future Work**| Explore hybrid attention-flows with external features; apply to event-based or missing-data scenarios.                                                       |

## 13. Multi-Scale Transformer Pyramid Networks for Multivariate Time Series Forecasting

| Section        | Details                                                                                                                                                  |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multi-Scale Transformer Pyramid Networks for Multivariate Time Series Forecasting (2024, IEEE Access)                                                  |
| **Motivation** | Overcome limited-scale granularity in prior transformer architectures by modeling fine to coarse-scale dependencies.                                     |
| **Problem**    | Existing models grow temporal scale exponentially, missing key seasonalities (e.g., daily cycles) and restricting flexibility.                           |
| **Solution**   | MTPNet: A transformer pyramid framework using Dimension-Invariant (DI) embedding and unconstrained patching for flexible multi-scale modeling.          |
| **Approach**   | Uses DI embedding (1D CNN preserving spatial/temporal info), patching into various scales, parallel multi-transformers, plus symmetric encoder-decoder. |
| **Pros**       | Fine-to-coarse scale flexibility; improved learning of varied temporal patterns; outperforms SoTA on high-resolution datasets.                          |
| **Cons**       | Increased model complexity and memory usage; DI embedding may not capture inter-channel correlations directly.                                          |
| **Limitations**| Does not explore attention sparsity or graph-based variable dependencies.                                                                               |
| **Future Work**| Combine with GNNs for spatial dependencies; extend to hybrid modalities (e.g., sensors + text); optimize patch scheduling dynamically.                   |

## 14. Multivariate Resource Usage Prediction With Frequency-Enhanced and Attention-Assisted Transformer

| Section        | Details                                                                                                                                               |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multivariate Resource Usage Prediction With Frequency-Enhanced and Attention-Assisted Transformer (FISFA) (2024, IEEE IoT Journal)                  |
| **Motivation** | Improve prediction of volatile cloud resource usage (CPU/memory) by enhancing both noise handling and frequency feature extraction.                  |
| **Problem**    | Traditional models struggle with nonlinearity, volatility, and noisy frequency information in cloud computing time series.                           |
| **Solution**   | FISFA: Combines Savitzky–Golay filter + FEDformer + frequency-enhanced attention module (FECAM) + metaheuristic optimizer (GSPSO).                   |
| **Approach**   | SG filter for noise smoothing → frequency-domain modeling (Fourier/DCT) → FEDformer for global dependencies → FECAM for frequency-aware attention.  |
| **Pros**       | Strong in high-noise/volatility environments; captures temporal + frequency domain features; outperforms vanilla LSTM, Transformer, Informer.       |
| **Cons**       | High model complexity; needs metaheuristic tuning for optimal hyperparams; not real-time adaptive.                                                   |
| **Limitations**| Focused on cloud data only; lacks generalization tests on non-volatile or multimodal settings.                                                       |
| **Future Work**| Generalize to other infrastructure/time series types; explore adaptive hyperparameter learning (online GSPSO).                                       |

## 15. Multivariate Segment Expandable Encoder-Decoder Model for Time Series Forecasting

| Section        | Details                                                                                                                                                  |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multivariate Segment Expandable Encoder-Decoder Model for Time Series Forecasting (MSEED) (2024, IEEE Access)                                          |
| **Motivation** | Enable extreme-adaptive and rolling multivariate forecasting for skewed, heavy-tailed time series like hydrologic data.                                 |
| **Problem**    | Existing models fail on long-term, skewed MTS with rare extreme events and require real-time rolling updates.                                           |
| **Solution**   | MSEED: Segment-wise encoder-decoder with GMM-based oversampling, quantile prediction layers, and a short-term-enhanced subnet for rolling forecasting. |
| **Approach**   | Combines feature assembling → PLR-like segmentation → segment quantile forecasting → rolling predictions every 4 hours (short-term bias-corrected).   |
| **Pros**       | Excels on extreme, non-Gaussian distributions; robust long- and short-term performance; interpretable segment-wise output.                              |
| **Cons**       | Tailored to hydrologic data; GMM-based oversampling may not generalize to all domains.                                                                  |
| **Limitations**| Segment-based prediction may lose inter-timestamp continuity; high memory for multi-layer quantile output.                                              |
| **Future Work**| Apply to broader domains (e.g., finance, healthcare); integrate graph/contextual side-info; explore generative extensions for missing data.             |

## 16. Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer

| Section        | Details                                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer (PCDformer), IEEE TAI, 2024             |
| **Motivation** | Address overfitting and inefficiencies due to information coupling and tangled temporal patterns in multivariate time series.                               |
| **Problem**    | Difficulty learning variable-specific temporal dependencies and disentangling overlapping trends/seasonal patterns in long sequence forecasting.            |
| **Solution**   | PCDformer: Uses parallelized convolutions per variable, series decomposition (trend/seasonal), and sparse self-attention to disentangle and focus learning.|
| **Approach**   | Converts 1D input into 2D imagelike format via Dim-trans → M-Inception convolutions per variable → series decomposition in encoder/decoder → sparse attention. |
| **Pros**       | Reduces info coupling; improves long-term learning; outperforms SoTA on multiple datasets (traffic, weather, energy).                                       |
| **Cons**       | Requires careful kernel tuning and decompositional window sizes; computationally heavier than simpler alternatives.                                         |
| **Limitations**| Only suitable for time series with clear season/trend; less effective on erratic or sparse patterns.                                                         |
| **Future Work**| Real-time streaming forecasting; adaptive decomposition methods; extend to multimodal inputs.                                                               |

## 16. Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer

| Section        | Details                                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multivariate Time-Series Modeling and Forecasting With Parallelized Convolution and Decomposed Sparse-Transformer (PCDformer), IEEE TAI, 2024             |
| **Motivation** | Address overfitting and inefficiencies due to information coupling and tangled temporal patterns in multivariate time series.                               |
| **Problem**    | Difficulty learning variable-specific temporal dependencies and disentangling overlapping trends/seasonal patterns in long sequence forecasting.            |
| **Solution**   | PCDformer: Uses parallelized convolutions per variable, series decomposition (trend/seasonal), and sparse self-attention to disentangle and focus learning.|
| **Approach**   | Converts 1D input into 2D imagelike format via Dim-trans → M-Inception convolutions per variable → series decomposition in encoder/decoder → sparse attention. |
| **Pros**       | Reduces info coupling; improves long-term learning; outperforms SoTA on multiple datasets (traffic, weather, energy).                                       |
| **Cons**       | Requires careful kernel tuning and decompositional window sizes; computationally heavier than simpler alternatives.                                         |
| **Limitations**| Only suitable for time series with clear season/trend; less effective on erratic or sparse patterns.                                                         |
| **Future Work**| Real-time streaming forecasting; adaptive decomposition methods; extend to multimodal inputs.                                                               |

## 17. PETformer: Long-Term Time Series Forecasting via Placeholder-Enhanced Transformer

| Section        | Details                                                                                                                                                      |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| PETformer: Long-Term Time Series Forecasting via Placeholder-Enhanced Transformer (IEEE TETCI, 2025)                                                        |
| **Motivation** | Restore confidence in Transformer efficacy for LTSF amid claims that simpler models (e.g., DLinear) outperform them.                                        |
| **Problem**    | Transformers struggle with temporal continuity, sparse semantics in time series, and neglect inter-channel correlations.                                    |
| **Solution**   | PETformer: Introduces Placeholder-Enhanced Technique (PET) + Long Sub-sequence Division (LSD) + Multi-channel Separation and Interaction (MSI).             |
| **Approach**   | Encoder-only model where future time steps are reserved as learnable placeholders, enabling bidirectional attention; applies RevIN, LSD patches, and MSI blocks. |
| **Pros**       | Restores temporal continuity via placeholders; improves performance on both small and large datasets; state-of-the-art across 8 public datasets.            |
| **Cons**       | More complex architecture; reliance on placeholder representation could affect robustness on highly nonstationary data.                                     |
| **Limitations**| Limited ablation on real-time inference; fixed patching windows may reduce generalizability.                                                                |
| **Future Work**| Dynamic placeholder learning; extend to multimodal and irregularly sampled time series.                                                                    |

## 18. SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting

| Section        | Details                                                                                                                                                          |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting (IEEE IoT Journal, 2024)                                                  |
| **Motivation** | Most models either ignore inter-series dependencies or treat all series equally, limiting forecasting for heterogeneous multivariate time series.               |
| **Problem**    | Inadequate modeling of both intra- and inter-series dynamics; mixing or isolating series blindly results in information loss or redundancy.                    |
| **Solution**   | SageFormer: Combines Transformer encoder with graph-enhanced attention using global tokens for series-aware interaction modeling.                               |
| **Approach**   | Introduces learnable global tokens + message-passing GNN layers between them; graph aggregation reduces inter-series redundancy; Transformer encodes temporal info. |
| **Pros**       | Captures both intra-/inter-series dependencies; scalable to long series and many channels; generalizable across Transformer variants.                           |
| **Cons**       | Graph structure learning requires careful tuning; interpretability may be lower due to joint modeling of graph and attention layers.                            |
| **Limitations**| Less effective when inter-series correlation is weak; graph aggregation may dilute unique series features.                                                       |
| **Future Work**| Apply to hybrid tasks (e.g., anomaly detection); explore dynamic graph learning; improve robustness under distribution shifts.                                 |

## 19. SAD-Net: Saliency-Aware Dual Embedded Attention Network for MTS Forecasting in IT Operations

| Section        | Details                                                                                                                                                           |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Saliency-Aware Dual Embedded Attention Network for Multivariate Time-Series Forecasting in Information Technology Operations (IEEE TII, 2024)                   |
| **Motivation** | IT system data is often aperiodic, spiky, and nonlinear—challenging for traditional forecasting models relying on periodic assumptions.                          |
| **Problem**    | Inability of existing models to forecast aperiodic and irregular AMTS data due to inadequate saliency extraction and feature fusion.                            |
| **Solution**   | SAD-Net: Combines CNN+Transformer+GRU in dual-branch setup, integrating Saliency Residual Module (SRM) and Channel Attention Module (CAM) + fusion.             |
| **Approach**   | One branch captures saliency via spectral residual and attention; the other learns long/short dependencies; FFM fuses both outputs.                             |
| **Pros**       | Strong performance on real-world IT and public datasets; saliency-aware processing enhances robustness to spiky data.                                            |
| **Cons**       | High model complexity; tuning fusion and CAM parameters is non-trivial.                                                                                          |
| **Limitations**| May not generalize to regular periodic datasets; lacks interpretability of which features dominate prediction.                                                    |
| **Future Work**| Explore 3D extensions for cross-time-variable fusion; apply to anomaly detection in IT operations.                                                               |

## 20. STformer: Spatial and Temporal Attention-Enabled Transformer for Multivariate Short-Term Residential Load Forecasting

| Section        | Details                                                                                                                                                       |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Spatial and Temporal Attention-Enabled Transformer Network for Multivariate Short-Term Residential Load Forecasting (IEEE TIM, 2023)                         |
| **Motivation** | Existing short-term residential load forecasting ignores spatial correlation across households and cannot quantify forecast uncertainty.                      |
| **Problem**    | Transformer models lack mechanisms for spatial dynamics across homes; probabilistic forecasting is also unsupported.                                         |
| **Solution**   | STformer: Combines spatial attention with temporal autocorrelation modules + MC dropout for uncertainty estimation.                                          |
| **Approach**   | Encoder-decoder structure with time-series decomposition → autocorrelation → spatial attention → trend/seasonal disentanglement → probabilistic prediction. |
| **Pros**       | Captures spatial-temporal dependencies; provides both deterministic and probabilistic forecasts; achieves SoTA accuracy on NY and LA datasets.              |
| **Cons**       | High training cost due to MC dropout sampling; not designed for very long-term forecasting.                                                                  |
| **Limitations**| Needs structured spatially related data (e.g., residential blocks); may underperform on irregular topologies.                                                 |
| **Future Work**| Apply to smart grid control; integrate dynamic spatial graphs; extend to commercial load forecasting.                                                        |

## 21. Are Transformers Effective for Time Series Forecasting? (LTSF-Linear)

| Section        | Details                                                                                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Are Transformers Effective for Time Series Forecasting?                                                                                                       |
| **Motivation** | Growing use of Transformers for long-term forecasting lacks critical validation; authors question their temporal modeling capability.                         |
| **Problem**    | Existing Transformers may lose crucial temporal information due to permutation-invariant self-attention and overhyped complexity.                             |
| **Solution**   | Proposes LTSF-Linear: simple one-layer direct multi-step linear models (Vanilla, DLinear, NLinear) as strong baselines.                                      |
| **Approach**   | Compare LTSF-Linear with state-of-the-art Transformers on 9 benchmarks. Also evaluate positional encoding, token embedding, and error accumulation.          |
| **Pros**       | Linear models outperform Transformers on all benchmarks (by 20–50%); reveal overfitting and lack of true temporal extraction in Transformers.                |
| **Cons**       | LTSF-Linear lacks capacity for nonlinear dependencies and ignores spatial correlations; mainly limited to smooth or periodic data.                          |
| **Limitations**| Benchmarks may not represent chaotic or very noisy datasets; Transformers might perform better in other time series applications like anomaly detection.     |
| **Future Work**| Suggests revisiting Transformers for time series applications; LTSF-Linear to be used as a new baseline in forecasting research.                             |

## 22. Multi-scale Transformer Pyramid Networks (MTPNet)

| Section        | Details                                                                                                                                                   |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multi-scale Transformer Pyramid Networks for Multivariate Time Series Forecasting (MTPNet)                                                               |
| **Motivation** | Existing methods model temporal dependencies at fixed/exponential scales, limiting their ability to capture arbitrary seasonalities.                     |
| **Problem**    | Prior approaches suffer from fixed-scale modeling and embedding that breaks temporal or spatial structure.                                                |
| **Solution**   | Introduces MTPNet: employs a feature pyramid of Transformers with unconstrained patch sizes and dimension-invariant embeddings to model all scales.       |
| **Approach**   | Seasonal/trend decomposition → DI embedding → patch-wise encoding → multi-scale Transformer pyramid → 1x1 CNN for final prediction.                      |
| **Pros**       | Outperforms SOTA on 9 datasets; flexible scale modeling; excellent on varied resolutions and data with mixed seasonalities.                              |
| **Cons**       | Hierarchical pyramid increases memory; some complexity in tuning patch scale grid; still linear decoder.                                                 |
| **Limitations**| Currently assumes fixed-length sequences; not directly tested on irregular data or streaming inputs.                                                     |
| **Future Work**| Explore continuous-scale modeling; integrate dynamic fusion; extend for streaming and irregularly-sampled time series.                                   |

## 23. iTransformer: Inverted Transformers for Time Series Forecasting

| Section        | Details                                                                                                                                               |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| iTransformer: Inverted Transformers Are Effective for Time Series Forecasting                                                                         |
| **Motivation** | Traditional Transformers mix multivariate data per timestamp, losing inter-series correlation and struggling with long sequences and lookback limits. |
| **Problem**    | Permutation-invariant attention over temporal tokens harms performance; embeddings fuse unrelated variates.                                           |
| **Solution**   | iTransformer inverts tokenization: treat each time series as a token (variate token), apply attention on variates, and FFN over time series.           |
| **Approach**   | Reuses Transformer encoder architecture → per-variate tokenization → attention across variates → FFN for temporal encoding.                          |
| **Pros**       | SoTA performance on 7 datasets; improved generalization; interpretable multivariate attention; compatible with efficient Transformers.                |
| **Cons**       | Lacks a generative decoder; not all Transformer tricks are invertible; may over-normalize short series.                                                |
| **Limitations**| Trained only on fixed time steps; requires full-length time series for tokenization; no explicit seasonal modeling.                                   |
| **Future Work**| Foundation model-style pretraining for multivariate time series; hybrid tokenization (series + patches); better low-resource generalization.           |

## 24. Multi-resolution Time-Series Transformer (MTST)

| Section        | Details                                                                                                                                          |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| Multi-resolution Time-Series Transformer for Long-term Forecasting (MTST)                                                                       |
| **Motivation** | Patch-level tokenization improves time-series Transformers but lacks multi-scale modeling for complex seasonalities.                           |
| **Problem**    | Transformers often operate at fixed patch scales, missing variable frequency patterns. Absolute PE impairs temporal interpretability.           |
| **Solution**   | MTST: Multi-branch Transformer layers with different patch sizes + relative PE to model patterns at diverse frequencies simultaneously.         |
| **Approach**   | Each layer → multiple branches → patch/tokenization → attention with RPE → feature fusion layer → stacked into deep architecture.               |
| **Pros**       | SoTA on all 28 experimental settings; clear interpretability via scale-specific branches; lightweight architecture; uses relative PE effectively. |
| **Cons**       | Slightly higher complexity than single-scale PatchTST; branch design tuning needed.                                                              |
| **Limitations**| Not yet tested for exogenous inputs or spatio-temporal data; assumes fixed-length, uniformly sampled input.                                     |
| **Future Work**| Dynamic branch selection; adaptive patching; extending to streaming/multivariate causal discovery and exogenous input modeling.                 |


## 25. MCformer: Mixed-Channels Transformer

| Section        | Details                                                                                                                                           |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| MCformer: Multivariate Time Series Forecasting with Mixed-Channels Transformer                                                                  |
| **Motivation** | CD strategy suffers from long-term feature distortion; CI strategy loses inter-channel info. Need hybridization for better multivariate modeling. |
| **Problem**    | Trade-off between generalization (CI) and correlation learning (CD); CI may cause inter-channel forgetting.                                       |
| **Solution**   | Introduces "Mixed Channels" strategy: flatten → blend channels in intervals → shared-attention encoding → unflatten for multivariate forecasts.   |
| **Approach**   | RevIN → flatten CI → mix every m channels → patch + project → Transformer encoder with positional encoding → final unflatten + denorm.           |
| **Pros**       | Maintains generalization from CI; restores cross-channel dependency via mixing; achieves SOTA on five real-world MTS datasets.                   |
| **Cons**       | Requires hyperparameter tuning for mixing interval m; assumes stationarity when using RevIN.                                                     |
| **Limitations**| Not explicitly tested for low-channel settings (<10); only encoder-based; fusion granularity fixed.                                               |
| **Future Work**| Adaptive mixed-channel scheduling; integrate graph modeling; extend to spatio-temporal and irregular channel domains.                            |


## 26. DifFormer: Multi-Resolutional Differencing Transformer With Dynamic Ranging for Time Series Analysis

| Section        | Details                                                                                                                                                                  |
|----------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| DifFormer: Multi-Resolutional Differencing Transformer With Dynamic Ranging for Time Series Analysis (TPAMI, 2023)                                                     |
| **Motivation** | Create a universal Transformer backbone for time-series analysis that can generalize across tasks (forecasting, classification, regression) without task-specific bias. |
| **Problem**    | Existing Transformers use fixed seasonal biases and smoothing attention that fail to capture nuanced patterns (outliers, local trends, aperiodicity).                  |
| **Solution**   | DifFormer introduces neural differencing, dynamic ranging, and multi-resolutional modeling to flexibly highlight diverse temporal and spectral patterns.               |
| **Approach**   | Dual-stream architecture (temporal & frequency); temporal stream applies multi-stage differencing, dynamic ranging, lagging; frequency stream uses FFT and attention. |
| **Pros**       | Handles diverse patterns (trend, season, cyclic, outlier); linear time/memory complexity; task-agnostic; strong on forecasting, classification, regression.           |
| **Cons**       | Complexity from two streams; signed attention and dynamic ranging can be costly to tune; lacks detailed ablations per task.                                            |
| **Limitations**| Doesn’t explore real-time inference or streaming data adaptation; relies on patch merging which may not suit all domains.                                              |
| **Future Work**| Add online learning capability; optimize dynamic lagging for irregular sampling; expand interpretability across domains.                                                |

## 27. GTformer: Graph-Based Temporal-Order-Aware Transformer for Long-Term Series Forecasting

| Section        | Details                                                                                                                                                                 |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| GTformer: Graph-Based Temporal-Order-Aware Transformer for Long-Term Series Forecasting (IEEE IoT Journal, 2024)                                                      |
| **Motivation** | Improve long-term forecasting for multivariate time series (MTS) in IoT environments by better modeling inter-series dependencies and strict temporal order.           |
| **Problem**    | Prior models neglect either (a) complex relationships between series or (b) the strict temporal order required in real-world MTS prediction.                           |
| **Solution**   | GTformer integrates graph neural networks (GNNs) for learning inter-series dependencies and a novel Temporal Order Aware (TOA) module for sequential encoding.         |
| **Approach**   | Combines Transformer encoder with: (1) adaptive graph learning for unidirectional and bidirectional series relationships, and (2) TOA via RNN-enhanced position enc.   |
| **Pros**       | Captures heterogeneity & causality between series; handles strict sequential ordering; outperforms 6 SOTA baselines across 8 datasets.                                |
| **Cons**       | Increased complexity due to GNN integration and graph adaptation; RNN-based TOA may limit scalability on very large datasets.                                          |
| **Limitations**| Currently relies on Pearson correlations for initial graph estimation; TOA module assumes fixed sequence lengths and uniform sampling.                                |
| **Future Work**| Explore other graph learning methods (e.g. learned similarity metrics); generalize TOA for irregular-sampled or streaming MTS scenarios.                              |

## 28. MCformer: Multivariate Time Series Forecasting With Mixed-Channels Transformer

| Section        | Details                                                                                                                                                        |
|----------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| MCformer: Multivariate Time Series Forecasting With Mixed-Channels Transformer (IEEE IoT Journal, 2024)                                                       |
| **Motivation** | Address the trade-off between channel-independent (CI) and channel-dependent (CD) strategies in multivariate forecasting by fusing their strengths.           |
| **Problem**    | CI models suffer from inter-channel correlation forgetting; CD models disrupt long-term feature extraction by overmixing channels.                            |
| **Solution**   | Proposes a "Mixed-Channels" strategy that integrates limited cross-channel fusion within a CI framework.                                                      |
| **Approach**   | Flattens multivariate series via CI → mixes channel slices at intervals → applies patching, projection, and vanilla Transformer encoder → reverses flattening. |
| **Pros**       | Retains CI’s generalization while restoring key inter-channel interactions; improves accuracy and robustness on diverse datasets.                             |
| **Cons**       | Needs tuning of mixing interval `m`; patch projection adds complexity; assumes availability of moderate channel count.                                       |
| **Limitations**| Not tested on ultra-high-channel or multimodal MTS; lacks spatio-temporal adaptation; attention is standard (not sparse or frequency-aware).                 |
| **Future Work**| Integrate adaptive mixing and GNN-based structure; apply to irregular and streaming MTS; hybrid decoder with temporal graphs.                                |

## 29. TFEformer: Temporal Feature Enhanced Transformer for Multivariate Time Series Forecasting

| Section        | Details                                                                                                                                                               |
|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Paper Title**| TFEformer: Temporal Feature Enhanced Transformer for Multivariate Time Series Forecasting (IEEE Access, 2024)                                                         |
| **Motivation** | Strengthen Transformers' ability to capture multivariate temporal dependencies and align misaligned variates in real-world MTS data.                                |
| **Problem**    | Time-point and series-wise tokenizations either miss deep dependencies or lose fine-grained temporal info; vanilla FFNs extract weak temporal features.             |
| **Solution**   | TFEformer fuses patch-wise and series-wise tokenizations; adds variate-wise attention, adaptive multi-scale fusion, and enhanced temporal FFN modules.              |
| **Approach**   | Encoder-only model using hybrid tokenization → Patch-Series Attention → Adaptive Fusion Unit (AFU) → deep FFN → lightweight linear decoder.                        |
| **Pros**       | SOTA performance on 8 datasets; robust under distribution shifts; generalizes well across lookback lengths and variate counts.                                      |
| **Cons**       | Adds architectural complexity vs. DLinear; patch size and token fusion must be tuned; no decoder customization.                                                      |
| **Limitations**| Fixed tokenization pipeline; no support for online adaptation or irregular time series; variate-wise attention assumes static topology.                             |
| **Future Work**| Integrate dynamic patching and adaptive series selection; apply to multi-modal tasks; fuse with graph attention for spatial dependencies.                          |

## 

# ⚙️ Computational Complexity of TAPFormer

TAPFormer is designed to balance **temporal modeling power** with **computational efficiency**, especially for
long-range forecasting. It leverages a **decoder-only architecture** and introduces efficient mechanisms without
significantly increasing cost.

---

### 🔧 Key Components and Their Complexity

| Module                      | Complexity         | Description                                                                 |
|-----------------------------|--------------------|-----------------------------------------------------------------------------|
| Self-Attention              | `O(T² · d)`        | Default full attention. Can be reduced to `O(T · W · d)` with local window `W`. |
| Soft Local Bias             | `O(T · W)`         | Learnable bias mask applied to attention logits — lightweight.              |
| Gated Local/Global Mixing   | `O(T · d)`         | Gating vector per head/layer — a simple sigmoid-based operation.            |
| Feed-Forward Network (FFN)  | `O(T · d²)`        | Standard transformer MLP — no change from baseline.                         |
| Temporal-Proj Encoding      | `O(T · d)`         | Learnable MLP applied to raw time features per timestep.                    |
| Encoder Stack               | ❌ Not used         | TAPFormer omits the encoder entirely — saves `O(L_enc · T² · d)`.           |

> **T** = sequence length,  
> **d** = embedding dimension,  
> **W** = local window size

---

### 🆚 Comparison with Existing Models

| Model         | Attention Type        | Positional Encoding | Overall Complexity          |
|---------------|------------------------|----------------------|------------------------------|
| Transformer   | Full                  | Sinusoidal           | `O(T² · d)`                  |
| Informer      | ProbSparse            | Learnable            | `O(T log T · d)`             |
| Crossformer   | Patch-based           | Implicit             | `O(T · d)`                   |
| Autoformer    | Series Decomposition  | Sinusoidal           | `O(T log T · d)`             |
| **TAPFormer** | Local + Global Hybrid | Temporal-Proj        | `O(T · W · d)` to `O(T² · d)` (configurable)

---

### 💡 Key Insights

- TAPFormer can operate in **linear time** (`O(T · W · d)`) with local attention, or default to full attention when
  needed.
- **No encoder stack** significantly reduces model depth and computation.
- All additional modules — **Temporal-Proj**, **gating**, and **bias** — are **parameter-light** and add negligible
  compute overhead.
- Suitable for long sequence lengths where traditional attention becomes costly.

---

### ✅ Summary

> TAPFormer achieves a practical trade-off between **expressivity** and **efficiency**. Its design scales well for long
sequences by leveraging configurable attention, efficient temporal encoding, and a streamlined decoder-only
architecture.
