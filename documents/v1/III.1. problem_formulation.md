# 📐 Problem Formulation

Multivariate time series forecasting (MTSF) is the task of predicting future values for multiple correlated variables
based on their historical observations. This problem is fundamental in IoT systems, where sensors continuously generate
high-dimensional time-dependent data streams that exhibit complex patterns across multiple temporal and spatial scales.

Formally, let  
$
\mathbf{X} = [x_1, x_2, \dots, x_L] \in \mathbb{R}^{L \times N}
$  
denote a multivariate time series input of length \(L\), where \(N\) is the number of variates (or sensor channels), and
\(x_t \in \mathbb{R}^N\) represents the measurements at time step \(t\). The forecasting objective is to predict the
next \(T\) time steps:

$
\mathbf{Y} = [x_{L+1}, x_{L+2}, \dots, x_{L+T}] \in \mathbb{R}^{T \times N}
$

Given the input \(\mathbf{X}\), the goal is to learn a function \(F_\theta\) parameterized by \(\theta\), such that the
model yields predictions:

$
\hat{\mathbf{Y}} = F_\theta(\mathbf{X}, \mathbf{T}_{\text{proj}})
$

where \(\mathbf{T}_{\text{proj}} \in \mathbb{R}^{T \times d_t}\) is a **temporal metadata tensor** containing structured
time information (e.g., hour of day, day of week, month of year) that is projected into the latent space using our
proposed **Temporal-Proj** module.

Unlike autoregressive or step-wise forecasting approaches, **TAPFormer** performs **horizon-level forecasting** —
predicting all future steps in parallel — via a decoder-only Transformer, enabling cross-step refinement and faster
inference.

To train the model, we minimize the **Mean Squared Error (MSE)** over a training dataset \(\mathcal{D} = \{(
\mathbf{X}^{(i)}, \mathbf{Y}^{(i)})\}_{i=1}^M\):

$
\mathcal{L}_{\text{MSE}} = \frac{1}{M} \sum_{i=1}^M \left\| F_\theta(\mathbf{X}^{(i)}, \mathbf{T}_{\text{proj}}^{(i)}) -
\mathbf{Y}^{(i)} \right\|_2^2
$

This formulation allows the model to exploit both long-range temporal dependencies and contextual semantics embedded in
time-aware encodings, providing a solid foundation for scalable, accurate forecasting in real-world IoT settings.
