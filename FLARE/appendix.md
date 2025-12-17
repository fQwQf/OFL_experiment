# Appendix

## A. Extended Probabilistic Derivation (Kendall Framework)

This section provides the complete probabilistic story behind AURORA's uncertainty weighting, extending Section 3.2 of the main text.

### A.1 Gaussian Likelihood Formulation

Following [Kendall et al., 2018], we model each task loss as a Gaussian likelihood with learnable observation noise:

$$p(y | f(x), \sigma) = \mathcal{N}(y; f(x), \sigma^2)$$

For regression tasks, the negative log-likelihood becomes:

$$-\log p(y | f(x), \sigma) = \frac{1}{2\sigma^2} \|y - f(x)\|^2 + \log\sigma$$

Generalizing to arbitrary loss functions $\mathcal{L}_i$:

$$\mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2} \mathcal{L}_i + \log\sigma_i$$

### A.2 Why σ² Tracks Loss Magnitude

Taking the derivative with respect to $\sigma^2$ and setting to zero:

$$\frac{\partial \mathcal{L}_{total}}{\partial \sigma^2} = -\frac{\mathcal{L}}{2\sigma^4} + \frac{1}{2\sigma^2} = 0$$

Solving: $\sigma^{2*} = \mathcal{L}$

**Interpretation:** At equilibrium, σ² equals the loss magnitude. A task with high loss (hard/noisy) has large σ², receiving smaller weight $(1/\sigma^2)$.

### A.3 From Kendall to AURORA: The Decoupling Step

In standard Kendall, the $1/\sigma^2$ coefficient directly scales gradients:

$$\nabla_\theta \mathcal{L}_{total} = \sum_i \frac{1}{2\sigma_i^2} \nabla_\theta \mathcal{L}_i$$

This causes **learning rate interference**: when σ² grows, gradients shrink.

**AURORA's decoupling:** We use two separate losses:
- $\mathcal{L}_W = \mathcal{L}_{local} + \lambda_{eff} \cdot \mathcal{L}_{align}$ for model weights (no σ scaling)
- $\mathcal{L}_\sigma$ with detached losses for σ updates only

This preserves the uncertainty-based weighting **for determining λ_eff** while avoiding gradient scaling issues.

---

## B. GradNorm Comparison

### B.1 GradNorm Overview

GradNorm [Chen et al., 2018] adjusts task weights to balance gradient magnitudes:

$$\mathcal{L}_{grad} = \sum_i |G_i(t) - \bar{G}(t) \cdot r_i^{-\alpha}|$$

where $G_i(t) = \|\nabla_W w_i \mathcal{L}_i\|$ is the gradient norm.

### B.2 Key Differences from AURORA

| Aspect | GradNorm | AURORA |
|--------|----------|--------|
| **Objective** | Balance gradient norms | Balance task uncertainty |
| **Mechanism** | Explicit grad norm calculation | Implicit via σ equilibrium |
| **Monotonicity** | No prior | Cosine prior on s(p) |
| **Per-client** | Same for all | Client-specific λ_k(t) |
| **Overhead** | Per-step grad norm computation | 2 scalar parameters |

### B.3 Empirical Comparison (CIFAR-100, α=0.05)

| Method | Accuracy | λ Behavior |
|--------|----------|------------|
| GradNorm | [TBD] | Oscillatory |
| AURORA | 40.43% | Smooth decay |

**Note:** GradNorm was designed for multi-task learning with shared encoders, not federated learning. We adapt it by treating local and alignment objectives as separate tasks.

---

## C. Additional Ablation Studies

### C.1 Effect of σ Learning Rate

| σ-lr | Accuracy | λ_eff Range |
|------|----------|-------------|
| 0.001 | [TBD] | Slow tracking |
| 0.005 (default) | 40.43% | Stable |
| 0.01 | [TBD] | Fast but noisy |

### C.2 Effect of λ_max Threshold

| λ_max | Accuracy | Trigger Rate |
|-------|----------|--------------|
| 20 | [TBD] | [TBD]% |
| 50 (default) | 40.43% | <5% |
| 100 | [TBD] | [TBD]% |

---

## D. Per-Client λ Trajectory Analysis

### D.1 Full Trajectory Data (All 5 Clients)

*CIFAR-100, α=0.05, Meta-Anneal without stability reg*

| Checkpoint | s(p) | C0 | C1 | C2 | C3 | C4 |
|------------|------|-----|-----|-----|-----|-----|
| 0 | 0.9 | 12.9 | [TBD] | [TBD] | [TBD] | 13.5 |
| ... | ... | ... | ... | ... | ... | ... |

### D.2 Correlation with Data Skew

| Client | # Classes Present | α_local (effective) | Final λ |
|--------|-------------------|---------------------|---------|
| 0 | [TBD] | [TBD] | 48.8 |
| 4 | [TBD] | [TBD] | 73.5 |

---

## E. Extended Related Work

### E.1 Multi-Task Weighting Methods

- **Uncertainty Weighting** [Kendall et al., 2018]: Homoscedastic uncertainty for automatic weighting
- **GradNorm** [Chen et al., 2018]: Gradient magnitude balancing
- **DWA** [Liu et al., 2019]: Dynamic Weight Average based on loss descent rate
- **PCGrad** [Yu et al., 2020]: Projecting conflicting gradients

### E.2 OFL Methods

[Extended discussion of FedAvg, FAFI, FedLPA, etc.]

---

## References (Appendix-specific)

- Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing," ICML 2018
- Liu et al., "End-to-End Multi-Task Learning with Attention," CVPR 2019
- Yu et al., "Gradient Surgery for Multi-Task Learning," NeurIPS 2020
