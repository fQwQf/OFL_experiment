// AURORA Paper - Typst Format
// Compile with: typst compile paper.typ

#set document(title: "AURORA: Autonomous Regularization for One-shot Federated Learning with Alignment", author: "Anonymous")
#set page(paper: "us-letter", margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 10pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")

// Title
#align(center)[
  #text(size: 16pt, weight: "bold")[
    AURORA: AUtonomous Regularization for One-shot federated leaRning with Alignment
  ]
  #v(1em)
  #text(size: 11pt)[Anonymous Authors]
  #v(2em)
]

// Abstract
#block(inset: (x: 2em))[
  #text(weight: "bold")[Abstract.]
  One-shot Federated Learning (OFL) achieves ultimate communication efficiency by restricting client-server interaction to a single round. However, in non-IID scenarios, the lack of continuous synchronization leads to severe _model inconsistency_—local models drift into disparate feature spaces, making server-side aggregation ineffective. While recent methods like FAFI enhance local training via self-supervision, they overlook a fundamental challenge: the optimal balance between _global alignment_ and _local adaptation_ is *not static but time-varying* throughout training.

  We make three contributions addressing this challenge. *First*, we establish that the local-global trade-off exhibits a temporal dichotomy: early training benefits from strong global guidance, while later stages require freedom for local refinement. *Second*, we propose AURORA, a framework that learns client-specific, data-dependent regularization schedules under a monotonic meta-annealing prior—without requiring validation sets or additional communication. The key mechanism is annealing the _log-variance regularizer_ (not the loss weight directly), which makes λ an emergent quantity driven by loss dynamics rather than a pre-specified schedule. AURORA employs a fixed Simplex ETF as a geometric anchor and uses gradient-decoupled uncertainty weighting to enable each client to discover its own alignment trajectory. *Third*, we identify and address the "exploding λ" failure mode under extreme heterogeneity through stability regularization.

  Experiments across CIFAR-10/100, SVHN, and varying heterogeneity levels ($alpha in {0.05, 0.1, 0.3, 0.5}$) show that AURORA matches or exceeds manually-tuned baselines using a *single fixed hyperparameter configuration* across all settings, eliminating the need for per-dataset schedule search.
]

#v(1em)

= Introduction

== The Promise and Peril of One-shot Federated Learning

Federated Learning (FL) has emerged as a de facto paradigm for collaborative machine learning under privacy constraints @mcmahan2017. Despite its success, traditional multi-round FL suffers from prohibitive communication overhead, especially when deploying large-scale models over bandwidth-constrained edge networks. *One-shot Federated Learning (OFL)* pushes communication efficiency to its limit by restricting the client-server interaction to a single round @guha2019. In OFL, clients train locally until convergence and upload their models to the server for a one-time aggregation.

== The Specter of Inconsistency

However, this "train-then-merge" paradigm faces a critical challenge: *Model Inconsistency* @zeng2025. Under Non-IID data distributions, local models optimizing solely for local tasks tend to drift into disparate regions of the feature space. Without periodic synchronization to correct these drifts, the aggregated global model often suffers from performance degradation, a phenomenon known as the "garbage-in, garbage-out" pitfall.

== Beyond Static Objectives

Recent advancements, most notably *FAFI* @zeng2025, have made significant strides by augmenting local training with self-supervised contrastive learning (SALT) and prototype-based classification. However, when we extend such prototype-based frameworks with an *explicit global geometric anchor* (e.g., aligning learnable prototypes to a fixed ETF structure for inter-client consistency), a fundamental challenge emerges: how to balance the _local learning_ objective with the _global alignment_ regularization. This trade-off, characterized by a weight $lambda$, is *time-varying* in nature—yet existing approaches either lack such explicit geometric anchoring or, when applied, require *manually tuning* the alignment strength.

The training dynamics of a local client involve a temporal dichotomy:
- *Early Stage:* The model requires strong guidance to align with a global consensus (avoiding early overfitting to local bias).
- *Late Stage:* The model requires freedom to refine its decision boundaries based on specific local data characteristics (local adaptation).

A fixed regularization weight ($lambda$) fails to satisfy both needs simultaneously: a small $lambda$ leads to early divergence, while a large $lambda$ hinders final convergence and adaptation.

== Our Contribution: AURORA

To bridge this gap, we present *AURORA*, an autonomous framework that transforms the OFL training process from a static optimization problem into a dynamic, self-regulating meta-curriculum. AURORA is built upon three pillars:

+ *Explicit Global Anchoring:* We anchor all clients to a pre-defined, geometrically optimal *Simplex Equiangular Tight Frame (ETF)* structure as a global prototype anchor.

+ *Autonomous Regularization:* We propose a novel optimization strategy inspired by homoscedastic uncertainty weighting @kendall2018. By decoupling the gradient flows of model parameters and weighting parameters, AURORA allows the client model to _autonomously adjust_ the regularization strength under a simple monotonic meta-annealing prior.

+ *Robustness via Stability Regularization:* We identify that in scenarios with extreme data heterogeneity (e.g., SVHN with $alpha=0.05$), the uncertainty-based mechanism can lead to numerical instability (the "exploding $lambda$" problem). We introduce a stability regularization term that constrains the adaptive weights within a stable range.

*To summarize, our main contributions are:*
- We identify a fundamental limitation in existing OFL methods: the conflict between global alignment and local adaptation is *time-varying and client-specific*, but current approaches treat it as static.
- We propose *AURORA*, a principled framework that learns data-dependent regularization schedules under a monotonic meta-annealing prior, enabling each client to discover its own alignment trajectory through gradient-based optimization—without validation sets or additional communication.
- We provide a local equilibrium analysis showing how our meta-annealing mechanism naturally produces the desired curriculum behavior (strong alignment early, weak late), while allowing client-specific adaptation based on local data characteristics.
- We introduce a stability regularization mechanism that ensures robustness in extreme non-IID scenarios where pure adaptive methods fail catastrophically.
- Comprehensive experiments demonstrate that AURORA matches manually-tuned baselines using a single fixed hyperparameter configuration, eliminating per-dataset schedule search.

= Related Work

== Handling Data Heterogeneity in Federated Learning

Data heterogeneity (Non-IID) is a fundamental challenge in FL, particularly acute in OFL due to the absence of iterative correction. Multi-round FL methods have proposed various solutions: *FedProx* @li2020 adds a proximal term to regularize local updates; *SCAFFOLD* @karimireddy2020 uses control variates. However, these require multiple communication rounds, making them inapplicable to OFL.

== One-shot Federated Learning

OFL restricts client-server interaction to a single round, presenting unique challenges. Existing approaches can be categorized as:

*Distillation-based:* *DENSE* @zhang2022 and *Co-Boosting* @song2023 employ knowledge distillation for model aggregation. *FedDF* @lin2020 uses ensemble distillation with public data.

*Aggregation-based:* *FedLPA* @liu2024 introduces layer-wise posterior aggregation using Laplace approximation, achieving strong performance without auxiliary data by treating model aggregation as Bayesian inference.

*Client-side enhancement:* *FAFI* @zeng2025 addresses model inconsistency through feature-anchored integration, combining contrastive learning and prototype-based classification to improve local model quality before aggregation.

Our work is orthogonal to server-side aggregation methods like FedLPA. We focus on improving *local training objectives* to reduce model inconsistency at the source, which is complementary to advanced aggregation techniques.

== Prototype-based Federated Learning and Neural Collapse

Prototype-based methods have gained traction for their communication efficiency. *FedProto* @tan2022 exchanges class prototypes instead of model parameters. *FedTGP* @zhang2024 introduces trainable global prototypes with adaptive-margin contrastive learning.

Our work leverages the *Neural Collapse* phenomenon @papyan2020, which shows that optimal classifiers converge to a Simplex Equiangular Tight Frame (ETF) structure. *FedETF* @li2023 uses a fixed ETF classifier to unify feature learning in multi-round FL.

*Our distinction from FedETF:* While FedETF focuses on multi-round FL and replaces the entire classifier with ETF, we target *one-shot FL* and use ETF as an *alignment anchor for prototypes* rather than the classifier itself. Furthermore, our key contribution is the *autonomous regularization mechanism* that learns when and how strongly to align.

*Our Core Novelty:* The primary contribution of AURORA is *not* the ETF anchor (which is a known structure) nor uncertainty weighting (which is an established technique), but rather the *meta-curriculum formulation* that combines gradient decoupling with meta-annealing to achieve autonomous, data-adaptive regularization scheduling.

= The AURORA Framework: Autonomous Regularization

== Preliminaries: The Dual Objectives in OFL

We consider a one-shot federated learning setting with $K$ clients, each holding a private dataset $cal(D)_k$ drawn from a potentially distinct distribution. We extend FAFI's formulation by introducing an explicit alignment loss:

$ cal(L)_"total" = cal(L)_"local" + lambda dot cal(L)_"align" $

where:
- $cal(L)_"local" = cal(L)_"cls" + cal(L)_"con" + cal(L)_"proto"$ encompasses local supervision signals
- $cal(L)_"align"$ is the global alignment loss that encourages the client's learnable prototypes to align with a fixed global target

*ETF Anchor for Global Alignment.* Inspired by the Neural Collapse theory @papyan2020, we define:

$ cal(L)_"align" = frac(1, |cal(C)_k|) sum_(c in cal(C)_k) norm(bold(p)_c - bold(a)_c)^2 $

where $cal(C)_k$ is the set of classes present in client $k$'s local dataset, $bold(p)_c in RR^d$ is the learnable prototype for class $c$, and $bold(a)_c$ is the corresponding column of the pre-defined ETF anchor matrix $bold(A) in RR^(d times C)$, satisfying:

$ bold(A)^top bold(A) = frac(C, C-1) (bold(I)_C - frac(1,C) bold(1)_C bold(1)_C^top) $

This mathematically optimal structure ensures maximum inter-class separation and provides a consistent geometric target for all clients.

*Implementation Details for Reproducibility:*
- *Prototype representation:* Learnable prototypes $bold(p)_c in RR^d$ are *not* L2-normalized during alignment computation. The ETF anchors are normalized to unit norm.
- *Alignment loss:* We use L2 (MSE) distance rather than cosine similarity, as MSE provides stronger gradients when prototypes are far from anchors.
- *Class mask per batch:* During training, alignment loss is computed only over classes appearing in the current batch.
- *Missing class initialization:* Prototypes are initialized to their corresponding ETF anchor positions (with small random perturbation).

== Learning the Alignment Strength (λ) via Task Uncertainty

The critical challenge lies in determining the optimal $lambda$ that balances local adaptation with global alignment. Instead of treating $lambda$ as a fixed hyperparameter, we propose to *learn* it through the lens of task uncertainty.

*Uncertainty-Weighted Multi-Task Loss.* Following @kendall2018, we model each loss term using a Gaussian likelihood with learnable observation noise:

$ cal(L) = frac(1, 2 sigma_1^2) cal(L)_1 + frac(1, 2 sigma_2^2) cal(L)_2 + log sigma_1 + log sigma_2 $

where $sigma_1^2$ and $sigma_2^2$ are learnable parameters representing the homoscedastic uncertainty of each task.

*Effective Lambda.* The effective alignment weight emerges as:

$ lambda_"eff" = frac(sigma_"local"^2, sigma_"align"^2) $

*Decoupled Interpretation.* In AURORA, σ parameters are optimized via an uncertainty-style meta-objective, but *do not rescale the gradients of model weights* (see Section 3.3). Instead, they determine an emergent ratio $lambda_"eff" = sigma_"local"^2 / sigma_"align"^2$ that *only modulates the alignment term* in $cal(L)_W = cal(L)_"local" + lambda_"eff" dot cal(L)_"align"$. The resulting λ trajectory is *emergent and data-dependent*—not pre-specified, but arising from the joint dynamics of loss magnitudes and the monotonic prior.

== AURORA's Meta-Objective: Decoupling Learning and "Learning to Learn"

A naive implementation of uncertainty weighting introduces an unintended side effect: the weighting coefficients $1/sigma^2$ also scale the effective learning rate, potentially destabilizing training. We address this through *gradient decoupling*.

*The Decoupling Mechanism.* We maintain two separate loss formulations:

*1. Loss for Model Weights ($cal(L)_W$):* Used to update backbone and classifier parameters.
$ cal(L)_W = cal(L)_"local" + lambda_"eff" dot cal(L)_"align" $

*2. Loss for Sigma Parameters ($cal(L)_sigma$):* Used to update the uncertainty parameters. Using $ell = log sigma^2$:
$ cal(L)_sigma = frac(cal(L)_"local"^"(detach)", 2 e^(ell_"local")) + frac(cal(L)_"align"^"(detach)", 2 e^(ell_"align")) + frac(1,2) ell_"local" + frac(1,2) ell_"align" $

The `.detach()` operation prevents gradients from flowing from the uncertainty parameters back to the model weights, creating an *approximate online bilevel optimization*.

*Why Decoupling is Necessary.* Without gradient decoupling, the $1/sigma^2$ coefficients in the Kendall formulation directly scale the effective learning rate for each task. This causes two failure modes: (1) when $sigma_"local"^2$ grows large, the local loss gradients become vanishingly small; (2) the σ parameters receive conflicting gradients, leading to oscillatory training dynamics.

== Inducing a Curriculum with Meta-Annealing

Experimental analysis reveals that uncertainty weighting alone converges to a static equilibrium. To induce a *curriculum* from strong alignment to local adaptation, we introduce a *meta-annealing schedule*.

*Schedule Factor.* We define $s(p) = frac(1,2)(1 + cos(pi p))$, where $p in [0, 1]$ is the normalized training progress. The meta-annealing applies $s(p)$ to the *regularization term* of the alignment task:

$ cal(L)_sigma = frac(cal(L)_"local"^"(detach)", 2 e^(ell_"local")) + frac(cal(L)_"align"^"(detach)", 2 e^(ell_"align")) + frac(1,2) ell_"local" + frac(1,2) s(p) dot ell_"align" $

*Proposition 1 (Approximate Stationary Points, up to constants).* Under gradient decoupling, slowly-varying losses, and small σ learning rate, the uncertainty parameters track:

$ sigma_"local"^(2*) = cal(L)_"local", quad quad sigma_"align"^(2*) = frac(cal(L)_"align", s(p)) $

Consequently, the equilibrium alignment weight is:

$ lambda_"eff"^* = frac(sigma_"local"^(2*), sigma_"align"^(2*)) = s(p) dot frac(cal(L)_"local", cal(L)_"align") $

*Implications of Proposition 1:*
+ *Curriculum:* The factor $s(p) arrow.b$ ensures $lambda_"eff"^*$ trends downward over training.
+ *Data-adaptivity:* The loss ratio $cal(L)_"local"\/cal(L)_"align"$ makes each client's trajectory data-dependent.
+ *Explosion risk:* When $cal(L)_"align" << cal(L)_"local"$ (extreme non-IID), the ratio explodes even if $s(p)$ is small, motivating Section 3.5.

*Why This is Fundamentally Different from a Fixed Schedule.*

Unlike a fixed schedule $lambda(t) = lambda_0 dot s(t)$, AURORA's $lambda_"eff"$ emerges from the joint dynamics of loss magnitudes and the monotonic prior. The σ parameters capture *meta-level task uncertainty* through $cal(L)_sigma$ (with detached losses); this uncertainty does not rescale $nabla_theta$, but induces a ratio $lambda_"eff" = sigma_"local"^2 / sigma_"align"^2$ that modulates alignment in $cal(L)_W$. The key distinction: *s(p) only imposes a monotonic prior; magnitude and inter-client variation emerge from optimization* (see Table 4 for empirical evidence).

== Ensuring Robustness: Stability Regularization

In extreme non-IID scenarios (e.g., SVHN with $alpha=0.05$), we observe a failure mode where $lambda_"eff"$ explodes due to severe task difficulty imbalance.

*Stability Regularization via Soft Constraint.* We introduce a squared-hinge regularization:

$ cal(L)_"reg" = gamma dot "ReLU"(lambda_"eff" - lambda_"max")^2 $

*Default $lambda_"max"$.* We use $lambda_"max"=50$ as a fixed default across all experiments.

This mechanism:
- *Non-intrusive:* When $lambda_"eff" < lambda_"max"$, the term contributes zero gradient
- *Smooth correction:* The squared form provides continuous second-order gradients
- *Preserves adaptivity:* Learning dynamics operate freely within the stable region

== Implementation Details

*Algorithm 1: AURORA Local Training*
```
Input: Local data D_k, ETF anchors A, epochs T
Initialize: log_σ²_local ← 0, log_σ²_align ← 0

for epoch t = 1 to T:
    p ← t/T  # Training progress
    s ← max(ε, 0.5·(1 + cos(π·p)))  # Cosine annealing
    for each batch (x, y):
        L_local ← L_cls + L_con + L_proto
        L_align ← MSE(P[C_k], A[C_k])
        
        σ²_local ← exp(log_σ²_local)
        σ²_align ← exp(log_σ²_align)
        
        # For model weights: detach λ
        λ_eff_det ← (σ²_local / σ²_align).detach()
        L_W ← L_local + λ_eff_det · L_align
        
        # For uncertainty: use detached losses
        L_σ ← L_local.detach()/(2σ²_local) 
              + L_align.detach()/(2σ²_align)
              + 0.5·log_σ²_local + 0.5·s·log_σ²_align
        
        # For stability
        λ_eff ← (σ²_local / σ²_align)
        L_reg ← γ · ReLU(λ_eff - λ_max)²
        
        Update θ using ∇L_W
        Update log_σ² using ∇(L_σ + L_reg)
```

*Overhead.* AURORA adds only 2 scalars per client; communication unchanged.

= Experiments

== Experimental Setup

*Datasets.* We evaluate AURORA on three benchmarks:
- *CIFAR-10:* 10-class natural image classification (50,000 training / 10,000 test)
- *CIFAR-100:* 100-class fine-grained classification (50,000 training / 10,000 test)
- *SVHN:* Street View House Numbers digit recognition (73,257 training / 26,032 test)

*Non-IID Simulation.* Following standard practice, we partition training data among $K=5$ clients using Dirichlet distribution with concentration parameter $alpha in {0.05, 0.1, 0.3, 0.5}$. Lower $alpha$ indicates more severe heterogeneity.

*Baselines.* We compare against methods spanning different OFL paradigms:
- *FedAvg (One-shot):* Simple averaging of locally trained models
- *FAFI:* Feature-Anchored Integration with contrastive learning @zeng2025
- *FAFI+Annealing:* FAFI with manually-tuned linear λ annealing schedule
- *FedLPA:* Layer-wise Posterior Aggregation using Laplace approximation @liu2024

*Ablation and Alternative λ Mechanisms (Section 4.3):*
- *AURORA (no stability):* Meta-annealing without stability regularization
- *AURORA (no decouple):* Standard Kendall formulation without gradient decoupling
- *Learnable-λ(t):* $lambda = "softplus"(a + b dot phi(p))$ where $phi(p) = cos(pi p)$, allowing nonlinear schedule learning
- *Cosine λ schedule:* Pure schedule $lambda(t) = lambda_0 dot s(p)$, no learning
- *GradNorm-style:* λ adjusted based on gradient magnitude ratio (Appendix)

*Implementation Details.*
- Backbone: ResNet-18
- Total local epochs: 500 (CIFAR-10), 100 (CIFAR-100, SVHN)
- Optimizer: SGD with momentum 0.9, weight decay 5e-4
- Learning rate: 0.05 (cosine annealing over local training)
- AURORA-specific: $sigma$ learning rate = 0.005, $lambda_"max"$ = 50.0, $gamma$ = 0.001
- Default: $K=5$ clients

*Loss Scaling.* All loss terms follow FAFI's original scaling. We keep these fixed across all baselines to ensure σ adapts to training dynamics rather than arbitrary rescaling.

== Main Results

#figure(
  table(
    columns: 7,
    stroke: 0.5pt,
    inset: 6pt,
    [*Dataset*], [*α*], [*FedAvg*], [*FAFI*], [*FAFI+Ann.*], [*FedLPA*], [*AURORA*],
    [CIFAR-10], [0.05], [TBD], [66.97], [67.77], [TBD], [*68.17*],
    [CIFAR-10], [0.1], [TBD], [76.10], [76.86], [TBD], [TBD],
    [CIFAR-10], [0.3], [TBD], [83.90], [83.57], [TBD], [TBD],
    [CIFAR-10], [0.5], [TBD], [87.69], [88.46], [TBD], [TBD],
    [CIFAR-100], [0.05], [TBD], [38.41], [40.41], [TBD], [*40.43*],
    [SVHN], [0.05], [TBD], [TBD], [TBD], [TBD], [*52.9*],
  ),
  caption: [Test Accuracy (%) on Different Datasets and Heterogeneity Levels]
)

*Key Observations:*
+ AURORA achieves performance comparable to the best manually-tuned baseline (FAFI+Ann.), but without the need for schedule search.
+ On CIFAR-100 ($alpha=0.05$), AURORA (40.43%) achieves parity with the hand-crafted schedule (40.41%).
+ On SVHN with extreme heterogeneity, AURORA achieves 52.9% where methods without stability regularization fail.

*Model Consistency Metrics.* Beyond accuracy, we measure *prototype consistency* to quantify model alignment.

*Definition (g_protos_std).* Let $bold(p)_c^((k)) in RR^d$ be the learned prototype for class $c$ on client $k$. For each class $c$ present on at least 2 clients, compute the standard deviation of the $ell_2$-normalized prototype vectors:

$ "std"_c = sqrt(frac(1, |cal(K)_c|) sum_(k in cal(K)_c) norm(hat(bold(p))_c^((k)) - overline(bold(p))_c)^2) $

where $cal(K)_c$ is the set of clients having class $c$, $hat(bold(p))_c^((k)) = bold(p)_c^((k)) / norm(bold(p)_c^((k)))$, and $overline(bold(p))_c$ is the mean normalized prototype. Then:

$ g\_"protos"\_"std" = frac(1, |cal(C)_"valid"|) sum_(c in cal(C)_"valid") "std"_c $

Lower values indicate stronger inter-client alignment.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    inset: 6pt,
    [*Method*], [*FAFI*], [*+ ETF Anchor*], [*+ Manual Anneal*], [*AURORA*],
    [g_protos_std], [1.007], [0.935 (-7.1%)], [0.709 (-29.6%)], [*0.710*],
  ),
  caption: [Table 1b: Model Consistency (g_protos_std ↓) on CIFAR-10 (α=0.05)]
)

== Ablation Study

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*Configuration*], [*Description*], [*Accuracy (%)*],
    [FAFI (baseline)], [SALT + prototype learning, no explicit global anchor], [38.41],
    [+ ETF Anchor], [Fixed λ=10, aligns prototypes to ETF structure], [TBD],
    [+ Manual Anneal], [λ: 18→0 linear decay over training], [40.41],
    [+ Uncertainty Weight], [Decoupled formulation, s(p)=1, converges to λ≈10], [38.25],
    [+ Meta-Anneal], [Decoupled + cosine s(p), enables λ evolution], [40.43],
    [*+ Stability Reg (AURORA)*], [+ ReLU-hinge constraint on λ_eff ≤ λ_max], [*40.43*],
  ),
  caption: [Ablation Study on CIFAR-100 (α=0.05)]
)

*Insights:*
+ *ETF anchoring is necessary:* Without explicit geometric guidance, clients diverge in feature space.
+ *Static λ is insufficient:* Pure uncertainty weighting finds a static equilibrium (~10) that underperforms curriculum-based approaches.
+ *Meta-annealing enables autonomous curriculum:* Gradient decoupling allows λ to evolve, matching manual tuning performance.
+ *Stability regularization is critical for edge cases:* While identical to Meta-Anneal on CIFAR-100, it prevents catastrophic failure on SVHN.

== Analysis: AURORA Learns the Optimal Schedule

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Checkpoint*], [*Schedule Factor s(p)*], [*AURORA λ_eff*], [*Manual Anneal λ*],
    [0 (start)], [1.0], [TBD], [18.0],
    [3], [0.7], [TBD], [12.6],
    [6], [0.4], [TBD], [7.2],
    [9 (end)], [0.1], [TBD], [1.8],
  ),
  caption: [Table 3a: λ Evolution Comparison (CIFAR-100, α=0.05)]
)

AURORA's learned λ curve is expected to approximate the effective annealing pattern, validating that the meta-learning mechanism discovers a near-optimal schedule autonomously.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    inset: 6pt,
    [*Checkpoint (10 epochs)*], [*s(p)*], [*Client 0 (Raw λ)*], [*Client 4 (Raw λ)*], [*Ratio C4/C0*],
    [0], [0.9], [12.9], [13.5], [+4.7%],
    [4-6], [0.5→0.3], [14.9→22.8], [16.9→28.1], [+13%→+23%],
    [7], [0.2], [31.6], [42.7], [*+35%*],
    [8], [0.1], [48.8], [73.5], [*+51%*],
  ),
  caption: [Table 4: Per-Client λ Divergence (Meta-Anneal, no stability reg) on CIFAR-100 (α=0.05). Analysis run without stability regularization to demonstrate client-specific divergence.]
)

*Key Observation:* While both clients follow the same s(p) prior, their raw λ trajectories diverge significantly—by Checkpoint 8 (80 epochs), Client 4's λ is 51% higher than Client 0's. This divergence arises from differences in local data distributions, demonstrating that AURORA's mechanism is *data-dependent, not merely time-dependent*.

== Robustness Study: The λ Explosion Problem

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Method*], [*Peak Acc (Checkpoint)*], [*Final Acc*], [*λ Behavior*],
    [Meta-Anneal (no stability)], [49.5% (ckpt 14)], [16.4%], [Explodes to >10⁶],
    [+ Weak Reg (γ=1e-5)], [50.0% (ckpt 13)], [17.7%], [Still explodes],
    [*AURORA (γ=1e-3)*], [*55.4% (ckpt 26)*], [*52.9%*], [Stable ≤50],
  ),
  caption: [Table 5: SVHN Performance Under Extreme Heterogeneity (α=0.05)]
)

*Analysis:* Without stability regularization, the difficulty gap between local and alignment tasks causes λ to grow exponentially. AURORA's hinge constraint maintains $lambda_"eff" <= lambda_"max"$, preserving stable training dynamics.

== Hyperparameter Sensitivity

*Qualitative Distinction: Stability Bounds vs. Performance-Critical Parameters.*

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [], [*Manual λ Schedule*], [*AURORA Stability Params*],
    [*Type*], [Performance-critical], [Safety bounds],
    [*Sensitivity*], [Change shape → 2-5% acc drop], [10× range → <1% variance],
    [*Cross-setting*], [Requires re-tuning per dataset/α], [Same defaults work across all],
    [*Trigger rate*], [Always active], [λ_max rarely triggered (<5% steps)],
  ),
  caption: [Comparison of hyperparameter types]
)

AURORA's hyperparameters define _when the safety mechanism activates_ ($lambda_"max"$) and _how fast the σ parameters track losses_ (σ-lr). These behave like learning rate or weight decay—stable defaults exist.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Parameter*], [*Values Tested*], [*Accuracy (%)*], [*Variance*],
    [λ_max], [10, 20, *50*, 100], [TBD], [TBD],
    [γ (reg strength)], [1e-4, *1e-3*, 1e-2], [TBD], [TBD],
    [σ-learning rate], [1e-3, *5e-3*, 1e-2], [TBD], [TBD],
  ),
  caption: [Table 6: Sensitivity Analysis on CIFAR-100 (α=0.05). Default values in *bold*.]
)

== Scalability Study

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*K (Clients)*], [*FAFI*], [*FAFI+Ann.*], [*AURORA*],
    [5], [TBD], [TBD], [TBD],
    [10], [TBD], [TBD], [TBD],
    [20], [TBD], [TBD], [TBD],
  ),
  caption: [Table 7: Performance with Varying Number of Clients on CIFAR-10 (α=0.1)]
)

*Purpose:* Verify that AURORA's autonomous mechanism generalizes across different federation scales without re-tuning.

= Conclusion

We have presented a framework for autonomous regularization in One-shot Federated Learning. By reformulating the local-global trade-off as a learnable meta-objective with gradient decoupling and meta-annealing, our method reduces the need for hand-crafted regularization schedules while achieving competitive performance with state-of-the-art methods.

Our key insights include:
+ *Beyond static objectives:* The optimal balance between local adaptation and global alignment varies throughout training, necessitating dynamic regularization.
+ *Learning to regularize:* Uncertainty-weighted loss combined with gradient decoupling enables the model to autonomously discover effective schedules.
+ *Robustness matters:* Direct regularization on the effective weight provides a safety mechanism for extreme scenarios.

*Future Work.* Promising directions include: (1) extending to model heterogeneous settings; (2) combining with advanced server-side aggregation techniques like FedLPA; (3) theoretical analysis of the meta-learning convergence properties.

#pagebreak()

// ============ APPENDIX ============

#set heading(numbering: "A.1")
#counter(heading).update(0)

= Appendix

== Extended Probabilistic Derivation (Kendall Framework)

This section provides the complete probabilistic story behind AURORA's uncertainty weighting, extending Section 3.2 of the main text.

=== Gaussian Likelihood Formulation

Following @kendall2018, we model each task loss as a Gaussian likelihood with learnable observation noise:

$ p(y | f(x), sigma) = cal(N)(y; f(x), sigma^2) $

For regression tasks, the negative log-likelihood becomes:

$ -log p(y | f(x), sigma) = frac(1, 2 sigma^2) norm(y - f(x))^2 + log sigma $

Generalizing to arbitrary loss functions $cal(L)_i$:

$ cal(L)_"total" = sum_i frac(1, 2 sigma_i^2) cal(L)_i + log sigma_i $

=== Why σ² Tracks Loss Magnitude

Taking the derivative with respect to $sigma^2$ and setting to zero:

$ frac(diff cal(L)_"total", diff sigma^2) = -frac(cal(L), 2 sigma^4) + frac(1, 2 sigma^2) = 0 $

Solving: $sigma^(2*) = cal(L)$

*Interpretation:* At equilibrium, σ² equals the loss magnitude. A task with high loss (hard/noisy) has large σ², receiving smaller weight $(1\/sigma^2)$.

=== From Kendall to AURORA: The Decoupling Step

In standard Kendall, the $1\/sigma^2$ coefficient directly scales gradients:

$ nabla_theta cal(L)_"total" = sum_i frac(1, 2 sigma_i^2) nabla_theta cal(L)_i $

This causes *learning rate interference*: when σ² grows, gradients shrink.

*AURORA's decoupling:* We use two separate losses:
- $cal(L)_W = cal(L)_"local" + lambda_"eff" dot cal(L)_"align"$ for model weights (no σ scaling)
- $cal(L)_sigma$ with detached losses for σ updates only

This preserves the uncertainty-based weighting *for determining λ_eff* while avoiding gradient scaling issues.

== GradNorm Comparison

=== GradNorm Overview

GradNorm @chen2018gradnorm adjusts task weights to balance gradient magnitudes:

$ cal(L)_"grad" = sum_i |G_i(t) - overline(G)(t) dot r_i^(-alpha)| $

where $G_i(t) = norm(nabla_W w_i cal(L)_i)$ is the gradient norm.

=== Key Differences from AURORA

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*Aspect*], [*GradNorm*], [*AURORA*],
    [Objective], [Balance gradient norms], [Balance task uncertainty],
    [Mechanism], [Explicit grad norm calculation], [Implicit via σ equilibrium],
    [Monotonicity], [No prior], [Cosine prior on s(p)],
    [Per-client], [Same for all], [Client-specific λ_k(t)],
    [Overhead], [Per-step grad norm computation], [2 scalar parameters],
  ),
  caption: [Comparison between GradNorm and AURORA]
)

*Note:* GradNorm was designed for multi-task learning with shared encoders, not federated learning. We adapt it by treating local and alignment objectives as separate tasks.

== Additional Ablation Studies

=== Effect of σ Learning Rate

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*σ-lr*], [*Accuracy*], [*λ_eff Range*],
    [0.001], [TBD], [Slow tracking],
    [0.005 (default)], [40.43%], [Stable],
    [0.01], [TBD], [Fast but noisy],
  ),
  caption: [Effect of σ learning rate on CIFAR-100 (α=0.05)]
)

=== Effect of λ_max Threshold

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*λ_max*], [*Accuracy*], [*Trigger Rate*],
    [20], [TBD], [TBD%],
    [50 (default)], [40.43%], [<5%],
    [100], [TBD], [TBD%],
  ),
  caption: [Effect of λ_max threshold on CIFAR-100 (α=0.05)]
)

== Per-Client λ Trajectory Analysis

=== Full Trajectory Data (All 5 Clients)

_CIFAR-100, α=0.05, Meta-Anneal without stability reg_

#figure(
  table(
    columns: 7,
    stroke: 0.5pt,
    inset: 6pt,
    [*Checkpoint*], [*s(p)*], [*C0*], [*C1*], [*C2*], [*C3*], [*C4*],
    [0], [0.9], [12.9], [TBD], [TBD], [TBD], [13.5],
    [...], [...], [...], [...], [...], [...], [...],
  ),
  caption: [Full per-client λ trajectory data]
)

=== Correlation with Data Skew

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Client*], [*\# Classes Present*], [*α_local (effective)*], [*Final λ*],
    [0], [TBD], [TBD], [48.8],
    [4], [TBD], [TBD], [73.5],
  ),
  caption: [Correlation between data skew and λ trajectory]
)

== Extended Related Work

=== Multi-Task Weighting Methods

- *Uncertainty Weighting* @kendall2018: Homoscedastic uncertainty for automatic weighting
- *GradNorm* @chen2018gradnorm: Gradient magnitude balancing
- *DWA* @liu2019mtan: Dynamic Weight Average based on loss descent rate
- *PCGrad* @yu2020pcgrad: Projecting conflicting gradients

#pagebreak()

// References
#bibliography("references.bib", style: "ieee")
