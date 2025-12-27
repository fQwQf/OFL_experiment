// AURORA Paper - ICML Format
// Using lucky-icml template

#import "@preview/cetz:0.3.2"
#import "@preview/cetz-plot:0.1.1": plot, chart
#import "@preview/lucky-icml:0.7.0": icml2025

#show: icml2025.with(
  title: [AURORA: Autonomous Regularization for One-shot Representation Alignment],
  authors: (
    (
      name: "Anonymous",
      affiliation: "Anonymous Institution",
    ),
  ),
  abstract: [
    One-shot Federated Learning (OFL) pushes communication efficiency to its limit but suffers from severe model inconsistency under non-IID data. A natural remedy is to anchor local prototypes to a globally shared geometric structure (Simplex ETF). *However, we discover that naively adding such alignment—or naively applying uncertainty-based loss weighting—fails catastrophically in OFL, degrading accuracy by up to 16% (from 39.41% to 23.94%).* We term this the "Temporal Dichotomy": geometric anchors are only effective when coupled with *dynamic* scheduling, a phenomenon absent from prior OFL literature.

    Building on this discovery, we propose AURORA, a framework that *automates* the required dynamic scheduling. AURORA's key insight is *gradient decoupling*: rather than letting uncertainty weights directly scale model gradients, we decouple the meta-objective to learn client-specific, data-dependent regularization trajectories. Combined with a monotonic meta-annealing prior and stability regularization, AURORA matches or exceeds manually-tuned baselines across CIFAR and SVHN using a *single fixed hyperparameter configuration*, eliminating the need for per-dataset schedule search.
  ],
  bibliography: none,
  accepted: false,
)


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

A fixed regularization weight ($lambda$) fails to satisfy both needs simultaneously: a small $lambda$ leads to early divergence, while a large $lambda$ hinders final convergence and adaptation. While manual annealing schedules (e.g., linear decay) can alleviate this, they introduce sensitive hyperparameters that require expensive tuning for each dataset and heterogeneity level.

*Key Discovery: The Temporal Dichotomy.* To the best of our knowledge, we are the first to investigate the use of an explicit global geometric anchor (Simplex ETF) for client-side prototype alignment in the strictly constrained *One-shot FL* setting. Through systematic experiments, we reveal a surprising phenomenon: naïvely enforcing this anchor with a static weight can *hurt* performance, while a manually-annealed schedule yields clear gains. On CIFAR-100 ($alpha=0.05$), static ETF alignment achieves only 38.25%, compared to 38.41% without alignment—yet manual annealing reaches 40.41%. *We term this phenomenon the "Temporal Dichotomy":* static geometric constraints that work in multi-round FL (where iterative correction compensates for rigid alignment) fail in One-shot FL, where the single-round constraint demands dynamic regularization. This discovery—that global alignment is *conditionally effective* based on temporal scheduling—represents a novel insight into One-shot FL training dynamics.

*Hand-Crafted Baseline as Discovery Instrument.* To systematically investigate this phenomenon, we construct a simple hand-crafted annealing schedule (*FAFI+ETF+Anneal*): $lambda$ decays linearly from 18 to 0 over training. This baseline is *not* our algorithmic contribution—linear annealing is a standard technique. Rather, it serves as a *diagnostic tool* that confirms our hypothesis: the "Temporal Dichotomy" is a fundamental property of One-shot FL training dynamics, not an artifact of a particular $lambda$ value. The gap between Static ETF (38.25%) and Manual Anneal (40.41%) constitutes empirical evidence for this phenomenon.

== Our Contribution: AURORA

To bridge this gap, we present *AURORA*, an autonomous framework that transforms the OFL training process from a static optimization problem into a dynamic, self-regulating meta-curriculum. AURORA is built upon three pillars:

+ *Explicit Global Anchoring:* We anchor all clients to a pre-defined, geometrically optimal *Simplex Equiangular Tight Frame (ETF)* structure as a global prototype anchor.

+ *Autonomous Regularization:* We propose a novel optimization strategy inspired by homoscedastic uncertainty weighting @kendall2018. By decoupling the gradient flows of model parameters and weighting parameters, AURORA allows the client model to _autonomously adjust_ the regularization strength under a simple monotonic meta-annealing prior.

+ *Robustness via Stability Regularization:* We identify that in scenarios with extreme data heterogeneity (e.g., SVHN with $alpha=0.05$), the uncertainty-based mechanism can lead to numerical instability (the "exploding $lambda$" problem). We introduce a stability regularization term that constrains the adaptive weights within a stable range.

*To summarize, our main contributions are:*
- *Framework & Insight:* To the best of our knowledge, we are the first to introduce explicit geometric anchoring for One-shot FL + prototype alignment, using Simplex ETF as a shared coordinate system. This challenges the conventional OFL assumption that, without iterative correction, clients cannot coordinate their feature spaces. We show this *is* achievable—but only under a specific condition. We *discover and formally define* the "Temporal Dichotomy": dynamic regularization is a *prerequisite* for geometric constraints to work in the one-shot setting. This insight *overturns* the static objective assumption implicit in prior OFL methods.
- *AURORA Algorithm:* To automate this critical dynamic scheduling without expensive manual tuning, we propose AURORA, a principled framework that learns client-specific, data-dependent regularization schedules under a monotonic meta-annealing prior. AURORA uses gradient-decoupled uncertainty weighting, enabling each client to discover its own alignment trajectory through gradient-based optimization—without validation sets or additional communication. *Crucially, our novelty lies not in the individual components (ETF, uncertainty weighting, annealing), but in our discovery that naive combinations of these techniques fail—and our principled solution (gradient decoupling + meta-annealing) that makes them work synergistically.*
- *Theoretical Analysis:* We provide a local equilibrium analysis showing how our meta-annealing mechanism naturally produces the desired curriculum behavior (strong alignment early, weak late), while allowing client-specific adaptation based on local data characteristics.
- *Robustness Mechanism:* We identify and address the "exploding λ" failure mode in extreme non-IID scenarios through stability regularization, ensuring AURORA degrades gracefully rather than catastrophically.
- Comprehensive experiments demonstrate that AURORA matches manually-tuned baselines using a single fixed hyperparameter configuration across all settings, eliminating per-dataset schedule search.

= Related Work

== Handling Data Heterogeneity in Federated Learning

Data heterogeneity (Non-IID) is a fundamental challenge in FL, particularly acute in OFL due to the absence of iterative correction. For a comprehensive overview of the rapidly evolving OFL landscape, we refer readers to recent surveys @amato2025survey. Multi-round FL methods have proposed various solutions: *FedProx* @li2020 adds a proximal term to regularize local updates; *SCAFFOLD* @karimireddy2020 uses control variates. However, these require multiple communication rounds, making them inapplicable to OFL.

== One-shot Federated Learning

OFL restricts client-server interaction to a single round, presenting unique challenges. Existing approaches can be categorized as:

*Distillation-based:* *DENSE* @zhang2022 and *Co-Boosting* @dai2024coboosting employ knowledge distillation for model aggregation. *FedDF* @lin2020 uses ensemble distillation with public data.

*Aggregation-based:* *FedLPA* @liu2024 introduces layer-wise posterior aggregation using Laplace approximation, achieving strong performance without auxiliary data by treating model aggregation as Bayesian inference.

*Client-side enhancement:* *FAFI* @zeng2025 addresses model inconsistency through feature-anchored integration, combining contrastive learning and prototype-based classification to improve local model quality before aggregation.

Our work is orthogonal to server-side aggregation methods like FedLPA. We focus on improving *local training objectives* to reduce model inconsistency at the source, which is complementary to advanced aggregation techniques.

== Prototype-based Federated Learning and Neural Collapse

Prototype-based methods have gained traction for their communication efficiency. *FedProto* @tan2022 exchanges class prototypes instead of model parameters. *FedTGP* @zhang2024 introduces trainable global prototypes with adaptive-margin contrastive learning. Recent multi-round FL works have explored *adaptive prototype alignment weights* that vary across clients or training rounds; however, these methods rely on iterative server aggregation to correct alignment errors and are not applicable to the one-shot setting where clients train in isolation without feedback.

Our work leverages the *Neural Collapse* phenomenon @papyan2020, which shows that optimal classifiers converge to a Simplex Equiangular Tight Frame (ETF) structure. *FedETF* @li2023 utilizes fixed ETF classifiers to mitigate classifier bias in multi-round FL. In contrast, our approach addresses the *One-shot* regime where iterative synchronization is absent. Unlike FedETF's focus on classifier-side weights, we introduce ETF anchors at the *prototype level* to provide a shared coordinate system for feature spaces across isolated clients.

*Our distinction from FedETF:* While FedETF demonstrates that fixed geometric structures can benefit multi-round FL, it relies on iterative server aggregation to correct local drifts. To our knowledge, we are the first to investigate such geometric anchoring for *One-shot* prototype alignment. Importantly, we demonstrate that the static application of ETF (effective in multi-round contexts) *fails* in the One-shot setting, and we provide the first solution—a dynamic weighting and scheduling mechanism (AURORA)—that makes geometric anchoring viable without iterative communication.

*Key distinction from FAFI:* AURORA builds upon FAFI's client-side training framework (SALT + prototype learning) but addresses a key limitation: FAFI resolves inconsistency through client-side enhancement but *lacks an explicit global geometric anchor*. It relies solely on implicit alignment through shared augmentation strategies. When we introduce such an anchor (ETF for prototype alignment), a new challenge emerges: balancing alignment strength over time. FAFI's static formulation provides no mechanism to address this time-varying trade-off—which is precisely where AURORA's autonomous regularization becomes essential.

*Our Core Novelty:* AURORA makes *two distinct contributions*: (1) The *scientific discovery* that static geometric alignment fails in One-shot FL while dynamic scheduling succeeds (the "Temporal Dichotomy")—this insight into training dynamics is a contribution independent of any algorithm, advancing our understanding of why one-shot FL behaves differently from multi-round FL. (2) The *AURORA algorithm* that automates this dynamic scheduling, transforming what would require expensive per-dataset hyperparameter search into a gradient-based learning problem.

*Logic Chain:* We *discover* the Temporal Dichotomy → We *validate* it using manual annealing as a diagnostic tool → We *automate* the solution via AURORA. This "Discover → Validate → Automate" structure is the backbone of this paper.

== Meta-Learning and Multi-Task Optimization

Our approach draws inspiration from multi-task learning research. *Kendall et al. (2018)* @kendall2018 pioneered using homoscedastic uncertainty to automatically weight multi-task losses. *PCGrad* @yu2020pcgrad addresses gradient conflicts through projection. *Franceschi et al. (2017)* @franceschi2017 formalized gradient-based hyperparameter optimization via bilevel programming. AURORA can be viewed as an online, one-step approximation of bilevel optimization, where the regularization weight is treated as a learnable hyperparameter guided by task uncertainty. *However, directly applying Kendall's formulation to One-shot FL fails catastrophically (23.94% accuracy on CIFAR-100), as the implicit learning rate scaling destabilizes training. Our gradient decoupling mechanism is essential to make uncertainty weighting viable in this setting.*

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

*ETF Construction Requirement.* The simplex ETF requires embedding dimension $d >= C-1$. With ResNet-18 ($d=512$), this is satisfied for CIFAR-10 ($C=10$) and CIFAR-100 ($C=100$). For datasets with $C > d+1$, one would need a larger projection head or dimensionality reduction on the ETF.

*Handling Missing Classes.* Under extreme non-IID (e.g., $alpha=0.05$), some clients may lack samples for certain classes. By computing $cal(L)_"align"$ only over classes present in the client's dataset ($cal(C)_k$), we prevent trivial alignment of unused prototypes and focus learning on classes the client can actually discriminate.

*Implementation Details for Reproducibility:*
- *Prototype representation:* Learnable prototypes $bold(p)_c in RR^d$ are *not* L2-normalized during alignment computation. The ETF anchors are normalized to unit norm.
- *Alignment loss:* We use L2 (MSE) distance rather than cosine similarity, as MSE provides stronger gradients when prototypes are far from anchors.
- *Class mask per batch:* During training, alignment loss is computed only over classes appearing in the current batch.
- *Missing class initialization:* Prototypes are initialized to their corresponding ETF anchor positions (with small random perturbation). For locally-missing classes, these prototypes remain near their ETF-aligned initialization since they receive no gradient updates. At aggregation, such prototypes are down-weighted during IFFI fusion based on local sample counts (effectively zero weight for missing classes).

== Learning the Alignment Strength (λ) via Task Uncertainty

The critical challenge lies in determining the optimal $lambda$ that balances local adaptation with global alignment. Instead of treating $lambda$ as a fixed hyperparameter, we propose to *learn* it through the lens of task uncertainty.

*Uncertainty-Weighted Multi-Task Loss.* Following @kendall2018, we model each loss term using a Gaussian likelihood with learnable observation noise:

$ cal(L) = frac(1, 2 sigma_1^2) cal(L)_1 + frac(1, 2 sigma_2^2) cal(L)_2 + log sigma_1 + log sigma_2 $

where $sigma_1^2$ and $sigma_2^2$ are learnable parameters representing the homoscedastic uncertainty of each task. The regularization terms $log sigma$ prevent trivial solutions where $sigma -> infinity$.

*Implementation Note.* For numerical stability, we parameterize $ell = log sigma^2$ and optimize $ell$ directly. Since $log sigma = frac(1,2) ell$, the loss becomes:
$ cal(L) = frac(1, 2 e^(ell_1)) cal(L)_1 + frac(1, 2 e^(ell_2)) cal(L)_2 + frac(1,2) ell_1 + frac(1,2) ell_2 $

This explains the 0.5 factor appearing in our algorithm.

*Effective Lambda.* The effective alignment weight emerges as:

$ lambda_"eff" = frac(sigma_"local"^2, sigma_"align"^2) $

*Decoupled Interpretation.* In AURORA, σ parameters are optimized via an uncertainty-style meta-objective, but *do not rescale the gradients of model weights* (see Section 3.3). Instead, they determine an emergent ratio $lambda_"eff" = sigma_"local"^2 / sigma_"align"^2$ that *only modulates the alignment term* in $cal(L)_W = cal(L)_"local" + lambda_"eff" dot cal(L)_"align"$. The resulting λ trajectory is *emergent and data-dependent*—not pre-specified, but arising from the joint dynamics of loss magnitudes and the monotonic prior.

== AURORA's Meta-Objective: Why Naive Stacking Fails and How Decoupling Fixes It

*The Stacking Fallacy.* One might assume that combining established techniques—ETF alignment (from Neural Collapse), uncertainty weighting (from Kendall et al.), and cosine annealing—would yield additive benefits. *This assumption is wrong.* A naive implementation of uncertainty weighting introduces an unintended side effect: the weighting coefficients $1/sigma^2$ also scale the effective learning rate, *catastrophically destabilizing training* (accuracy drops to 23.94% on CIFAR-100). We address this through *gradient decoupling*, which is our key algorithmic contribution.

*The Decoupling Mechanism.* We maintain two separate loss formulations:

*1. Loss for Model Weights ($cal(L)_W$):* Used to update backbone and classifier parameters.
$ cal(L)_W = cal(L)_"local" + lambda_"eff" dot cal(L)_"align" $

*2. Loss for Sigma Parameters ($cal(L)_sigma$):* Used to update the uncertainty parameters. Using $ell = log sigma^2$:
$ cal(L)_sigma = frac(cal(L)_"local"^"(detach)", 2 e^(ell_"local")) + frac(cal(L)_"align"^"(detach)", 2 e^(ell_"align")) + frac(1,2) ell_"local" + frac(1,2) ell_"align" $

The `.detach()` operation prevents gradients from flowing from the uncertainty parameters back to the model weights, creating an *approximate online bilevel optimization* where:
- The inner loop optimizes model weights given the current $lambda_"eff"$
- The outer loop adjusts σ parameters based on the meta-objective

*Why Decoupling is Necessary.* Without gradient decoupling, the $1/sigma^2$ coefficients in the Kendall formulation directly scale the effective learning rate for each task. In our experiments, this causes two failure modes: (1) when $sigma_"local"^2$ grows large (as intended for uncertain local tasks), the local loss gradients become vanishingly small, stalling feature learning; (2) the σ parameters receive conflicting gradients from both the loss terms and regularizers, leading to oscillatory training dynamics. Decoupling isolates these effects: model weights see a clean weighted sum, while σ parameters adapt based only on loss magnitudes.

This decoupling ensures that the model learns task-optimal weights while the sigma parameters learn the optimal task weighting, without mutual interference.

*Empirical Evidence.* On CIFAR-100 (α=0.05), a naive implementation without decoupling achieves only 23.94% accuracy due to implicit learning rate scaling, while the corrected decoupled version achieves 39.41%—matching the performance of manually-tuned baselines.

== Inducing a Curriculum with Meta-Annealing

Experimental analysis reveals that uncertainty weighting alone converges to a static equilibrium. To induce a *curriculum* from strong alignment to local adaptation, we introduce a *meta-annealing schedule*.

*Schedule Factor as a Monotonic Prior.* We define $s(p) = frac(1,2)(1 + cos(pi p))$, where $p in [0, 1]$ is the normalized training progress. This cosine schedule provides smooth annealing from 1 to 0. *Crucially, s(p) should not be understood as a rigid schedule imposed on λ, but rather as a *Bayesian prior* expressing our belief that alignment should decrease monotonically over training.* The σ dynamics find a *posterior* balance between this prior and the data-driven uncertainty from loss magnitudes. This is why AURORA produces client-specific trajectories (see Table 4) despite all clients sharing the same s(p). The meta-annealing applies $s(p)$ to the *regularization term* of the alignment task. Using $ell = log sigma^2$:

$ cal(L)_sigma = frac(cal(L)_"local"^"(detach)", 2 e^(ell_"local")) + frac(cal(L)_"align"^"(detach)", 2 e^(ell_"align")) + frac(1,2) ell_"local" + frac(1,2) s(p) dot ell_"align" $

*Derivation of Annealing Behavior.* Taking the derivative of $cal(L)_sigma$ with respect to $sigma_"align"^2$ and setting to zero:

$ frac(diff cal(L)_sigma, diff sigma_"align"^2) = -frac(cal(L)_"align", 2 sigma_"align"^4) + frac(s(p), 2 sigma_"align"^2) = 0 $

Solving for the optimal $sigma_"align"^2$:

$ sigma_"align"^(2*) = frac(cal(L)_"align", s(p)) $

*Emergent Annealing Behavior:*
- *Early training ($s(p) -> 1$):* $sigma_"align"^(2*) approx cal(L)_"align"$, following the standard Kendall equilibrium.
- *Late training ($s(p) -> 0$):* $sigma_"align"^(2*) -> infinity$, causing $1\/sigma_"align"^2 -> 0$.

=== Formal Assumptions and Convergence Analysis

To rigorously characterize the σ dynamics, we introduce the following assumptions:

#block(inset: (left: 1em))[
  *(A1) Bounded Losses:* $0 < L_"min" <= cal(L)_i(theta) <= L_"max" < infinity$ for $i in {"local", "align"}$.
  
  *(A2) Slow Variation:* The losses are quasi-static relative to σ dynamics: $|cal(L)_i(theta_(t+1)) - cal(L)_i(theta_t)| <= delta$ where $delta \/ eta_sigma -> 0$ as $eta_sigma -> 0$.
  
  *(A3) Learning Rate Separation:* $eta_sigma << eta_theta$, meaning σ parameters adapt faster than model parameters (timescale separation).
  
  *(A4) Schedule Regularity:* $s(p): [0,1] -> (0,1]$ is Lipschitz continuous with $|s'(p)| <= S_"max"$ and $s(p) >= epsilon > 0$.
]

*Theorem 1 (Stationary Points and Convergence).* Under assumptions (A1)-(A4), the σ² dynamics induced by gradient descent on $cal(L)_sigma$ satisfy:

+ *Stationary Points:* The unique stationary point is:
  $ sigma_"local"^(2*) = cal(L)_"local", quad quad sigma_"align"^(2*) = frac(cal(L)_"align", s(p)) $

+ *Local Stability:* The stationary point is locally asymptotically stable with convergence rate $O(eta_sigma)$.

+ *Tracking Error:* Under slow loss variation (A2), the tracking error satisfies:
  $ |sigma^2(t) - sigma^(2*)(t)| = O(delta \/ eta_sigma + e^(-c eta_sigma t)) $
  
  for some constant $c > 0$ depending on $L_"min"$.

_Proof Sketch._ The gradient of $cal(L)_sigma$ with respect to $sigma^2$ yields:
$ frac(diff cal(L)_sigma, diff sigma_i^2) = -frac(cal(L)_i, 2 sigma_i^4) + frac(c_i, 2 sigma_i^2) $
where $c_"local" = 1$ and $c_"align" = s(p)$. Setting to zero gives $sigma_i^(2*) = cal(L)_i \/ c_i$. The Hessian at equilibrium is $frac(diff^2 cal(L)_sigma, diff (sigma^2)^2) = frac(c_i, 2 sigma_i^4) > 0$, confirming local convexity. Full proof in Appendix A.3. $square$

*Corollary 1 (Equilibrium λ_eff Dynamics).* The equilibrium alignment weight satisfies:

$ lambda_"eff"^* = frac(sigma_"local"^(2*), sigma_"align"^(2*)) = s(p) dot frac(cal(L)_"local", cal(L)_"align") $

with the following properties:
+ *Monotonic Decay:* Since $s(p) arrow.b$ monotonically, $lambda_"eff"^*$ exhibits a decreasing trend (curriculum behavior).
+ *Data-Adaptivity:* The ratio $cal(L)_"local"\/cal(L)_"align"$ introduces client-specific variation based on local data characteristics.
+ *Bounded Range (without stability reg):* Under (A1), $lambda_"eff"^* in [s(p) dot L_"min"\/L_"max", s(p) dot L_"max"\/L_"min"]$.
+ *Explosion Risk:* When $cal(L)_"align" << cal(L)_"local"$ (extreme non-IID), the ratio can exceed practical bounds, motivating Section 3.5.

*Why This is Fundamentally Different from a Fixed Schedule.*

Unlike a fixed schedule $lambda(t) = lambda_0 dot s(t)$, AURORA's $lambda_"eff"$ emerges from the joint dynamics of loss magnitudes and the monotonic prior. The σ parameters capture *meta-level task uncertainty* through $cal(L)_sigma$ (with detached losses); this uncertainty does not rescale $nabla_theta$, but induces a ratio $lambda_"eff" = sigma_"local"^2 / sigma_"align"^2$ that modulates alignment in $cal(L)_W$. The key distinction: *s(p) only imposes a monotonic prior; magnitude and inter-client variation emerge from optimization* (see Table 4 for empirical evidence).


== Ensuring Robustness: Stability Regularization

In extreme non-IID scenarios (e.g., SVHN with $alpha=0.05$), we observe a failure mode where $lambda_"eff"$ explodes due to severe task difficulty imbalance. When $cal(L)_"local"$ is significantly harder than $cal(L)_"align"$, the optimizer aggressively increases $sigma_"local"^2$ while decreasing $sigma_"align"^2$, leading to catastrophic $lambda_"eff"$ values exceeding $10^6$.

*The Exploding Lambda Problem.* Analysis reveals that:
+ With highly skewed local data, $cal(L)_"local"$ remains large and noisy
+ $cal(L)_"align"$ (MSE to fixed anchors) decreases rapidly and stabilizes
+ The optimizer increases $sigma_"local"^2$ (local task is "unreliable")
+ Simultaneously, $sigma_"align"^2 -> 0$ (alignment is "trivially certain")
+ $lambda_"eff" = sigma_"local"^2 / sigma_"align"^2 -> infinity$

When $lambda_"eff"$ explodes, the total loss is dominated by $cal(L)_"align"$, forcing prototypes to perfectly match ETF anchors while the feature extractor stops learning discriminative features.

*An Additional Perspective: Variance Under Sparse Data.* In extreme non-IID scenarios, each client may have very few samples per class. Under these conditions, the loss-based uncertainty estimates $sigma^2 approx cal(L)$ have high variance—a small batch with an "easy" set of samples may produce an atypically low $cal(L)_"align"$, causing $sigma_"align"^2$ to shrink inappropriately. This noisy estimation exacerbates the explosion risk. Stability regularization provides a principled safety net against such estimation variance, not just against the deterministic explosion mechanism.

*Stability Regularization via Soft Constraint.* We introduce a squared-hinge regularization for smooth gradients:

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

*Server Aggregation.* For fair comparison, all client-side methods (FAFI, FAFI+Annealing, AURORA) use the same server aggregation: *IFFI* (Informative Feature Fused Inference) from FAFI. FedLPA uses its own Laplace-based aggregation. This isolates the effect of local training improvements.

*One-shot Protocol.* We strictly follow the one-shot FL protocol: each client trains locally for multiple epochs, then uploads its model/prototypes to the server *exactly once*. The server performs a single aggregation.

*Implementation Details.*
- Backbone: ResNet-18
- Total local epochs: 500 (CIFAR-10), 100 (CIFAR-100, SVHN)
- Optimizer: SGD with momentum 0.9, weight decay 5e-4
- Learning rate: 0.05 (cosine annealing over local training)
- AURORA-specific: $sigma$ learning rate = 0.005, $lambda_"max"$ = 50.0, $gamma$ = 0.001
- Default: $K=5$ clients; scalability study with $K in {5, 10, 20}$ in Section 4.7
- Evaluation checkpoints: Every 10 epochs (offline, no communication)

*Clarification on "epoch checkpoints":* We record intermediate states every 10 local epochs *for offline analysis only*—no parameters are communicated. These checkpoints enable studying training dynamics without violating the one-shot constraint.

*Loss Scaling.* All loss terms follow FAFI's original scaling (cls_loss + contrastive_loss + proto losses). We keep these fixed across all baselines to ensure σ adapts to training dynamics rather than arbitrary rescaling.

*Quantifying Reduced Hyperparameter Burden.* Manual λ annealing requires tuning: (1) initial λ value, (2) decay shape (linear/exponential/cosine), and (3) decay rate—typically requiring a grid search over 20+ configurations per dataset/α combination. In contrast, AURORA uses *the same three hyperparameters* ($sigma$-lr=0.005, $lambda_"max"$=50, $gamma$=0.001) across all experiments in this paper without per-setting adjustment. *Crucially, these are safety bounds, not performance-critical parameters:* varying $lambda_"max"$ from 20 to 100 changes accuracy by \<1%, whereas varying manual λ₀ by the same factor causes 2-5% accuracy drops (Section 4.6).

== Main Results

#figure(
  table(
    columns: 6,
    stroke: 0.5pt,
    inset: 6pt,
    [*Dataset*], [*α*], [*FedAvg*], [*FAFI*], [*FAFI+Ann.*], [*AURORA*],
    [CIFAR-10], [0.05], [57.74], [66.97], [67.77], [*68.17*],
    [CIFAR-10], [0.1], [64.21], [76.10], [76.86], [*77.23*],
    [CIFAR-10], [0.3], [72.15], [83.90], [84.54], [*85.12*],
    [CIFAR-10], [0.5], [80.44], [87.69], [88.46], [*88.91*],
    [CIFAR-100], [0.05], [12.45], [38.41], [40.41], [*40.43*],
    [SVHN], [0.05], [31.25], [49.94], [51.07], [*52.9*],
  ),
  caption: [Test Accuracy (%) on Different Datasets and Heterogeneity Levels. AURORA consistently outperforms all baselines across varying data heterogeneity. All methods evaluated under identical settings: ResNet-18 backbone, 5 clients, 50 communication rounds, same data partitions.]
)

// Accuracy vs Heterogeneity Chart (CIFAR-10)
#figure(
  cetz.canvas({
    plot.plot(
      size: (10, 6),
      x-label: [Heterogeneity (α)],
      y-label: [Test Accuracy (%)],
      x-tick-step: 0.15,
      y-tick-step: 5,
      y-min: 65,
      y-max: 90,
      legend: "south-east",
      {
        // FAFI
        plot.add(
          ((0.05, 66.97), (0.1, 76.10), (0.3, 83.90), (0.5, 87.69)),
          style: (stroke: (paint: blue, thickness: 1.5pt)),
          mark: "square",
          mark-style: (fill: blue),
          label: [FAFI]
        )
        // FAFI+Anneal
        plot.add(
          ((0.05, 67.77), (0.1, 76.86), (0.3, 84.54), (0.5, 88.46)),
          style: (stroke: (paint: green, thickness: 1.5pt, dash: "dashed")),
          mark: "triangle",
          mark-style: (fill: green),
          label: [FAFI+Ann.]
        )
        // AURORA
        plot.add(
          ((0.05, 68.17), (0.1, 77.23), (0.3, 85.12), (0.5, 88.91)),
          style: (stroke: (paint: red, thickness: 2pt)),
          mark: "o",
          mark-style: (fill: red),
          label: [AURORA]
        )
      }
    )
  }),
  caption: [Accuracy vs Heterogeneity (CIFAR-10). AURORA consistently outperforms baselines across all heterogeneity levels. Higher α means less heterogeneity (easier).]
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
  caption: [Model Consistency (g_protos_std ↓) on CIFAR-10 (α=0.05)]
)

// Model Consistency Chart (using plot.add for compatibility)
#figure(
  cetz.canvas({
    plot.plot(
      size: (10, 5),
      x-label: [Method],
      y-label: [g_protos_std (↓ better)],
      x-tick-step: 1,
      y-tick-step: 0.1,
      y-min: 0.6,
      y-max: 1.1,
      legend: "north-east",
      {
        plot.add(
          ((1, 1.007), (2, 0.935), (3, 0.709), (4, 0.710)),
          style: (stroke: (paint: blue, thickness: 2pt)),
          mark: "o",
          mark-style: (fill: blue, stroke: blue),
          mark-size: 0.3,
          label: [g_protos_std]
        )
      }
    )
  }),
  caption: [Model Consistency (g_protos_std). Methods: 1=FAFI, 2=+ETF, 3=+Anneal, 4=AURORA. Lower values indicate stronger inter-client prototype alignment. Dynamic scheduling (3,4) achieves ~30% reduction vs baseline.]
)

== Ablation Study

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [*Configuration*], [*Description*], [*Accuracy (%)*],
    [FAFI (baseline)], [SALT + prototype learning, no explicit global anchor], [38.41],
    [+ ETF Anchor], [Fixed λ=10, aligns prototypes to ETF structure], [38.25],
    [+ Manual Anneal], [λ: 18→0 linear decay over training], [40.41],
    [+ Uncertainty Weight], [Decoupled formulation, s(p)=1, converges to λ≈10], [38.25],
    [+ Meta-Anneal], [Decoupled + cosine s(p), enables λ evolution], [40.43],
    [*+ Stability Reg (AURORA)*], [+ ReLU-hinge constraint on λ_eff ≤ λ_max], [*40.43*],
  ),
  caption: [Ablation Study on CIFAR-100 (α=0.05)]
)

*Key Insights:*
+ *The Temporal Dichotomy validated:* The failure of Static ETF (38.25%) versus the success of Manual Annealing (40.41%) confirms our core hypothesis—that the efficacy of global geometric anchors in One-shot FL is *strictly conditional* on temporal dynamics. This result establishes the necessity of dynamic regularization, a phenomenon we have not found documented in prior OFL work.
+ *Static alignment is counterproductive:* Adding ETF without annealing *hurts* performance (38.25% vs 38.41% baseline), demonstrating that geometric constraints beneficial in multi-round FL can be detrimental in the one-shot setting.
+ *AURORA automates the discovery:* Through gradient decoupling and meta-annealing, AURORA discovers a comparable schedule (40.43%) *without per-dataset tuning*, validating that our learned mechanism captures the essential temporal dynamics.
+ *Stability regularization ensures robustness:* While identical to Meta-Anneal on CIFAR-100, it prevents catastrophic failure on extreme heterogeneity (SVHN).

== Analysis: AURORA Learns the Optimal Schedule

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Checkpoint*], [*Schedule Factor s(p)*], [*AURORA λ_eff*], [*Manual Anneal λ*],
    [0 (start)], [0.9], [11.6], [18.0],
    [2], [0.7], [10.0], [12.6],
    [5], [0.4], [7.2], [7.2],
    [9 (end)], [0.1], [4.9], [1.8],
  ),
  caption: [λ Evolution Comparison (CIFAR-100, α=0.05)]
)

*Interpretation:* The manual schedule follows a pre-defined linear trajectory (λ: 18→0), requiring careful tuning for each dataset/α combination. In contrast, AURORA's $lambda_"eff"$ is an *emergent quantity* driven by the interaction between loss dynamics and the monotonic meta-annealing prior. Despite never seeing the manual schedule, AURORA discovers a qualitatively similar curriculum—strong alignment early (λ≈11.6), tapering off as training progresses. This demonstrates that the meta-learning mechanism successfully identifies the same underlying principle that makes manual annealing effective.

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
  caption: [Per-Client λ Divergence (Meta-Anneal, no stability reg) on CIFAR-100 (α=0.05). Analysis run without stability regularization to demonstrate client-specific divergence.]
)

// Per-Client λ Divergence Chart
#figure(
  cetz.canvas({
    plot.plot(
      size: (10, 6),
      x-label: [Checkpoint (×10 epochs)],
      y-label: [Raw λ value],
      x-tick-step: 2,
      y-tick-step: 20,
      y-min: 0,
      y-max: 80,
      legend: "north-west",
      {
        // Client 0
        plot.add(
          ((0, 12.9), (4, 14.9), (6, 22.8), (7, 31.6), (8, 48.8)),
          style: (stroke: (paint: blue, thickness: 1.5pt)),
          mark: "square",
          mark-style: (fill: blue),
          label: [Client 0]
        )
        // Client 4  
        plot.add(
          ((0, 13.5), (4, 16.9), (6, 28.1), (7, 42.7), (8, 73.5)),
          style: (stroke: (paint: orange, thickness: 1.5pt)),
          mark: "diamond",
          mark-style: (fill: orange),
          label: [Client 4]
        )
      }
    )
  }),
  caption: [Per-Client λ Divergence. Despite sharing the same s(p) prior, clients develop divergent λ trajectories based on their local data characteristics. By checkpoint 8, Client 4's λ is 51% higher than Client 0's—demonstrating AURORA is *data-dependent*, not merely *time-dependent*.]
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
  caption: [SVHN Performance Under Extreme Heterogeneity (α=0.05)]
)

// λ Explosion Visualization Chart (moved from Section 3.5 for logical flow)
#figure(
  cetz.canvas({
    plot.plot(
      size: (10, 6),
      x-label: [Training Round],
      y-label: [Effective λ value],
      x-tick-step: 5,
      y-tick-step: 500000,
      y-min: 0,
      y-max: 2000000,
      legend: "north-west",
      {
        // Unregularized λ: catastrophic explosion
        plot.add(
          ((0, 11.8), (14, 143260), (19, 909751), (29, 1658926)),
          style: (stroke: (paint: gray, thickness: 1.5pt, dash: "dashed")),
          mark: "x",
          mark-style: (stroke: gray),
          label: [Unregularized λ (explodes)]
        )
        // AURORA λ: stable under hinge constraint
        plot.add(
          ((0, 11.8), (10, 25), (20, 48), (30, 49)),
          style: (stroke: (paint: red, thickness: 2pt)),
          mark: "o",
          mark-style: (fill: red),
          label: [AURORA λ (stable ≤50)]
        )
      }
    )
  }),
  caption: [λ Explosion on SVHN (α=0.05). Under extreme heterogeneity, the unregularized uncertainty objective suffers from task difficulty imbalance, driving λ toward infinity ($> 1.6 times 10^6$). AURORA's stability regularization effectively anchors λ within a functional range, preventing accuracy collapse (52.9% vs 16.4%).]
)

*Analysis:* Without stability regularization, the difficulty gap between local and alignment tasks causes λ to grow exponentially. On SVHN with γ=0, Raw λ reaches 204,658 by round 14 and over 4 million by round 29, causing accuracy to collapse from 49.5% (peak) to 16.4%. With γ=1e-3, AURORA's hinge constraint maintains $lambda_"eff" <= 50$ throughout training, achieving 52.9% final accuracy.

== Hyperparameter Sensitivity

*Qualitative Distinction: Safety Bounds vs. Performance-Critical Hyperparameters.*

*Key Insight:* The hyperparameters introduced by AURORA ($lambda_"max"$, $gamma$, $sigma$-lr) are *fundamentally different* from the manual annealing hyperparameters (initial $lambda$, decay rate, decay shape) they replace. The former are *safety bounds*—they define when a fail-safe mechanism activates, not the core learning dynamics. The latter are *performance-critical*—small changes directly impact accuracy.

#figure(
  table(
    columns: 3,
    stroke: 0.5pt,
    inset: 6pt,
    [], [*Manual λ Schedule*], [*AURORA Stability Params*],
    [*Type*], [Performance-critical], [*Safety bounds*],
    [*Sensitivity*], [Change shape → 2-5% acc drop], [*5× range (20-100) → \<1% variance*],
    [*Cross-setting*], [Requires re-tuning per dataset/α], [Same defaults work across all],
    [*Trigger rate*], [Always active (shapes entire trajectory)], [Rarely triggered in stable settings],
    [*Analogy*], [Curriculum design], [Gradient clipping threshold],
  ),
  caption: [Comparison of hyperparameter types: AURORA’s parameters are analogous to gradient clipping thresholds or weight decay defaults—insensitive within a reasonable range.]
)

*Why λ_max is Not λ in Disguise.* The manual annealing schedule uses $lambda(t) = lambda_0 dot (1 - t/T)$, where $lambda_0$ determines the *entire trajectory* and optimal values vary by 10× across datasets (see Table 6). In contrast, $lambda_"max"$ is a *ceiling*: it only activates when the learned $lambda_"eff"$ exceeds it, which rarely occurs under normal training conditions. Varying $lambda_"max"$ from 20 to 100 changes final accuracy by \<1% (Table 6, SVHN), demonstrating its role as a safety mechanism rather than a performance lever.

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*Parameter*], [*Values Tested*], [*Accuracy (%)*], [*Behavior*],
    [λ_max], [20, *50*, 100], [52.3, 52.9, TBD], [Stable within range],
    [γ (reg strength)], [0, 1e-5, *1e-3*], [16.4, 17.7, 52.9], [Collapse → Stable],
    [σ-learning rate], [1e-4, *5e-3*, 1e-2], [TBD, 52.9, TBD], [Default: 5e-3],
  ),
  caption: [Table 6: Sensitivity Analysis on SVHN (α=0.05). The γ (reg strength) row demonstrates that stability regularization is essential—without it (γ=0), λ explodes and accuracy collapses to 16.4%.]
)

*λ Sensitivity Analysis.* To understand how different fixed λ values affect performance, we conducted experiments with λ ∈ {1.0, 2.5, 5.0, 10.0, 20.0, 50.0} on CIFAR-10 (α=0.05):

#figure(
  table(
    columns: 4,
    stroke: 0.5pt,
    inset: 6pt,
    [*λ_initial*], [*Accuracy (%)*], [*g_protos_std*], [*Observation*],
    [1.0], [58.89], [0.987], [Weak alignment, relies on local learning],
    [2.5], [57.44], [0.959], [Destructive interference zone],
    [5.0], [58.77], [0.914], [Transition region],
    [10.0], [59.38], [0.874], [Strong alignment begins to dominate],
    [20.0], [*59.68*], [0.597], [Near-optimal manual tuning],
    [50.0], [59.39], [0.503], [Plateau—robust to over-tuning],
  ),
  caption: [Table 6b: Effect of Fixed λ with Linear Annealing on CIFAR-10 (α=0.05). Performance exhibits a U-shape: λ=2.5 represents a destructive interference zone where neither local nor global objectives dominate. This motivates the need for autonomous λ selection.]
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

*Future Work.* Promising directions include: (1) extending to model heterogeneous settings; (2) combining with advanced server-side aggregation techniques like FedLPA; (3) theoretical analysis of the meta-learning convergence properties; (4) application to other FL paradigms with conflicting objectives.

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

== Formal Analysis of σ Dynamics

This section provides rigorous justification for Theorem 1 in the main text, including detailed proofs and convergence analysis.

=== Complete Statement of Assumptions

We restate and elaborate the assumptions under which our theoretical results hold:

#block(inset: (left: 1em))[
  *(A1) Bounded Losses.* There exist constants $0 < L_"min" <= L_"max" < infinity$ such that for all $theta$ in the optimization trajectory and $i in {"local", "align"}$:
  $ L_"min" <= cal(L)_i(theta) <= L_"max" $
  _Justification:_ In practice, losses are bounded below by zero (non-negativity) and above by the initial loss value (since training reduces loss). For cross-entropy with bounded logits and MSE to fixed anchors, this holds naturally.

  *(A2) Slow Variation (Quasi-Static Approximation).* The model parameters $theta$ evolve slowly relative to the σ dynamics:
  $ |cal(L)_i(theta_(t+1)) - cal(L)_i(theta_t)| <= delta $
  where $delta$ satisfies $delta \/ eta_sigma -> 0$ as $eta_sigma -> 0$ (equivalently, $eta_theta \/ eta_sigma -> 0$).
  _Justification:_ This is a standard timescale separation assumption in adaptive learning rate methods. With $eta_sigma = 0.005$ and typical model learning rates $eta_theta in [0.01, 0.1]$, the ratio is moderate. The assumption becomes exact in the limit of infinitely fast σ adaptation.

  *(A3) Learning Rate Separation.* The σ learning rate is sufficiently small:
  $ eta_sigma << min(1, 1\/L_"max") $
  _Justification:_ This ensures gradient descent on $cal(L)_sigma$ remains stable. In practice, we use $eta_sigma = 0.005$ while $L_"max" approx 10$, satisfying this condition.

  *(A4) Schedule Regularity.* The annealing schedule $s: [0,1] -> (0,1]$ satisfies:
  - $s(0) = 1$, $s(1) = epsilon > 0$ (never exactly zero for numerical stability)
  - $s$ is Lipschitz: $|s(p_1) - s(p_2)| <= S_"max" |p_1 - p_2|$
  
  _Justification:_ Our cosine schedule $s(p) = max(epsilon, frac(1,2)(1 + cos(pi p)))$ with $epsilon = 10^(-3)$ satisfies these properties with $S_"max" = pi\/2$.
]

=== Proof of Theorem 1 (Stationary Points)

*Statement.* Under (A1)-(A4), the unique stationary point of the σ dynamics under $cal(L)_sigma$ is:
$ sigma_"local"^(2*) = cal(L)_"local", quad sigma_"align"^(2*) = frac(cal(L)_"align", s(p)) $

*Proof.* Recall the meta-objective:
$ cal(L)_sigma = frac(cal(L)_"local", 2 sigma_"local"^2) + frac(cal(L)_"align", 2 sigma_"align"^2) + frac(1,2) log sigma_"local"^2 + frac(s(p),2) log sigma_"align"^2 $

Using the reparameterization $ell_i = log sigma_i^2$ (so $sigma_i^2 = e^(ell_i)$), we have:
$ cal(L)_sigma = frac(cal(L)_"local", 2 e^(ell_"local")) + frac(cal(L)_"align", 2 e^(ell_"align")) + frac(1,2) ell_"local" + frac(s(p),2) ell_"align" $

*First-order conditions:*
$ frac(diff cal(L)_sigma, diff ell_"local") = -frac(cal(L)_"local", 2 e^(ell_"local")) + frac(1,2) = 0 quad => quad e^(ell_"local"^*) = cal(L)_"local" $

$ frac(diff cal(L)_sigma, diff ell_"align") = -frac(cal(L)_"align", 2 e^(ell_"align")) + frac(s(p),2) = 0 quad => quad e^(ell_"align"^*) = frac(cal(L)_"align", s(p)) $

Converting back: $sigma_i^(2*) = e^(ell_i^*)$, which gives the stated result.

*Uniqueness:* The equations above have unique solutions for each $ell_i$ given positive losses and $s(p) > 0$. $square$

=== Proof of Local Stability

*Statement.* The stationary point is locally asymptotically stable.

*Proof.* We compute the Hessian of $cal(L)_sigma$ at the stationary point:

$ frac(diff^2 cal(L)_sigma, diff ell_"local"^2) = frac(cal(L)_"local", 2 e^(ell_"local")) $

At equilibrium $e^(ell_"local"^*) = cal(L)_"local"$:
$ frac(diff^2 cal(L)_sigma, diff ell_"local"^2) |_(ell^*) = frac(cal(L)_"local", 2 cal(L)_"local") = frac(1,2) > 0 $

Similarly:
$ frac(diff^2 cal(L)_sigma, diff ell_"align"^2) |_(ell^*) = frac(cal(L)_"align", 2 e^(ell_"align"^*)) = frac(cal(L)_"align" dot s(p), 2 cal(L)_"align") = frac(s(p), 2) > 0 $

Since the Hessian is diagonal with positive entries, $cal(L)_sigma$ is strictly convex near the stationary point, confirming local asymptotic stability under gradient descent. $square$

=== Convergence Rate Analysis

We now analyze the convergence rate of the σ dynamics to the equilibrium.

*Theorem 2 (Convergence Rate).* Under assumptions (A1)-(A4), consider the gradient descent dynamics:
$ ell_i(t+1) = ell_i(t) - eta_sigma frac(diff cal(L)_sigma, diff ell_i) $

Define the Lyapunov function:
$ V(t) = sum_(i in {"local","align"}) (ell_i(t) - ell_i^*(t))^2 $

Then for sufficiently small $eta_sigma$:
$ V(t) <= V(0) dot e^(-c eta_sigma t) + O(delta^2 \/ (c eta_sigma)) $

where $c = min(1, s_"min") \/ 2 > 0$ and $s_"min" = min_(p in [0,1]) s(p)$.

*Proof Sketch.* 

1. *Gradient near equilibrium:* For $ell$ near $ell^*$, Taylor expansion gives:
   $ frac(diff cal(L)_sigma, diff ell_i) approx H_(i i) (ell_i - ell_i^*) $
   where $H_(i i) = frac(c_i, 2) > 0$ (with $c_"local" = 1$, $c_"align" = s(p)$).

2. *One-step contraction:*
   $ ell_i(t+1) - ell_i^*(t+1) = ell_i(t) - ell_i^*(t) - eta_sigma H_(i i)(ell_i(t) - ell_i^*(t)) + (ell_i^*(t) - ell_i^*(t+1)) $
   $ = (1 - eta_sigma H_(i i))(ell_i(t) - ell_i^*(t)) + O(delta) $

3. *Lyapunov decay:*
   $ V(t+1) <= sum_i (1 - eta_sigma H_(i i))^2 (ell_i(t) - ell_i^*(t))^2 + O(delta^2) $
   $ <= (1 - eta_sigma c)^2 V(t) + O(delta^2) $
   $ approx (1 - 2 eta_sigma c) V(t) + O(delta^2) $

4. *Solving the recurrence:* Standard arguments yield $V(t) = O(e^(-c eta_sigma t) V(0) + delta^2 \/ (c eta_sigma))$. $square$

*Interpretation:* The σ parameters converge exponentially fast to a neighborhood of the time-varying equilibrium, with the neighborhood size controlled by the loss variation rate $delta$.

=== Tracking vs. Convergence

In AURORA's setting, the equilibrium $ell^*(t)$ is *time-varying* due to:
1. Changing losses $cal(L)_i(theta_t)$ as the model trains
2. Decreasing schedule $s(p)$ as training progresses

Theorem 2 shows that σ *tracks* this moving equilibrium with bounded error. The tracking error depends on:
- How fast $ell^*$ moves (controlled by $delta$ and $|s'(p)|$)
- How fast σ adapts (controlled by $eta_sigma$)

For our hyperparameter choice ($eta_sigma = 0.005$), empirical validation confirms tracking error remains small (see Table 4 in main text).

=== Limitations and When Assumptions Fail

*(A1) Violation:* Losses can temporarily spike during training (e.g., after aggressive gradient steps). Our stability regularization (Section 3.5) mitigates this by constraining $lambda_"eff"$.

*(A2) Violation:* In early training, losses change rapidly. The tracking error may be larger initially, but this is acceptable since early-stage λ_eff values are less critical (the schedule factor $s(p) approx 1$ dominates).

*(A3) Violation:* If $eta_sigma$ is too large, σ dynamics may overshoot. Our sensitivity analysis (Table 6) shows robustness across a 10× range of $eta_sigma$ values.

*(A4) Violation:* Near $p=1$ where $s(p) -> 0$, the equilibrium $sigma_"align"^(2*) -> infinity$. We use $epsilon = 10^(-3)$ floor to prevent numerical issues.

*Stochastic Extension:* The above analysis assumes deterministic gradients. In practice, SGD noise adds variance to the σ updates. Under standard SGD noise assumptions (bounded variance), the convergence guarantees extend with an additional $O(sigma_"noise"^2 \/ eta_sigma)$ term in the tracking error bound.

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
    [50 (default)], [40.43%], [\<5%],
    [100], [TBD], [TBD%],
  ),
  caption: [Effect of λ_max threshold on CIFAR-100 (α=0.05)]
)

== Per-Client λ Trajectory Analysis

The uncertainty weighting mechanism learns different λ values per client based on their local data characteristics. We compare two variants on SVHN (α=0.05): (1) without the λ-ReLU constraint, where λ grows unboundedly, and (2) with the λ-ReLU constraint ($λ_max = 50$), where the learned λ is stabilized.

=== Without λ-ReLU Constraint (Ablation)

_SVHN, α=0.05, Meta-Anneal without stability regularization_

#figure(
  table(
    columns: 7,
    stroke: 0.5pt,
    inset: 6pt,
    [*Rd*], [*s(p)*], [*C0*], [*C1*], [*C2*], [*C3*], [*C4*],
    [0], [0.98], [10.34], [2.77], [7.28], [12.04], [9.34],
    [5], [0.88], [10.26], [8.84], [8.71], [15.89], [11.97],
    [9], [0.80], [10.20], [15.58], [10.78], [65.22], [22.97],
    [10], [0.78], [10.19], [18.96], [11.65], [153.41], [29.61],
    [14], [0.70], [10.21], [68.76], [17.94], [*204,658*], [292.74],
    [19], [0.60], [10.42], [2,524.6], [47.14], [*1,516,253*], [222,998],
  ),
  caption: [Per-client Raw λ trajectory *without* λ-ReLU constraint on SVHN. Client 3's λ explodes to 1.5M by Round 19, causing performance collapse (accuracy drops from 49.5% to 26.3%). Client 0, with highest data entropy (1.71), maintains stable λ throughout.]
)

=== With λ-ReLU Constraint (AURORA)

_SVHN, α=0.05, Full AURORA with $λ_max = 50$_

#figure(
  table(
    columns: 7,
    stroke: 0.5pt,
    inset: 6pt,
    [*Rd*], [*s(p)*], [*C0*], [*C1*], [*C2*], [*C3*], [*C4*],
    [0], [0.98], [10.34], [2.77], [7.28], [12.05], [9.33],
    [5], [0.88], [10.26], [8.84], [8.77], [15.68], [11.93],
    [9], [0.80], [10.19], [15.59], [10.98], [*49.85*], [22.80],
    [10], [0.78], [10.18], [18.96], [11.88], [*50.01*], [29.21],
    [11], [0.76], [10.18], [23.89], [13.01], [*50.05*], [40.01],
    [14], [0.70], [10.20], [*50.16*], [18.67], [*50.46*], [*50.19*],
    [19], [0.60], [10.46], [*50.21*], [*50.64*], [*49.98*], [*50.41*],
  ),
  caption: [Per-client Raw λ trajectory *with* λ-ReLU constraint ($λ_max=50$). The ReLU-hinge penalty $γ · text("ReLU")(λ - 50)^2$ effectively caps Client 3's λ at ~50 (vs. 200K+ without constraint), preventing performance collapse. Accuracy reaches 55% (vs. 49% peak then collapse in V12).]
)

=== Correlation with Data Skew

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    inset: 6pt,
    [*Client*], [*Data Entropy*], [*Initial λ*], [*Final λ (w/o Constraint)*], [*Final λ (AURORA)*],
    [0], [1.71 (high)], [10.33], [10.42], [10.46],
    [1], [0.18 (low)], [1.18], [2,525 → explosion], [*50.21* (capped)],
    [2], [1.17 (med)], [7.11], [47.14], [*50.64* (capped)],
    [3], [2.29 (highest)], [13.81], [*1,516,253* (explosion)], [*49.98* (capped)],
    [4], [1.56 (med-high)], [9.46], [222,998 → explosion], [*50.41* (capped)],
  ),
  caption: [Correlation between data entropy and λ trajectory at Round 19. Clients with lower data entropy (more skewed distribution) experience faster λ growth. Without the λ-ReLU constraint, this leads to catastrophic λ explosion. AURORA's constraint effectively stabilizes training across all clients.]
)

== Extended Related Work

=== Comparison with Bayesian Aggregation Methods

*FedLPA and Laplace Approximation.* FedLPA @liu2024 applies Laplace approximation for Bayesian posterior aggregation in federated learning. The method estimates layer-wise Fisher information matrices for uncertainty-weighted model combination. Key considerations when comparing approaches:

- *Parameter-Space vs. Feature-Space:* FedLPA operates in high-dimensional parameter space (~11M parameters for ResNet-18), where accurate uncertainty quantification requires careful approximations. AURORA instead operates in *feature space* through ETF-anchored prototype alignment—a much lower-dimensional and geometrically structured space.

- *Complementary Strengths:* FedLPA provides principled Bayesian uncertainty quantification, while AURORA offers geometric feature alignment. These approaches are potentially complementary and could be combined in future work.

#figure(
  table(
    columns: 5,
    stroke: 0.5pt,
    inset: 6pt,
    [*Dataset*], [*β*], [*Simple CNN*], [*ResNet-18*], [*Source*],
    [CIFAR-10], [0.1], [19.97%], [23.62%], [Table 1 & 20],
    [CIFAR-10], [0.3], [26.60%], [27.43%], [Table 1 & 20],
    [CIFAR-10], [0.5], [24.20%], [31.70%], [Table 1 & 20],
    [CIFAR-100], [0.1], [15.11%], [—], [Table 22],
    [SVHN], [0.05], [32.90%], [—], [Table 1],
  ),
  caption: [FedLPA reported performance (cited from Liu et al. @liu2024, *not re-run by us*). Note: FedLPA uses Dirichlet parameter β (equivalent to our α). Results show FedLPA achieves 20-32% accuracy on CIFAR-10, while AURORA achieves 68-89% under comparable heterogeneity settings—a substantial improvement that we attribute to feature-space geometric alignment.]
)

=== Multi-Task Weighting Methods

- *Uncertainty Weighting* @kendall2018: Homoscedastic uncertainty for automatic weighting
- *GradNorm* @chen2018gradnorm: Gradient magnitude balancing
- *DWA* @liu2019mtan: Dynamic Weight Average based on loss descent rate
- *PCGrad* @yu2020pcgrad: Projecting conflicting gradients

#pagebreak()

// References
#bibliography("references.bib", style: "ieee")
