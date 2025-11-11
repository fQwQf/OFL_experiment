# Why v11 fails

我们将不确定性加权和退火机制结合在一起形成了 v11 ，实验结果如下。

### 实验结果总结

实验设置:
*   算法: V11 ，分别应用了两种不同的全局调度策略。
*   数据集: CIFAR-100, alpha=0.05 (极度Non-IID)。
*   本地训练: local_epochs = 10。

两个并行的实验:
1.  V11-Linear: 服务器执行线性退火 (annealing_factor 从1.0线性衰减至0.0)。
2.  V11-Cosine: 服务器执行余弦退火 (annealing_factor 从1.0按余弦曲线衰减至0.0)。

#### 最终性能对比表格 (Round 9)

| 实验版本 | 退火策略 | Lambda 调度 | 最终准确率 |
| :--- | :--- | :--- | :--- |
| V11-Linear | 线性 | 强制衰减 | 38.35% |
| V11-Cosine | 余弦 | 强制衰减 | 38.51% |
| (历史最佳) V7 | 线性 | 强制衰减 | *~40.41%* |
| (历史次佳) V9 | 线性 | 强制衰减 | *~39.83%* |

在V11框架下，余弦退火（38.51%）的表现略优于线性退火（38.35%），但两者都非常接近，且都略低于之前纯粹的V7/V9版本。这暗示了V10的在线学习机制在退火面前可能引入了不必要的复杂性。

---

### Lambda分析

我们来分析lambda的变化。

| Round | 退火系数 (线性/余弦) | V11-Linear λ (C0 / C1) | V11-Cosine λ (C0 / C1) | 
| :---: | :---: | :---: | :---: | 
| 0 | 1.0 / 1.0 | 11.9 / 12.3 | 11.9 / 12.3 | 
| 1 | 0.9 / 0.975 | 12.1 / 13.4 | 11.2 / 12.4 | 
| 2 | 0.8 / 0.904 | 13.0 / 14.0 | 11.5 / 12.5 | 
| 3 | 0.7 / 0.793 | 14.7 / 15.7 | 12.9 / 13.8 | 
| 4 | 0.6 / 0.654 | 16.4 / 17.4 | 15.1 / 16.0 | 
| 5 | 0.5 / 0.500 | 19.3 / 21.3 | 19.3 / 21.4 | 
| 6 | 0.4 / 0.345 | 22.5 / 25.4 | 26.0 / 29.3 | 
| 7 | 0.3 / 0.206 | 30.1 / 34.0 | 43.8 / 49.5 | 
| 8 | 0.2 / 0.095 | 42.0 / 47.8 | 87.9 / 100.1 | 
| 9 | 0.1 / 0.024 | 84.0 / 93.0 | 342.8 / 380.2 | 

意外地， lambda 的上升压倒了退火，使得最终实际有效的 lambda 几乎没有变化。在训练的最后阶段（Round 8-9），当外部的annealing_factor将align_loss项的重要性强制压低到接近0时，V10的在线学习机制开始极大地增加 lambda。

让我们回到V11-Rescaled的损失函数：

` L = (base_loss + λ_eff * align_loss * annealing_factor) + 正则项 `

其中 `λ_eff = exp(log_σ_local² - log_σ_align²)`

训练后期，当 annealing_factor 趋近于0时，无论 λ_eff 多大，第二项 `λ_eff * align_loss * annealing_factor` 的值和梯度都接近于0。align_loss不再对模型权重产生影响。log_σ_align 这个参数的梯度，主要来自于align_loss项。当这一项被屏蔽后，log_σ_align 几乎收不到任何有效的梯度信号来阻止它变化。那么，损失函数中对 sigma 参数有影响的只剩下正则项 log(σ_local²) + log(σ_align²) 。优化器为了最小化这个正则项，会倾向于让 log_σ 的值变得越小越好（趋于负无穷）。log_σ_local 仍然能从 base_loss 获得梯度，所以它保持相对稳定。但log_σ_align 被松绑了，优化器为了最小化正则项，会疯狂地将它的值推向负无穷。这导致 λ_eff = exp(log_σ_local² - log_σ_align²) 中的 - log_σ_align² 这一项急剧增大。结果，Effective Lambda 爆炸性增长。

V11的学习机制，在没有 align_loss 的有效梯度约束后，其内部的优化目标（最小化正则项）导致了 lambda 值的失控。退火应该不影响梯度反向传播才对，应该在 sigma 反向传播梯度计算之后再乘上 annealing_factor。

我们需要一种方法，让：
1.  模型权重 W 的更新，使用被退火因子调节过的对齐损失。
2.  sigma 参数的更新，使用未经调节的、原始的对齐损失。

这需要一种更优雅的实现。为了达到这个目的，我们将使用元学习 (Meta-Learning) 或双层优化 (Bilevel Optimization)的思想，通过.detach()来精确地控制梯度流，从而在一个 loss.backward 调用中实现两个不同的优化目标。

我们借鉴以下研究的思路：

Forward and Reverse Gradient-Based Hyperparameter Optimization, Franceschi, L., et al. (ICML 2017), arXiv:1703.01785 [stat.ML], https://doi.org/10.48550/arXiv.1703.01785  
这篇论文是梯度式超参数优化领域的奠基之作。它系统地阐述了如何通过计算梯度来优化那些通常需要手动调整的超参数。我们的V12可以被看作是这篇论文中提出思想的一个在线、一步近似 (one-step approximation) 的特例。

Forward and Reverse Gradient-Based Hyperparameter Optimization, Franceschi, L., et al. (ICML 2017), arXiv:1703.01785 [stat.ML], https://doi.org/10.48550/arXiv.1703.01785  
这篇论文系统地阐述了两种计算超参数梯度的主要方法：反向模式（Implicit Function Theorem）和前向模式（Forward Mode Differentiation）。V12 实现可以被看作是一种计算成本极低、单步（one-step）展开的反向模式近似。
