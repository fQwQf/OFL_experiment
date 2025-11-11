# How to use Excel

我们该如何实现V12？  

退火的lambda曲线是当前已知最优的策略。V10能自适应地降低 lambda ，但最终稳定在一个非零的均衡点，未能进一步下降。而V11证明外部强制归零与内部学习机制冲突。

因此，我们不能强制lambda归零，而必须改变学习算法的内在激励机制，让它“自愿地、最优地”选择归零。

让我们回到V10为lambda（通过sigma参数）设计的损失函数：

` loss_for_sigma = (0.5/sigma_sq_local) * base_loss + (0.5/sigma_sq_align) * align_loss + 0.5 * (log(sigma_sq_local) + log(sigma_sq_align)) `

在训练后期，base_loss 和 align_loss 都已经收敛到一个较小但非零的稳定值。 base_loss 的存在，激励系统减小 sigma_sq_local（即增大1/sigma_sq_local的权重，等效于增大lambda）。align_loss 的存在，激励系统减小 sigma_sq_align（即增大1/sigma_sq_align的权重，也等效于增大lambda）。正则项的存在，则防止任何一个sigma变得过小（趋近于0）而导致数值不稳定。

优化器找到的lambda~9这个稳定点，正是在这个均衡状态下，对两个损失项和正则项的综合妥协。在这个数学框架下，没有任何内在动力驱使lambda必须降到0，因为只要align_loss还存在，学习算法就认为它有价值，需要分配一定的权重。

我们的目标是在不引入外部强制冲突的前提下，让学习算法在训练后期主动忽略 align_loss。

核心思想是修改 loss_for_sigma，引入一个基于训练进度的内部衰减因子，这个因子只作用于sigma的学习，而不直接干预模型权重的学习。这是一种元课程学习，我们会教模型的学习算法在不同阶段该关注什么。

#### V12 新的loss_for_sigma设计

我们将引入一个衰减函数 s(p)，其中 p 是全局训练进度。s(p) 在 p=0 时为1，在 p=1 时为0。

修改后的loss_for_sigma如下：

` loss_for_sigma_v13 = (0.5/sigma_sq_local) * base_loss.detach() + s(p) * (0.5/sigma_sq_align) * align_loss.detach() + 0.5 * (log(sigma_sq_local) + log(sigma_sq_align)) `

而用于更新模型权重的 loss_for_weights 保持不变：

` loss_for_weights = base_loss + effective_lambda * align_loss `

#### 预期结果

1.  训练初期 (p -> 0, s(p) -> 1):
    *   loss_for_sigma_v13 与原始的 loss_for_sigma 完全相同。
    *   lambda会学习到一个较高的值来优先降低巨大的初始align_loss。

2.  训练中期 (p -> 0.5, s(p) < 1):
    *   衰减因子 s(p) 开始生效。对于sigma的学习过程而言，align_loss的重要性被人为地降低了。
    *   即使align_loss本身的值没有变化，但由于其对loss_for_sigma_v13的贡献被s(p)削弱，优化器会开始倾向于稍微增大sigma_sq_align（即减小lambda），因为这样做带来的惩罚变小了。

3.  训练后期 (p -> 1, s(p) -> 0):
    *   s(p) 趋近于0，导致align_loss这一项在loss_for_sigma_v13中完全消失。
    *   此时sigma的学习目标变为最小化 (0.5/sigma_sq_local) * base_loss.detach() + 0.5 * (log(sigma_sq_local) + log(sigma_sq_align))
    *   为了最小化这个表达式，尤其是 log(sigma_sq_align) 这一项，优化器的选择就是将sigma_sq_align推向无穷大。
    *   当sigma_sq_align变得巨大时，effective_lambda = sigma_sq_local / sigma_sq_align 自然而然地、平滑地趋近于0。

通过这一改动，我们就将退火融入了自适应学习的框架中，创建了一个理论上更优越、行为上更符合期望的算法。