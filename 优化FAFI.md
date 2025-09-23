# 优化FAFI

针对于FAFI，有两套创新方案：  

## 方案1：基于FAFI的修补  

### ​1. 改进负样本采样策略，以应对 Non-IID 数据挑战 

FAFI 中的对比学习损失函数旨在拉近同语义样本的特征，推开不同语义样本的特征。在 Non-IID 场景下，客户端本地批次中的负样本多样性可能不足，或者某些类别在本地数据中稀缺，导致对比学习的鉴别力下降。  
可以引入更高级的负样本采样策略，以确保即使在 Non-IID 环境下也能为对比学习提供丰富且信息量大的负样本。  

- ​动态负样本队列/内存库 (Dynamic Negative Queue/Memory Bank)：​​（已证明有负效果）  
客户端在本地维护一个特征表示的队列或内存库，存储来自历史批次或不同客户端的特征，作为负样本源。这样可以打破本地批次的限制，提供更广泛的负样本集，增强对比学习的有效性。  
Sun, R., Guo, S., Guo, J., Li, W., Zhang, X., Guo, X., & Pan, Z. (2024). GraphMoCo: A graph momentum contrast model for large-scale binary function representation learning. Neurocomputing, 575, 127273. https://doi.org/10.1016/j.neucom.2024.127273  
​
- 跨客户端负样本共享（通过服务器协助）：​​ 服务器在聚合阶段可以利用从客户端上传的特征信息（在满足隐私要求的前提下），构建一个全局的负样本池。客户端在本地训练时可以请求服务器提供一部分全局负样本来辅助本地对比学习，但这需要小心设计以避免隐私泄露和通信开销。  
Wang, Q., Chen, S., Wu, M., & Li, X. (2025). Digital Twin-Empowered Federated Incremental Learning for Non-IID Privacy Data. IEEE Transactions on Mobile Computing, 24(5), 3860–3877. https://doi.org/10.1109/tmc.2024.3517592  
问题在于这种机制必须要有客户端和服务端的反复通信，显然违背了 one-shot FL “仅一轮交互”的核心原则。不过，如果服务器可以获取一部分公共数据，或许可以在开始之前依据这些数据构建一个固定的公共负样本池，并一次性下发给客户端，作为本地对比学习的辅助负样本源。然而效果可能会受影响。  
或者，客户端在本地训练完成后，仅上传其特征表示，服务器在聚合时才利用这些信息构建一个全局的“负样本知识”，用于最终模型的评估或未来迭代的初始化（但不直接影响本次训练）。  

- 基于硬负样本挖掘（Hard Negative Mining）：​​ 识别那些模型难以区分的负样本，并赋予它们更高的权重，从而迫使模型学习更细粒度的特征表示。这可以与上述队列/内存库结合使用。  
Dong, H., Long, X., & Li, Y. (2024). Synthetic Hard Negative Samples for Contrastive Learning. Neural Processing Letters, 56(1). https://doi.org/10.1007/s11063-024-11522-2  

### ​2. 引入对 Non-IID 更具鲁棒性的对比损失函数：​​  

传统的 InfoNCE 损失在 Non-IID 数据下可能受到限制。  
可以​ 探索使用或修改对比损失函数，使其在类别不平衡或类别稀缺时表现更好。  
针对 Non-IID 数据中存在的类别不平衡问题，可以设计一种对每个类别提供等效监督信号的对比损失，例如通过调整每个类别正负样本的权重或使用平衡采样。例如，Stratify方法通过平衡采样策略来解决Non-IID数据下的数据异质性问题，而不是简单地对FedAvg进行增量调整，从而显著提升了模型性能。另外，FEDIC框架通过Calibrated Distillation来处理Non-IID和长尾分布数据，确保在类别不平衡的情况下也能获得良好的深度学习模型。  
Shang, X., Lu, Y., Cheung, Y., & Wang, H. (2022). FEDIC: Federated Learning on Non-IID and Long-Tailed Data via Calibrated Distillation (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2205.00172  
Wong, H. Y., Lim, C. K., & Chan, C. S. (2025). Stratify: Rethinking Federated Learning for Non-IID Data through Balanced Sampling (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2504.13462  

### 3. 结合其他无监督对齐技术​

虽然对比学习很强大，但结合其他机制可能带来互补优势。  
FAFI的SupConLoss 是一种样本级别的对齐。对于模型间不一致，FAFI试图在推理阶段通过IFFI来弥补这个问题，但没有在训练阶段主动解决它。除了样本级别的对比，可以额外引入一个正则化项，直接最小化不同客户端特征提取器输出的全局特征分布差异。  
Jing, C., Huang, Y., Zhuang, Y., Sun, L., Xiao, Z., Huang, Y., & Ding, X. (2023). Exploring personalization via federated representation Learning on non-IID data. Neural Networks, 163, 354–366. https://doi.org/10.1016/j.neunet.2023.04.007  
然而，这需要谨慎设计，因为它可能需要客户端共享更多信息或服务器进行更复杂的计算。并且，需要客户端在训练时知晓实时的、由当前其他客户端数据汇集而成的全局分布信息，并据此进行对齐，那么它就会像负样本共享一样，需要额外的客户端-服务器交互，违背了oneshot FL。不过如果客户端对齐的目标是一个预定义的、固定的全局参考分布（例如，在One-shot FL开始前，服务器基于公共数据集或过往经验确定并一次性下发），而不需要实时获取其他客户端的聚合信息，那么就可以避免冲突。  

## 方案2：创新架构

### 理论基础

大量研究指出：深度神经网络的不同层次具有显著不同的泛化特性和异构性敏感度。  
Qi, G., Qu, Z., Lyu, S.-H., Jia, N., & Ye, B. (2024). Personalized Federated Learning with Feature Alignment via Knowledge Distillation. In Pacific Rim International Conference on Artificial Intelligence, 121-133, 2024.  
Chen, J., Ding, L., Yang, Y., & Xiang, Y. (2023). Active diversification of head-class features in bilateral-expert models for enhanced tail-class optimization in long-tailed classification. Engineering Applications of Artificial Intelligence, 126, 106982. https://doi.org/10.1016/j.engappai.2023.106982  

具体而言：  
浅层特征(Base)：主要学习低级视觉模式，具有较强的跨域泛化能力  
Dong, Z., Niu, S., Gao, X., & Shao, X. (2024). Coarse-to-fine online latent representations matching for one-stage domain adaptive semantic segmentation. Pattern Recognition, 146, 110019. https://doi.org/10.1016/j.patcog.2023.110019  
深层特征(Head)：高度任务特定化，对数据分布变化极其敏感  
Chen, J., Ding, L., Yang, Y., & Xiang, Y. (2023). Active diversification of head-class features in bilateral-expert models for enhanced tail-class optimization in long-tailed classification. Engineering Applications of Artificial Intelligence, 126, 106982. https://doi.org/10.1016/j.engappai.2023.106982  

这一洞察启发我们提出 Adaptive Base-Head Splitting 策略：根据客户端数据分布的相似性程度，动态决定模型的最优分离点，从而实现"共享通用知识，保持个性化特征"的平衡。  

### 技术框架设计

1. 自适应分离算法  
核心思想：通过分析客户端间的数据分布相似性，动态确定最优的模型分离策略。在联邦学习中，基于模型相似性或数据相似性的客户端聚类已广泛应用于处理数据异构性。  
Chen, J., Li, M., & He, X. (2025). FedCK:addressing label distribution skew in federated learning via clustering-efficient and knowledge distillation. The Journal of Supercomputing, 81(14). https://doi.org/10.1007/s11227-025-07728-3  
Li, Z., Ohtsuki, T., & Gui, G. (2023). Communication Efficient Heterogeneous Federated Learning based on Model Similarity. In 2023 IEEE Wireless Communications and Networking Conference (WCNC), 1-5, 2023.  
可以引入根据平均相似性，动态调整共享层数，从而决定模型的最优分离点。这是一种新颖的自适应机制。这种动态调整的优势在于，当客户端数据相似性高时，可以共享更多的层以增强全局模型的泛化能力；当相似性低时，则减少共享层数以更好地适应本地个性化需求。  
Liu, Z., Luo, Y., Zhu, T., Chen, Z., Mao, T., Pi, H., & Lin, Y. (2025). FedBH: Efficient federated learning through shared base and adaptive hyperparameters in heterogeneous systems. Computer Communications, 239, 108190. https://doi.org/10.1016/j.comcom.2025.108190  
问题在于，如果“动态决定”意味着在客户端本地训练期间，服务器需要根据客户端数据动态计算最佳分离点，并反馈给客户端进行调整，那就会产生多轮交互，从而违反 oneshot FL 的特性。
不过也有折中的方案：  
    1. 在正式开始之前，可以有一个非常轻量级的前置交互阶段（例如，客户端上传少量元数据或统计信息），服务器据此计算并一次性确定一个全局或局部最佳分离点，然后下发给客户端，客户端在本地训练时便使用这个固定的分离点。但这可能会增加初始设置的复杂性，且前置交互阶段也应尽量精简以符合OFL精神。  
    2. 客户端完全在本地，根据自己的数据特征，自行决定分离点，不依赖服务器的实时反馈。客户端上传其训练好的模型/特征，服务器在聚合时才根据这些信息决定一个“最佳分离策略”用于最终模型构建。这种情况下，客户端本地训练仍然是一次性的、独立的，不冲突。这避免了交互，但可能牺牲全局最优性。

2. 差异化聚合策略  
采用特征对齐 + 知识蒸馏的混合策略。可以延续FAFI的思路。  

3. 动态原型更新机制  
为解决FAFI静态原型的局限性，设计自适应原型更新策略，通过考虑客户端可靠性和类别置信度进行加权更新，以更好地适应数据变化和噪声。  
Pei, F., Xie, Y., Shi, M., & Xu, T. (2025). Adaptive aggregation for federated learning using representation ability based on feature alignment. Knowledge-Based Systems, 318, 113560. https://doi.org/10.1016/j.knosys.2025.113560  
在此处，这种“动态”是单轮交互中更复杂的智能处理。比如如服务器在聚合本地原型时，可以采用加权策略（如基于客户端可靠性、数据量或模型性能），使全局原型更好地反映整体或更重要的客户端。动态更新机制可以在一定程度上反映客户端的局部数据特征，有助于学习更具区分度的个性化原型，从而提升个性化联邦学习的性能。  
Xu, J., Tong, X., & Huang, S.-L. (2023). Personalized Federated Learning with Feature Alignment and Classifier Collaboration (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2306.11867  