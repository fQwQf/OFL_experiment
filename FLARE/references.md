# FLARE Paper References with DOI/arXiv

本文档列出论文中所有参考文献的真实出处。

---

## ✅ 已验证文献 (真实存在)

### 核心方法论

| 引用 | 标题 | 出处 | DOI/arXiv |
|------|------|------|-----------|
| **Kendall et al., 2018** | Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | CVPR 2018 | **DOI:** [10.1109/CVPR.2018.00781](https://doi.org/10.1109/CVPR.2018.00781) |
| **Papyan et al., 2020** | Prevalence of neural collapse during the terminal phase of deep learning training | PNAS 117(40) | **DOI:** [10.1073/pnas.2015509117](https://doi.org/10.1073/pnas.2015509117) |
| **Li et al., 2023** | No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier | ICCV 2023 | **DOI:** [10.1109/ICCV51070.2023.00490](https://doi.org/10.1109/ICCV51070.2023.00490) |
| **Franceschi et al., 2017** | Forward and Reverse Gradient-Based Hyperparameter Optimization | ICML 2017 | **arXiv:** [1703.01785](https://arxiv.org/abs/1703.01785) |
| **Yu et al., 2020** | Gradient Surgery for Multi-Task Learning (PCGrad) | NeurIPS 2020 | **arXiv:** [2001.06782](https://arxiv.org/abs/2001.06782) |

### 联邦学习基础

| 引用 | 标题 | 出处 | DOI/arXiv |
|------|------|------|-----------|
| **McMahan et al., 2017** | Communication-Efficient Learning of Deep Networks from Decentralized Data | AISTATS 2017 | **arXiv:** [1602.05629](https://arxiv.org/abs/1602.05629) |
| **Guha et al., 2019** | One-Shot Federated Learning | arXiv 2019 | **arXiv:** [1902.11175](https://arxiv.org/abs/1902.11175) |
| **Li et al., 2020 (FedProx)** | Federated Optimization in Heterogeneous Networks | MLSys 2020 | **arXiv:** [1812.06127](https://arxiv.org/abs/1812.06127) |
| **Karimireddy et al., 2020 (SCAFFOLD)** | SCAFFOLD: Stochastic Controlled Averaging for Federated Learning | ICML 2020 | **arXiv:** [1910.06378](https://arxiv.org/abs/1910.06378) |

### Prototype-based FL

| 引用 | 标题 | 出处 | DOI/arXiv |
|------|------|------|-----------|
| **Tan et al., 2022 (FedProto)** | FedProto: Federated Prototype Learning across Heterogeneous Clients | AAAI 2022 | **arXiv:** [2105.00243](https://arxiv.org/abs/2105.00243) |
| **Zhang et al., 2024 (FedTGP)** | Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning | AAAI 2024 | **arXiv:** [2401.03230](https://arxiv.org/abs/2401.03230) |
| **Liu et al., 2024 (FedLPA)** | One-shot Federated Learning with Layer-Wise Posterior Aggregation | NeurIPS 2024 | **arXiv:** [2310.00339](https://arxiv.org/abs/2310.00339) |

---

## ✅ FAFI 参考文献 (已验证)

| 引用 | 标题 | 出处 | 链接 |
|------|------|------|------|
| **Zeng et al., 2025** | Does One-shot Give the Best Shot? Mitigating Model Inconsistency in One-shot Federated Learning | ICML 2025 (under review) | **OpenReview:** [2XvF67vbCK](https://openreview.net/forum?id=2XvF67vbCK) |

**作者:** Hui Zeng, Wenke Huang, Tongqing Zhou, Xinyi Wu, Guancheng Wan, Yingwen Chen, Zhiping Cai

```bibtex
@inproceedings{zeng2025oneshot,
  title={Does One-shot Give the Best Shot? Mitigating Model Inconsistency in One-shot Federated Learning},
  author={Zeng, Hui and Huang, Wenke and Zhou, Tongqing and Wu, Xinyi and Wan, Guancheng and Chen, Yingwen and Cai, Zhiping},
  booktitle={ICML},
  year={2025}
}
```

---

## 论文中References部分的修订建议

```bibtex
@inproceedings{kendall2018multi,
  title={Multi-task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics},
  author={Kendall, Alex and Gal, Yarin and Cipolla, Roberto},
  booktitle={CVPR},
  pages={7482--7491},
  year={2018},
  doi={10.1109/CVPR.2018.00781}
}

@article{papyan2020prevalence,
  title={Prevalence of neural collapse during the terminal phase of deep learning training},
  author={Papyan, Vardan and Han, X. Y. and Donoho, David L.},
  journal={PNAS},
  volume={117},
  number={40},
  pages={24652--24663},
  year={2020},
  doi={10.1073/pnas.2015509117}
}

@inproceedings{li2023nofear,
  title={No Fear of Classifier Biases: Neural Collapse Inspired Federated Learning with Synthetic and Fixed Classifier},
  author={Li, Zexi and Shang, Xinyi and He, Rui and Lin, Tao and Wu, Chao},
  booktitle={ICCV},
  pages={5296--5306},
  year={2023},
  doi={10.1109/ICCV51070.2023.00490}
}

@inproceedings{franceschi2017forward,
  title={Forward and Reverse Gradient-Based Hyperparameter Optimization},
  author={Franceschi, Luca and Donini, Michele and Frasconi, Paolo and Pontil, Massimiliano},
  booktitle={ICML},
  year={2017}
}

@inproceedings{yu2020gradient,
  title={Gradient Surgery for Multi-Task Learning},
  author={Yu, Tianhe and Kumar, Saurabh and Gupta, Abhishek and Levine, Sergey and Hausman, Karol and Finn, Chelsea},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{mcmahan2017communication,
  title={Communication-Efficient Learning of Deep Networks from Decentralized Data},
  author={McMahan, Brendan and Moore, Eider and Ramage, Daniel and Hampson, Seth and Arcas, Blaise Aguera y},
  booktitle={AISTATS},
  year={2017}
}

@article{guha2019one,
  title={One-Shot Federated Learning},
  author={Guha, Neel and Talwalkar, Ameet and Smith, Virginia},
  journal={arXiv preprint arXiv:1902.11175},
  year={2019}
}

@inproceedings{li2020fedprox,
  title={Federated Optimization in Heterogeneous Networks},
  author={Li, Tian and Sahu, Anit Kumar and Zaheer, Manzil and Sanjabi, Maziar and Talwalkar, Ameet and Smith, Virginia},
  booktitle={MLSys},
  year={2020}
}

@inproceedings{karimireddy2020scaffold,
  title={SCAFFOLD: Stochastic Controlled Averaging for Federated Learning},
  author={Karimireddy, Sai Praneeth and Kale, Satyen and Mohri, Mehryar and Reddi, Sashank and Stich, Sebastian and Suresh, Ananda Theertha},
  booktitle={ICML},
  pages={5132--5143},
  year={2020}
}

@inproceedings{tan2022fedproto,
  title={FedProto: Federated Prototype Learning across Heterogeneous Clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  booktitle={AAAI},
  volume={36},
  number={8},
  pages={8432--8440},
  year={2022}
}

@inproceedings{zhang2024fedtgp,
  title={FedTGP: Trainable Global Prototypes with Adaptive-Margin-Enhanced Contrastive Learning for Data and Model Heterogeneity in Federated Learning},
  author={Zhang, Jianqing and Liu, Yang and Hua, Yang and Cao, Jian},
  booktitle={AAAI},
  year={2024}
}
```

---

## 总结

- **12/12 文献已验证** 存在于公开出版物或OpenReview
- 所有参考文献均可追溯到真实来源
