# lambda debug

在实验中的某个已不可考证的环节，我们曾猜测提高 lambda 可以提升效果，因此 alpha=0.1 和 0.05 的 lambda 被从5提升到了20。然而，尽管这个优化没有被采纳到 V7 ，在之后的实验中我们既没有把 alpha=0.3 和 0.5 的 lambda 提升到 20 ，也没有把 alpha=0.1 和 0.05 的 lambda 恢复到5。这个事实对于之前得出的一些实验结论提出了挑战。  

除此之外，我们意外发现在之前的所有实验中，服务器端的 IFFI 都没有被启用。  

考虑到这两点，我们认为重新实验是有必要的。  


### **Alpha = 0.05**

#### **Algorithm: OneshotOursV7 (DRCL with ETF Anchors and Lambda Annealing) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV7+Simple | 0.3111 | 0.000500 | 1.00230 |
| 10 | OursV7+Simple | 0.4850 | 0.001434 | 0.94234 |
| 20 | OursV7+Simple | 0.5680 | 0.002026 | 0.88015 |
| 30 | OursV7+Simple | 0.6212 | 0.002508 | 0.81882 |
| 40 | OursV7+Simple | 0.6484 | 0.002924 | 0.76189 |
| 50 | OursV7+Simple | 0.6777 | 0.003293 | 0.70978 |

#### **Algorithm: OneshotOursV6 (DRCL and Lambda Annealing) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV6+Simple | 0.3007 | 0.000510 | 1.00220 |
| 10 | OursV6+Simple | 0.5093 | 0.001431 | 0.94247 |
| 20 | OursV6+Simple | 0.6019 | 0.002027 | 0.87989 |
| 30 | OursV6+Simple | 0.6365 | 0.002508 | 0.81893 |
| 40 | OursV6+Simple | 0.6564 | 0.002923 | 0.76179 |
| 50 | OursV6+Simple | 0.6817 | 0.003296 | 0.70938 |

#### **Algorithm: OneshotOurs (V4) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV4+Simple | 0.2829 | 0.000509 | 1.00606 |
| 10 | OursV4+Simple | 0.4973 | 0.001430 | 1.00622 |
| 20 | OursV4+Simple | 0.5784 | 0.002012 | 1.00636 |
| 30 | OursV4+Simple | 0.6238 | 0.002477 | 1.00648 |
| 40 | OursV4+Simple | 0.6384 | 0.002880 | 1.00661 |
| 50 | OursV4+Simple | 0.6697 | 0.003234 | 1.00672 |

#### **Algorithm: OneshotOursV6 (DRCL and Lambda Annealing) + Advanced IFFI Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV6+Advanced | 0.2520 | 0.000510 | 1.00220 |
| 10 | OursV6+Advanced | 0.5121 | 0.001431 | 0.94247 |
| 20 | OursV6+Advanced | 0.6141 | 0.002027 | 0.87989 |
| 30 | OursV6+Advanced | 0.6736 | 0.002508 | 0.81893 |
| 40 | OursV6+Advanced | 0.6975 | 0.002923 | 0.76179 |
| 50 | OursV6+Advanced | 0.7211 | 0.003296 | 0.70938 |

#### **Algorithm: OneshotOurs (V4) + Advanced IFFI Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV4+Advanced | 0.2445 | 0.000509 | 1.00606 |
| 10 | OursV4+Advanced | 0.4949 | 0.001430 | 1.00622 |
| 20 | OursV4+Advanced | 0.5916 | 0.002012 | 1.00636 |
| 30 | OursV4+Advanced | 0.6447 | 0.002477 | 1.00648 |
| 40 | OursV4+Advanced | 0.6590 | 0.002880 | 1.00661 |
| 50 | OursV4+Advanced | 0.6978 | 0.003234 | 1.00672 |

### **Alpha = 0.1**

#### **Algorithm: OneshotOursV7 (DRCL with ETF Anchors and Lambda Annealing) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV7+Simple | 0.3158 | 0.000556 | 1.00131 |
| 10 | OursV7+Simple | 0.5620 | 0.001603 | 0.92845 |
| 20 | OursV7+Simple | 0.6596 | 0.002283 | 0.85353 |
| 30 | OursV7+Simple | 0.7161 | 0.002830 | 0.78167 |
| 40 | OursV7+Simple | 0.7435 | 0.003305 | 0.71536 |
| 50 | OursV7+Simple | 0.7686 | 0.003730 | 0.65452 |

#### **Algorithm: OneshotOursV6 (DRCL and Lambda Annealing) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV6+Simple | 0.3080 | 0.000558 | 1.00132 |
| 10 | OursV6+Simple | 0.5447 | 0.001607 | 0.92906 |
| 20 | OursV6+Simple | 0.6453 | 0.002291 | 0.85352 |
| 30 | OursV6+Simple | 0.7015 | 0.002816 | 0.78187 |
| 40 | OursV6+Simple | 0.7343 | 0.003315 | 0.71540 |
| 50 | OursV6+Simple | 0.7595 | 0.003742 | 0.65502 |

#### **Algorithm: OneshotOurs (V4) + Simple Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV4+Simple | 0.3132 | 0.000554 | 1.00604 |
| 10 | OursV4+Simple | 0.5465 | 0.001613 | 1.00621 |
| 20 | OursV4+Simple | 0.6534 | 0.002278 | 1.00637 |
| 30 | OursV4+Simple | 0.7024 | 0.002816 | 1.00651 |
| 40 | OursV4+Simple | 0.7321 | 0.003283 | 1.00664 |
| 50 | OursV4+Simple | 0.7610 | 0.003697 | 1.00676 |

#### **Algorithm: OneshotOursV7 (DRCL with ETF Anchors and Lambda Annealing) + Advanced IFFI Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV7+Advanced | 0.2879 | 0.000554 | 1.00604 |
| 10 | OursV7+Advanced | 0.5575 | 0.001613 | 1.00621 |
| 20 | OursV7+Advanced | 0.6657 | 0.002278 | 1.00637 |
| 30 | OursV7+Advanced | 0.7067 | 0.002816 | 1.00651 |
| 40 | OursV7+Advanced | 0.7447 | 0.003283 | 1.00664 |
| 50 | OursV7+Advanced | 0.7720 | 0.003703 | 1.00676 |

#### **Algorithm: OneshotOursV6 (DRCL and Lambda Annealing) + Advanced IFFI Server Aggregation**

| Round | Algorithm | Accuracy | Model Variance (Mean) | G-Protos Std |
| :---: | :---: | :---: | :---: | :---: |
| 0 | OursV6+Advanced | 0.2977 | 0.000558 | 1.00132 |
| 10 | OursV6+Advanced | 0.5587 | 0.001613 | 0.92906 |
| 20 | OursV6+Advanced | 0.6614 | 0.002278 | 0.85352 |
| 30 | OursV6+Advanced | 0.7162 | 0.002818 | 0.78187 |
| 40 | OursV6+Advanced | 0.7536 | 0.003285 | 0.71540 |
| 50 | OursV6+Advanced | 0.7789 | 0.003703 | 0.65502 |