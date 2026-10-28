# 论文差异清单 — MR-MoE 自实现 vs arXiv:2606.12240v1（S2-gate-checks ④，D4 冻结）

- 论文：Zong, Boker, Eldardiry. *Multi-Rate Mixture of Experts for Accelerating Liquid
  Neural Network Training.* arXiv:2606.12240v1 [cs.LG] 2026-06-10（本地 PDF
  G:/research-papers/2606.12240.pdf，19 页）。
- 本清单口径：凡论文未给/未述而本实现自行冻结的项，逐条记录「论文怎么说 → 我们怎么定 →
  为什么」；声称范围=分析范围（design §1.3），不声称复现论文绝对数值。
- 关联产物：experiments/mr_moe.py（实现）、experiments/data_pipeline.py（数据）、
  results/operator_bench.md（算子）、results/ncps_tau_probe.json（接口）。

## 1. 论文已明确、本实现沿用（无差异）

| 项 | 论文 | 本实现 |
|---|---|---|
| 架构骨架 | Eq.8-11：K 个固定时间常数专家 + softmax 门控 + 加权读出 | 同（mr_moe.MRMoE） |
| K | 3（§3.3「K=3 experts」） | 3（D4） |
| 专家隐藏维 | 1500 neurons（§3.3） | 1500（D4） |
| τ 关系 | τ₁≪τ₂≪…≪τ_K（Eq.8，具体值未给） | 1:10:100（dt=1h 网格，D4 冻结） |
| 优化器/学习率 | Adam，lr=1e-3（§3.3） | Adam lr=1e-3（D4） |
| 损失 | BCE（§3.5.1 述 LSTM 用 BCE；MR-MoE 同设定） | BCEWithLogits（D4） |
| 数据/预处理 | Moor et al. 2023 败血症；normalization + forward filling；Che 2018 划分（§3.1） | 回退源 PhysioNet-CinC 2019（S2 实测 Moor 不可得，见 results/data_check.json）；ffill 保留（D3） |
| 评测指标 | AUROC / AUPRC（§3.4） | 同，另按 D8 分入院级/小时级两层 |
| 无 attention 基线 | §3.2 区分 MR-MoE（无 attention）与 MR-MoE-Attention | 基线取无 attention 的 Eq.8 基础 MR-MoE（D4；attention 变体是另一个模型，不进本轮） |

## 2. 论文未给、本实现冻结（单变量纪律内自定，逐条留痕）

| # | 项 | 论文状态 | 本实现冻结值 | 理由 |
|---|---|---|---|---|
| 1 | τ 数值 | Eq.8 只给序关系，未给数 | τ=(1,10,100)（dt=1h 下 1h/10h/100h） | 论文无值可复现；取数量级分隔的三档，冻结后 A/B 两臂共享（D10） |
| 2 | f_k 胞体形式 | 「nonlinear dynamical function」（Eq.10），未给形式 | ncps 0.0.2 CfC default 胞体（闭式连续解） | 论文无代码（checklist #5「Not yet」）；CfC 为 ncps 标准 LNN 胞体；LTC 路径因 hidden 1500 在 8G OOM 被实测排除（results/operator_bench.md §5） |
| 3 | batch size | 「fixed across all experiments」，未给值 | 64（D4） | 8G 显存与 D9 计时约束下的冻结值 |
| 4 | 门控网络尺寸 | 「a small MLP」，未给尺寸 | Linear(d,64)+tanh+Linear(64,K)，softmax | 「small」的操作化；容量对三专家门控足够 |
| 5 | 读出层 | Eq.5/17：C_k 线性读出（y=Cx） | 每专家 Linear(1500,1) + 末端 sigmoid（经 BCEWithLogits 数值稳定路径） | sigmoid 为概率输出所需；与论文线性读出差一个链接函数，记录在案 |
| 6 | epochs | 未给 | pilot E=6（2/2/1/1）；full E=20（8/6/4/2）（D2） | 论文无值；按 D9 算力纪律冻结 |
| 7 | 种子/统计 | 单次运行，无误差棒（checklist #7「No」） | 3 种子固定表 {0,1,2}，mean±std（D6） | 补论文缺失的多种子统计（课题贡献点之一） |
| 8 | 序列窗口/网格 | 未述（只说 multivariate clinical time-series） | 小时网格 dt=1、前 72h 截断（D3） | 与 6h-ahead 预测目标和 ICU 早期预测场景匹配；PhysioNet 2019 数据原生小时步 |
| 9 | 特征集 d | 未给 | PhysioNet 2019 的 34 个时变临床特征（8 生命体征+26 实验室）；Age/Gender/Unit/HospAdmTime/ICULOS 静态列不作输入通道 | 论文只说「vital signs and laboratory values」；常数通道对每小时预测无增量 |
| 10 | 缺失值处理 | forward filling（§3.1） | ffill + 缺失 mask 输入通道拼接（禁 0 填充裸奔，D3；taste 死法①） | 论文只述 ffill；mask 通道防实现层缺陷污染对照（design §1.3） |
| 11 | 归一化 | normalization，未给式 | min-max，仅 train 划分拟合（D3） | 防划分泄漏；论文未给式 |
| 12 | 数据划分 | Che et al. 2018 划分（§3.1） | PhysioNet 2019 未随数据提供官方划分文件 → 固定种子患者级随机划分 70/10/20 | Che 2018 为论文引用的划分依据；操作化留痕；严禁按小时切分（同患者跨划分=泄漏） |
| 13 | 标签定义 | 「predict the onset of sepsis」，未给公式 | PhysioNet 2019 随数据 SepsisLabel（挑战标准 6h-ahead：septic 患者 onset 前 6h 起 label=1） | D3 公式 y(t)=1[onset∈(t,t+6h]] 与挑战标注在 onset 后时段有界差（公式=0/挑战=1）；取挑战标注与官方评测口径一致，差异记录在案 |
| 14 | 单体 LNN 基线 | §3.5.2：单 LNN，hidden 1500，同训练设定 | 单 CfC hidden 1500，timespans=None（≡dt=1），线性+sigmoid 读出 | 论文未给其时间常数；均匀网格是与论文 A 臂 core(x) 语义一致的自然缺省 |

## 3. 实现环境差异

- ncps 0.0.2（本机在位）≠ 论文同款实现（论文未述框架版本；admission U5 记录本机
  0.0.2，非 0.0.7）。接口行为已实测（results/ncps_tau_probe.json）：固定 τ 走 (1,L)
  常向量库路径；B>1 的 (B,L) timespans 崩溃为 0.0.2 已知限制，用 (1,L) 注入绕行
  （与 manual-loop 数值恒等）。
- 论文 MR-MoE AUROC 0.61 / AUPRC 0.42（§3.5.4）显著低于 Moor 原文 0.846——论文自述
  预处理/划分必有差异。**本课题不以复现论文绝对数值为目标**（D4/U4），以「固定 τ 的
  MR-MoE 在本实现下优于单体 LNN」作方向性健全性（D5，对应论文 0.61>0.55 的方向）。

## 4. 声称边界（与 design §1.3 一致）

- 不声称「分阶段训练 MoE」机制新颖（EvoMoE arXiv:2112.14397 等为先例）。
- 贡献=论文 §4.1 点名缺口（多时间尺度专家解耦训练）上的受控单变量实验 + 多种子统计 +
  梯度冲突代理；所有上述自冻结项都是该受控实验的控制变量，不是贡献声称。
