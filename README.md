# 分阶段训练多时间尺度液态专家（MR-MoE）：受控单点实验全记录

**一句话**：针对论文 arXiv:2606.12240v1 §4.1 点名的缺口——「多个时间尺度专家联合训练会互相干扰，作者建议 hierarchical/alternating 解耦训练但未验证」——我们在真实败血症数据（PhysioNet/CinC 2019）上做了预注册的受控单变量实验：分阶段训练臂 B vs 论文式联合训练臂 A。**主假设 H1 被证伪（falsified）**：分阶段臂在三项预注册门线上全部未过；机制假设 H2 维持 inconclusive。本仓库完整记录问题、做法、复现入口与全部读数，**不把负结果包装成支持性结论**。

- 结论口径：**H1 = falsified**；**H2(a) = inconclusive（遥测补齐后仍不呈预注册方向，记「不支持」）**；**H2(b) = 机制事实记录（不提升为 H2 支持）**；**H2 overall = inconclusive / not established**。
- 本仓库为本地复现仓库，**未做任何公开推送**（是否推送、以何种许可证推送，需用户明确授权后另行决定）。
- 每个数字都能回到 `results/` 下的原始文件；复现入口 = `results/_s6_recompute.py`（H1）与 `results/_d7fix_recompute.py`（D7/H2(a)），一条命令见下文「怎么跑」。

---

## 1. 问题：论文缺口与我们的问法

**论文缺口（主论文 §4.1 原文核实）**：MR-MoE（Multi-Rate Mixture of Experts，K 个固定时间常数 τ 的液态 CfC 专家 + softmax 门控）在单一优化过程里联合训练多个时间尺度专家会引入干扰；作者把「分层/交替式解耦训练」列为 future work，没有给实验。

**我们的问法（单点改动，预注册后可证伪）**：把训练流程从「联合」改成「按时间尺度分阶段解耦」——阶段一只训快专家 τ₁，阶段二训中/慢专家 τ₂/τ₃，阶段三只训门控与读出，阶段四全部参数解冻做短联合校准——在**同一模型、同一数据、同一超参**下，指标会不会更好、更稳？

**定位纪律**：不声称「分阶段训练 MoE」机制新颖（EvoMoE arXiv:2112.14397 等在先）；贡献 = 这个具体缺口上的受控单变量实验 + 多种子统计 + 梯度冲突代理。完整预注册协议见 `PROTOCOL.md`（假设 H1/H2、竞争解释 E1–E5、设计冻结 D1–D10、停止条件 F1–F5）。

## 2. 改动：实验设计（预注册冻结后执行）

**三个臂**（种子 {0,1,2}，full schedule E=20）：

| 臂 | 训练方式 | epoch 安排 |
|---|---|---|
| A | 论文式联合训练（全部参数单一优化过程） | 20 |
| B | 分阶段解耦 | 8/6/4/2（阶段四=2 epoch 全参数联合校准） |
| B-no-cal | 分阶段解耦、**去掉阶段四** | 8/6/4 = 18（与 B 的阶段 1–3 逐字相同，唯一差异=无联合校准；用于 H2(b)/E2 判据） |

**模型与数据**（冻结配置，论文未给项的自冻结值逐条留痕于 `results/baseline_diff.md`）：

- 模型：K=3 个固定 τ=(1,10,100) 的 CfC 液态专家，hidden 1500；门控 softmax(MLP64)；读出线性+sigmoid；Adam lr=1e-3；BCE；batch 64；无 attention。
- 数据：PhysioNet/CinC 2019 回退源（主源 Moor 2023 harmonized 需四库认证、本夜不可得，证据在 `results/data_check.json`）；小时网格 dt=1、前 72h 截断、前向填充 + 缺失 mask 输入通道（禁 0 填充裸奔）、min-max 仅 train 拟合；标签 = 6h-ahead Sepsis-3 onset。
- 规模：D9 阶梯定档 **2000 例/run**（全文语料下载实测 ~1.17 文件/s，8000 档 ETA 超窗，按预授权阶梯降档；禁合成数据）；划分 train/val/test = 1400/200/400，三臂共享子采样/划分/数据顺序/scaler（D10 单变量纪律）。
- 评测：主 = 入院级 AUROC/AUPRC（窗口概率最大值参评）；次 = 小时级。

## 3. 怎么跑（复现入口）

**一条命令**（仅标准库，不训练、不下载、不联网、不需要 GPU，任何 Python 3.8+ 均可）：

```bash
bash reproduce.sh
```

它做两件事，并把关键读数与本仓库冻结值逐项核对（不一致即非零退出）：

1. `results/_s6_recompute.py` —— 从 9 个原始 run JSON + CSV 复算三臂 mean±std（ddof=1）、配对差、H1 三门线，并校验 CSV↔JSON 一致性（≤5.1e-5）。
2. `results/_d7fix_recompute.py` —— 复算 D7 stage4 遥测修复的通过条件（33/33）、配置不变性（D10）、修复前后对照、性能 bit 级对齐（最大差 0.0）与 H2(a) 配对余弦。

手动单跑也行：

```bash
python results/_s6_recompute.py      # → results/_s6_recompute.json
python results/_d7fix_recompute.py   # → results/_d7fix_recompute.json（幂等，仅刷新 generated_at）
```

**想从头重跑训练（长路径，需要 GPU + 数据，不属于一条命令复现的范畴）**：

```bash
# 1) 下载 PhysioNet 2019 语料（公开，断点续跑）
python experiments/download_physionet2019.py
# 2) 跑九臂正式实验（每 epoch 原子 checkpoint；单 run 94–228s < D9 上限 720s）
python experiments/full_train.py --arm A --seed 0 --n-patients 2000
# 3) D7 修复单测（需 torch；18 项：AST 源码级 + 四 stage 计数 + stage4 三对 cos 功能级）
python experiments/_d7fix_unit_test.py
# 4) 复算（同上面一条命令）
bash reproduce.sh
```

环境：本批实验用 Python 3.11.9 + torch 2.10.0+cu128 + ncps 0.0.2（G:/ai/python11，RTX 5060 8G）。算子基准（CfC 前向反向全有限、梯度范数 0.034–0.223、峰值 549MiB、≈34.7s/epoch）见研究仓库 `results/operator_bench.md`；LTC 路径 hidden 1500 在 8G OOM 为已知备选墙（不影响本实验，F8 未触发）。

## 4. 结果表

### 4.1 主结果（H1；val，入院级为主；mean±std，ddof=1）

| 臂 | n | val 入院级 AUROC | val 入院级 AUPRC | val 小时级 AUROC | val 小时级 AUPRC |
|---|---|---|---|---|---|
| A（联合） | 3 | **0.8301±0.0136** | 0.2290±0.0624 | 0.7390±0.0058 | 0.0302±0.0017 |
| B（分阶段） | 3 | **0.7479±0.0346** | 0.1206±0.0201 | 0.6824±0.0427 | 0.0226±0.0049 |
| B-no-cal | 3 | 0.6409±0.0336 | 0.0679±0.0113 | 0.6464±0.0558 | 0.0205±0.0054 |

配对差（逐种子明细在 `results/full_joint_vs_staged.md`，原始浮点在 `results/full_runs/*.json`）：

| 配对 | val 入院级 AUROC | val 入院级 AUPRC | val 小时级 AUROC |
|---|---|---|---|
| **B−A（H1 方向）** | **−0.0822±0.0318**（3/3 种子负） | −0.1084±0.0567 | −0.0566±0.0376 |
| B−B-no-cal（阶段四效应，H2(b)/E2） | **+0.1070±0.0015**（3/3 正） | +0.0527±0.0100 | +0.0360±0.0165 |
| A−B-no-cal（参考） | +0.1893±0.0304 | +0.1611±0.0620 | +0.0926±0.0514 |

test 口径 B−A 入院级 AUROC = −0.0327±0.0238（仅描述性核对；预注册主门线用 validation）。

### 4.2 H1 预注册门线读数（design §六；三项同时满足才 supported）

| 门线 | 要求 | 实测 | 判定 |
|---|---|---|---|
| ① ΔAUROC(B−A) | ≥ +0.01 | −0.0822±0.0318 | **FAIL** |
| ② ΔAUPRC(B−A) | ≥ −0.01 | −0.1084±0.0567 | **FAIL** |
| ③ std(B) ≤ std(A) | 0.0136 | 0.0346 | **FAIL** |

|ΔAUROC|/配对差 std ≈ 2.588σ；pilot（E=6，2/2/1/1）同向为负 −0.1239±0.0529 → full 收敛至 −0.0822 但未翻正。**H1 终判 = falsified**（S6 正式 verdict：`results/final_verdict.md`）。

### 4.3 机制读数（H2）

**H2(b) / E2（阶段四联合校准效应）**：B−B-no-cal = +0.107033±0.001550（3/3 种子正，std 比种子噪声小两个量级）→ §二「B-no-cal 不能复现 B 全部增益」的读数方向成立；§三 E2 字面预测「B-no-cal 与 B 差异不显著」与 §二对同一读数的失败映射相反，系预注册笔误，已按 claim 科学含义重述判据（裁决记录在研究仓库 DECISIONS〔52〕）。E2 窄义获数据支持（阶段四 +0.107 是 B 臂内部唯一正向贡献；阶段一至三解耦相对 A 为 −0.189），但 **H1 已 falsified，B 没有增益可解释，该读数只作机制事实记录，不提升为 H2 支持**。

**H2(a)（梯度冲突；design §二：A 全程专家对余弦应比 B 阶段四更负）**：

S5 曾因遥测 bug 缺证（stage4 六个 epoch 日志全空），S6-final-verdict-redesign 修复后同 2000 例配置补跑 B×3（新 tag `full_B_seed{0,1,2}_d7fix`，不覆盖 S5 产物）。修复前后对照：

| 口径 | 修复前（S5） | 修复后（d7fix） |
|---|---|---|
| stage4 epoch 数 | 6（3 seed×2） | 6 |
| 逐 epoch logged batches | 0 / 22 | **22 / 22（6/6）** |
| logged batches 合计 | 0 | **132** |
| 定义良好余弦（三对） | 0（全 null） | **18（6×3）** |
| 通过条件（预注册） | — | **33/33 PASS** |

单变量纪律实证：d7fix 与 S5 配置逐字段相等，8 指标×3 seed 性能最大绝对差 = **0.0（bit 级复现）**——修复只动遥测、不动训练数学。

遥测补齐后的方向核对（A 全程 60 个 epoch 级值 vs B-stage4 6 个；配对差 A−B-stage4，mean±std）：

| 专家对 | 配对差 | 方向 |
|---|---|---|
| cos_e1_e2 | −0.004868±0.005645 | A 更负 |
| cos_e1_e3 | **+0.001202±0.004605** | **反向** |
| cos_e2_e3 | −0.001893±0.004788 | A 更负 |

9 个 seed×pair 单元 5 个为负、三对非全负、全部 |mean| ≪ std（种子噪声带内）；max|cos|：A 0.017018 vs B-stage4 0.004199（两臂均 ≈0）；A 末 2 epoch 稳健性视图同型（−0.004178/+0.000313/−0.002386）。**H2(a) = 不呈预注册方向，记 inconclusive（不支持）**。三条不包装纪律：修复前的缺失≠零值；2/3 对为负≠方向成立；≈0≠无冲突。design §二字面映射「(a) 不成立 → H2 证伪」与指令边界的张力记载在案（效应量比 22 batch/epoch 抽样噪声低两个量级、B-stage4 仅 6 个 epoch 级值 vs A 60，统计功效不足外推证伪，故维持 inconclusive 不升级）。详见 `results/d7_stage4_repair.md`。

**H2 overall = inconclusive / not established**：H2 以「若 H1 成立」为前提，H1 已 falsified。

### 4.4 其他在盘读数

- 九 run 全 done、无 NaN；单 run 94.1–227.8s（D9 上限 720s）；每 epoch 原子 checkpoint。
- 梯度范数比 τ₃/τ₁ = 0.846–6.567×（不爆，F3 未触发）；S5 全量 216 条定义良好余弦（A 180 + B stage2 18 + B-no-cal stage2 18）+ 修复后 B stage4 18 条；全量 max|cos| = 0.017018 ≈ 0。
- D5 sanity（pilot）：A 优于单体 LNN（val +0.1499 / test +0.1138），实现伪影解释被削弱。
- 数据：2000 例子采样 septic 率 0.063；val 200 例中每个 run 恰 9 个阳性入院。

## 5. 结论与证据边界（falsified 打包口径，逐条写明）

1. **H1 falsified**：冻结设置（tier=2000、τ=1:10:100、E=20、8/6/4/2、3 种子）下，分阶段臂未达预注册增益/稳定性门线。这是**假设层负结果**，不是「实验跑不通」——九个 run 全部完成、指标全部可算。结论只针对该冻结设置，不声称所有分阶段 MoE 都有害。
2. **设计固有混杂（必须随结论保留）**：B/A 专家更新预算比 ≈0.5 系 4:3:2:1 比例固有——A 每个专家训 20 epoch；B 的 τ₁ = 阶段一 8 + 阶段四 2 = 10 epoch（≈A 的 0.50），τ₂/τ₃ = 阶段二 6 + 阶段四 2 = 8 epoch（≈A 的 0.40）。**负差不能单独归因于「解耦本身有害」**；pilot→full 负差收敛（−0.124→−0.082）与该混杂的预期方向一致，但未翻正。
3. **小样本边界**：val 每 run 只有 **9 个阳性入院**（200 例）、3 种子共享同一验证划分、tier=2000。噪声底与效应同阶——「不优于」结论稳（3/3 同向、约 2.6σ 出门线），但「差多少」的精确值受限（test 口径 −0.0327 即弱于 val）。D7 遥测为 22 batch/epoch（design 原文 ≥200 口径在 2000 档不可行，已在案），cos 估计噪声底与效应同阶。
4. **不包装纪律**：本仓库不把负结果写成支持性结论；不把缺失当零值；不把机制事实记录（+0.107）说成 H2 得证；不倒因为果。
5. **已知问题（如实记录）**：S5 执行备注曾称「B stage2+stage4 已记录」与实际数据不符（实为 stage2 only，stage4 因组名前缀计数 bug 全 null）；S6 称已发布 researcher 裁决 event `89a94b5444e14a5897254062d89ed585`，盘上未检索到，裁决由判读单元直接落盘。两处均已在研究仓库 STATUS/DECISIONS 记载。

## 6. 论文协议与差异清单

- 预注册协议（假设/门线/竞争解释/设计冻结/停止条件）：`PROTOCOL.md`。
- 与论文 arXiv:2606.12240v1 的协议差异（14 项论文未给项的自冻结值 + 环境差异 + 声称边界）：`results/baseline_diff.md`。要点：τ=(1,10,100)（论文只给序关系）、batch 64、门控 MLP64、72h 窗、34 个时变特征 + mask 通道、患者级随机划分（论文引 Che 2018 但数据未随附划分文件）、6h-ahead 标签取挑战标注。**不以复现论文绝对数值为目标**（论文 MR-MoE AUROC 0.61 vs Moor 原文 0.846，其预处理/划分必有差异）；方向性健全性 = 固定 τ MR-MoE 优于单体 LNN（D5，pilot 已 PASS）。

## 7. 目录结构

```
.
├── README.md            # 本文件（问题/改动/怎么跑/结果表/结论与边界）
├── reproduce.sh         # 一条命令复现（仅标准库；核对关键读数 vs 冻结值）
├── PROTOCOL.md          # 预注册协议摘要（H1/H2/E1-E5/D1-D10/F1-F5 + 执行期偏差留痕）
├── experiments/         # 产生原始结果的代码（mr_moe/data_pipeline/full_train/pilot_train/下载器/D7 修复驱动+单测）
└── results/
    ├── full_joint_vs_staged.csv / .md   # A/B 六 run 汇总表（结论级 + 逐 run + 逐种子配对）
    ├── full_bnocal.csv                  # B-no-cal 三 run 表
    ├── full_runs/*.json                 # 12 个原始 run JSON（含 d7fix 新 tag 3 个）
    ├── full_progress.json               # 九 run 逐 epoch 进度（n_pos=9 实证）
    ├── full_gradient_conflict.json      # S5 D7 日志（修复前；B stage4 全 null）
    ├── full_gradient_conflict_d7fix.json# 修复后新 tag 日志（stage4 22/22、三对 cos 非 null）
    ├── _s6_recompute.py / .json         # H1 终判独立复算（入口 1）
    ├── _d7fix_recompute.py / .json      # D7 修复复算（入口 2）
    ├── final_verdict.md                 # S6 正式 verdict（H1 falsified / H2 inconclusive）
    ├── d7_stage4_repair.md              # D7 stage4 修复 + 补跑 + H2(a) 判读全文
    ├── baseline_diff.md                 # 论文协议与差异清单
    └── data_check.json                  # 数据源定影证据（Moor 不可得 → PhysioNet 2019 回退）
```

## 8. 已知限制与后续方向（不是结论，是未验清单）

按 F1 纪律（负结果落盘、不调参再追 H1），机制证据升级方向各带 30 分钟验法与失败条件，全文在研究仓库 `notes/RETRO.md`：

1. D7 原文口径回归（≥200 batch/epoch，需 ~16000 档重跑 A/B seed0）；失败=三对仍非全负或 |mean|≪std → H2(a) 定案不支持。
2. batch 级配对检验（落逐 batch cos + bootstrap CI）；失败=CI 跨 0 → 定案；CI 全负 → 弱支持升级。
3. 冲突干预探针（stage4 换 PCGrad 式消解跑 2 epoch）；验证 cos 遥测是否有分辨力。
4. 等专家预算 B′=18/18/2/2（去 B/A 0.5 混杂）、τ=(1,5,25) 敏感性、8000 档扩大验证——三条未验死，任一可排产。

## 9. 推送状态

- 本仓库只在本地 git（无 remote、无公开推送）。是否公开、用什么许可证，需用户明确授权后另行处理。
- 数据：PhysioNet/CinC 2019 公开数据（需遵守其使用条款）；模型代码为按论文自实现（ncps 0.0.2 CfC 胞体）。
- 研究过程全记录（判读、决策、复盘）在研究仓库内，不在本公开候选仓库范围内。
