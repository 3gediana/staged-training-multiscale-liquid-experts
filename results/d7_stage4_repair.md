# S6-final-verdict-redesign：D7 stage4 梯度遥测修复 + 同配置 B×3 补跑（H2(a) 证据缺口关闭判读）

- 单元：lab-executor 执行单元（kind=experiment，role=executor），2026-10-29 00:4x-00:5x
- 输入判据：`results/final_verdict.md/json`（H2(a) 证据边界+代码级根因）、`notes/researcher-interventions/b87fa36b2c62488dbf631cd70c62df01.md`（S6 判读：D7 补测优先级确认）、`design.md` §二 H2(a)/§四 D7、`results/_full_tier_decision.json`（2000 档定档）
- 实验记录：`results/run-20261029-004358/run.log` + `results/_d7fix_train.log`；复算：`results/_d7fix_recompute.py` → `results/_d7fix_recompute.json`（仅标准库、不训练）；单测：`experiments/_d7fix_unit_test.py`（18/18 PASS）

## 一、修了什么（根因→修复→防再发）

**根因（S5 判读〔52〕代码级定位，本单元读码复核属实）**：`experiments/full_train.py:398` 原以 param group 名前缀 `startswith("expert")` 计「联合可训专家组数」；`experiments/mr_moe.py:141-142` 的 `param_groups_for_stage(4)` 只有单组且名为 `"all"`（stage4=全部参数解冻联合校准）→ 计数 0 → `:410 log_grad=False` → S5 B 臂 stage4 六个 epoch（3 seed×2）`grad_logged_batches=0`、三对 cos 全 null。设计点名的 H2(a) 比较期（A 全程 vs B 阶段四）full 尺度无数据。

**修复（:398，按实际联合可训专家参数集合判定）**：

```python
trainable_ids = {id(p) for g in groups for p in g["params"]}
n_expert_groups = sum(1 for expert in model.experts
                      if any(id(p) in trainable_ids for p in expert.parameters()))
```

stage1=1 / stage2=2 / stage3=0 / stage4=3，与组名命名约定彻底解耦；A 臂不变（恒=3）。单测实证：旧组名前缀口径 stage4=0（复现 bug），新口径 stage4=3；stage4 全解冻后 `expert_gradient_stats` 三对 cos 全部非 null 且有限。

**防再发（每 epoch 非空断言，epoch 末/阶段切换处生效）**：`log_grad=True` 的 epoch 必须 ①`grad_logged_batches>0` 且 ②联合可训专家对余弦全部非 null 且有限（n=3 时即三对；stage2 的 n=2 时该联合对），否则 `AssertionError` 立即终止 run——S5 的 stage4 全 null 正因缺此断言而全程无声。本单元三跑零触发（`run.log` 无 `[d7-assert]`），断言存在性由单测 AST 级校验。

## 二、D7 stage4 修复前后对照

| 口径 | S5（修复前，`full_gradient_conflict.json`） | d7fix（修复后，`full_gradient_conflict_d7fix.json`） |
|---|---|---|
| stage4 epoch 数 | 6（3 seed×2） | 6 |
| 逐 epoch logged batches | 0 / 22 | **22 / 22（6/6 epoch）** |
| logged batches 合计 | 0 | **132** |
| 定义良好余弦（三对） | 0（全 null） | **18（6×3，逐 epoch 三对全非 null 有限）** |
| expert_grad_norm | None | 3/3 非 null（逐 epoch） |

通过条件（预注册）：每个 B seed 的两个 stage4 epoch 均 22/22 logged batches 且三对 cos 非 null——**3/3 seed 全满足（11 项检查×3 seed=33/33 PASS）**；stage2 行为与 S5 一致（6 epoch×22/22、cos_e2_e3 非 null）、stage1/3 仍不记录（无非联合专家对）。**修复生效，遥测缺口关闭。**

## 三、同 2000 例配置验证（D10 单变量纪律）

- 配置逐字段相等（d7fix vs S5，3/3 seed）：n_patients=2000、epochs=20（8/6/4/2）、lr=1e-3、batch=64、hidden=1500、τ=(1,10,100)、gate_hidden=64、seq_len=72、log_grad_every=1、子采样 seed 20261027/划分 seed 0/loader seed 12345、同一 npz 缓存与 scaler。显式 `--n-patients 2000` 钉档（D9 阶梯若重跑会按在盘 ~3.6 万 psv 选 8000 档，显式传参保证唯一变量=D7 遥测修复）。
- 性能指标对齐：8 指标（val/test × 入院级/小时级 × AUROC/AUPRC）×3 seed，**最大绝对差=0.0（bit 级复现）**——修复只动遥测、不动训练数学的实证；val 入院级 AUROC 0.7661/0.7696/0.7080 与 S5 逐位一致。
- D9：单 run 115.2/93.5/117.2s（cap 720s）；每 epoch 原子 ckpt，新增 60 个 `full_B_seed{0,1,2}_d7fix_ep*.pt`；**S5 产物零改动**（`git status`：仅 full_train.py 修复+progress 追加 3 个新 run_id 键；9 个 S5 progress 条目与 HEAD 逐字节相同；未跑 `--summary`，S5 的 CSV/JSON 汇总未被 glob 重写）。

## 四、H2(a) 预注册方向核对：A 全程 vs B-stage4 专家对余弦

预注册方向（design §二）：「A 臂训练期各专家梯度贡献余弦相似度（专家对之间）比 B 臂阶段四校准期更负」；判据=A−B-stage4<0。口径：A 臂每 seed 全部 20 epoch（共 60 epoch 级值，每值为 22 batch 均值）vs B-stage4 每 seed 2 epoch（共 6）。

| 专家对 | A 全程 per-seed mean | B-stage4 per-seed mean | 配对差 A−B-stage4（mean±std, ddof=1） | 方向 |
|---|---|---|---|---|
| cos_e1_e2 | −0.007464 / +0.001944 / −0.004772 | +0.001257 / +0.000333 / +0.002722 | **−0.004868±0.005645** | A 更负 |
| cos_e1_e3 | +0.000197 / +0.002506 / −0.003623 | −0.001943 / −0.002758 / +0.000177 | **+0.001202±0.004605** | B-stage4 更负 |
| cos_e2_e3 | −0.000897 / −0.004910 / +0.002259 | +0.000061 / +0.002169 / −0.000100 | **−0.001893±0.004788** | A 更负 |

- 9 个 seed×pair 单元中 5 个为负；**三对并非全部为负**（e1_e3 反向）；所有 |mean| 均小于配对差 std（种子噪声带内）。
- 量级：A 臂 max|cos|=0.017018（e1_e2，60 epoch）、B-stage4 max|cos|=0.004199（e1_e3，6 epoch）；两臂 epoch 级均值 |cos|≤0.0049——与 S5「绝对值均≈0」的观察一致，遥测补齐后量级判断不变。
- 稳健性视图（非预注册，仅防「A 早期 epoch 主导」质疑）：A 末 2 epoch vs B-stage4 同型——e1_e2 −0.004178±0.006516、e1_e3 +0.000313±0.005744、e2_e3 −0.002386±0.005666，不改变结论。
- 参考对照（S5 旧日志，不同阶段/对照，不作替代）：B-stage2 cos_e2_e3 mean +0.000509±0.001005（n=18）。

## 五、判读与证据边界

1. **修复判读=生效**：通过条件全满足（22/22、三对非 null、33/33 检查 PASS）；配置不变性+性能 bit 级复现证明唯一变量是遥测。
2. **H2(a) 方向判读=不支持（inconclusive 边界）**：遥测补齐后，A 全程 vs B-stage4 **不呈预注册「A 更负」方向**——2/3 对为负但 e1_e3 反向，且全部差值落在种子噪声内、绝对值≈0。按本阶段预注册边界记 **H2(a) 不支持 / inconclusive**：不把修复前的缺失当零值，也不把「三对中两对为负」包装成方向成立，更不把 |cos|≈0 反向包装成「证明无冲突」。design §二字面失败条件（(a) 不成立→H2 证伪）与指令边界（记 inconclusive）的张力如实记载：鉴于效应量比抽样噪声低两个量级、遥测抽样为 22 batch/epoch（design 原文 ≥200 口径在 2000 档不可行，S3/S5 已在案），本单元按指令取 inconclusive，不升级为证伪。
3. **H2 overall 维持 inconclusive / not established**：H1 前提已 falsified（S6 终判），H2 是「若 H1 成立」的机制解释；H2(b) 读数（B−B-no-cal=+0.107033±0.001550，3/3 正）仍只作机制事实记录，不因本次补测提升为 H2 支持。
4. **H1 不重开、不追认**：本补测只服务 H2(a) 机制证据；full 尺度三门线结论（falsified）不变。
5. **证据边界**：①抽样=22 batch/epoch（2000 档 train 1400 例），cos 估计噪声底与效应同阶，要更强机制证据需 full 语料到位后回归 design 原文 ≥200 batch/epoch 口径；②B-stage4 仅 2 epoch×3 seed=6 个 epoch 级值，A 60 个，比较是分布级而非逐 epoch 配对；③val 9 阳性等小样本限制不影响本遥测判读（无分类指标）；④本单元不复核 A 臂遥测本身（A 全程日志 S5 已完整，180 条定义良好余弦经 S5/S6 双重复算）。

## 六、复现入口

```
G:/ai/python11/python.exe results/_d7fix_recompute.py     # 独立复算（仅标准库，只读原始 JSON）
G:/ai/python11/python.exe experiments/_d7fix_unit_test.py  # 修复单测（18 项：AST/计数/功能/开关）
G:/ai/python11/python.exe experiments/_d7fix_drive.py      # B×3 补跑驱动（同 2000 例配置，tag=d7fix）
```

原始产物：`results/full_runs/full_B_seed{0,1,2}_d7fix.json`、`results/full_gradient_conflict_d7fix.json`（新 tag，不覆盖 S5）、`results/_d7fix_recompute.json`、`results/_d7fix_train.log`、`results/run-20261029-004358/run.log`。
