# S6-final-verdict：预注册终判

判读对象与方法

S5 实际执行的是 PhysioNet/CinC 2019 回退集、D9 子采样 2,000 例、固定 τ=(1,10,100)、hidden=1500、batch=64、lr=1e-3：A 联合训练 20 epoch ×3 种子，B 分阶段 8/6/4/2 ×3，B-no-cal 8/6/4 共18 epoch ×3。九个 run 均 done；val 入院级主指标 n=200，且每个 run 只有 9 个阳性入院。这里的“full”指 full 训练 schedule/多臂实验，不代表全量语料训练；D9 实际 tier 是 2,000。

本单元没有重跑训练。独立脚本 `results/_s6_recompute.py` 直接读取九个 `results/full_runs/full_*.json` 原始指标及两个 CSV，以样本标准差 ddof=1 复算臂均值、配对差、H1 门线，并逐 epoch 核验 D7 和 progress；72 个原始 JSON↔CSV 指标最大绝对差 4.9786e-5（CSV 四位小数导出误差）。执行方复算脚本与前序 researcher 判读脚本均另行执行/核对。两脚本有显示精度边缘问题：执行方汇总比较出现非门线相关的小误差告警；前序判读脚本对不同列名的 per-seed 检查会报 CSV None，且以 5e-5 比较 B−B-no-cal 的四位小数样本标准差会产生边界 FAIL。以上不作为结果证据；判决以原始 full_runs JSON 和在盘 CSV 为准，主效应差距远大于导出误差，门线不受影响。

H1：falsified（按 design.md §二、§六 的预注册映射）

- 以在盘 CSV 四位小数逐行独立复算（ddof=1）为主：A val 入院级 AUROC = 0.830133±0.013598；B = 0.747900±0.034599；配对 B−A = −0.082233±0.031776，逐种子 −0.0797/−0.0518/−0.1152，3/3 为负。full_runs JSON 保留精度复算分别为 A 0.830134±0.013630、B 0.747915±0.034638、B−A −0.082218±0.031780；两者差在 CSV 四位小数显示误差内，H1 门线结论相同。
- 配对 B−A val 入院级 AUPRC = −0.108367±0.056736；小时级 AUROC = −0.056600±0.037545，方向同为负。test 入院级 AUROC 差仅作描述性核对：−0.032700±0.023775；预注册主门线仍使用 validation 指标。
- 三门线逐项全未通过：ΔAUROC≥+0.01（实际 −0.082233，FAIL）；ΔAUPRC≥−0.01（实际 −0.108367，FAIL）；std(B)≤std(A)（0.034599≤0.013598，FAIL）。|ΔAUROC|/配对差 std=2.588；pilot B−A 约 −0.124，full schedule 仍为负、收敛至 −0.082，但没有翻正。
- 实现/数据限制不改变 H1 门线映射：S4 已核验 A/B 臂冻结语义与 D1/D10 一致；九跑均 done、无 NaN，三臂共享子采样/划分/顺序/scaler。S5 的 D7 日志 bug 是机制遥测缺口，不是 AUROC/AUPRC 表缺失或训练臂语义偏离。故按 §六“任一门线未达且实现类问题已排除”判 falsified，而非把效应改判为噪声带内 inconclusive。
- 限定解释：验证集 200 个入院只有 9 个阳性，n=3 种子且使用同一验证划分，数值效应大小与跨划分泛化仍有限；test 差异较小也提醒不要把 −0.082 当成精确总体效应。结论只针对该冻结设置与当前 tier，不声称所有分阶段 MoE 都有害。
- 设计固有训练预算混杂必须随结论保留：A 每个专家训练 20 epoch；B 中 τ1 为 stage1 8+stage4 2=10 epoch，τ2/τ3 各为 stage2 6+stage4 2=8 epoch（分别约 A 的 0.50 与 0.40），并非完全等专家更新预算。负差不能单独归因于“解耦本身有害”。

H2：overall inconclusive / not established（不以缺失日志作零值）

H2 是以“若 H1 成立”为前提的机制解释；H1 已 falsified，因此本结果不能证明 H2 所称的 H1 增益机制。分项保留如下：

- H2(a) 梯度冲突：无法按预注册 D7 对照 adjudicate。`full_gradient_conflict.json` 有 216 个定义良好余弦，但它们是 A 全程 180 条、B stage2 18 条、B-no-cal stage2 18 条；B stage4 的 6 个 epoch（3 seed×2）全部 `grad_logged_batches=0`、余弦为 null。根因已由前序代码审计核实：`full_train.py:398` 以 group 名 `expert` 前缀计联合可训专家组；`mr_moe.py:142` stage4 全部参数只放在名为 `all` 的单组中，故计数为 0、`:410` 的 `log_grad` 为 false。设计点名的 A 全程 vs B stage4 full 尺度比较无数据。当前可见的 A max|cos|=0.017018 与 B-stage2 max|cos|=0.002229 是不同训练阶段/对照，且绝对值均小；不得拿它们替代 stage4 比较或宣称无冲突。
- H2(b) / E2：按在盘 CSV 复算，B−B-no-cal 入院级 AUROC +0.107033±0.001550，逐种子 +0.1064/+0.1088/+0.1059（3/3 正）；AUPRC +0.052667±0.009981。full_runs JSON 保留精度结果为 +0.107039±0.001539 和 +0.052670±0.009952。按 H2 §二的可测定义“B-no-cal 不能复现 B 的全部增益”，该子读数方向上通过；但它识别的是“stage4 全参数联合更新 + 额外两 epoch”的整体 package，不能单独归因于校准机制。按 §三 E2 的字面预测“B-no-cal 与 B 差异不显著”，本读数不符合 E2 预测；该句与 §二对 B-no-cal≈B 的失败映射相反，故以 §二 H2 子假设定义报告 H2(b) 方向通过，同时把 E2 标为未获支持并保留预注册文本冲突，不将子读数提升为完整 H2 支持。H1 前提也未成立，H2 overall 仍 inconclusive/not established。已向 researcher 发出复核请求（event `89a94b5444e14a5897254062d89ed585`），请求裁决矛盾并排序 D7 补测；本报告不等待该请求而保留明确边界。

负结果归类（FAILURE-PATTERNS 语义）

- 科学类：假设层负结果——冻结设置下，分阶段臂未达预注册增益/稳定性门线，H1 falsified。不是“实验跑不通”：九个 run 完成，指标均可计算。
- 机制证据类：实现/仪表化缺口——D7 stage4 记录开关由组名命名约定错误关闭，导致 H2(a) 缺证；它不污染主性能指标，但阻断机制比较。
- 统计/范围类：val 阳性只有 9、三种子共享划分、tier=2,000；适用于收窄效应量与泛化声称，不推翻清楚失败的预注册门线。

替代方向（各为最多 30 分钟的小验；不是继续调参追 H1）

1. D7 stage4 记录修复与定点机制诊断（优先）：把联合专家组识别改为按专家参数 `requires_grad`/参数集合实际交集判断，或显式 stage4 判定；新 tag 保存，不覆盖 S5。用同一 2,000 例配置重跑 B×3，旧运行实测每跑约108–112秒、三跑约6分钟。通过条件：每个 B seed 的两个 stage4 epoch 都有 22/22 logged batches，三对 cos 均非 null；失败条件：任一 stage4 epoch 仍无完整日志，或 A 全程与 B-stage4 的三对配对余弦不能满足预注册“A 更负”方向，则 H2(a) 不支持（数据无效则仍 inconclusive）。这只是补机制证据，不重开 H1 性能门线。
2. 等专家更新量的预算诊断：用 2,000 例、同 seed0、同初始化/划分，单跑探索性 B′=18/18/2/2（stage1/2/3/4，共40 epoch），令三个专家各获 20 个可训练 epoch（stage1+4=20；stage2+4=20），与现有 A seed0 的每专家20 epoch 对照。按现有 B 吞吐外推约4–6分钟。失败条件：B′ 入院级 AUROC 仍比 A seed0 低至少0.01；这会削弱“主要由专家更新预算不足导致”的解释。若翻正也仅为单 seed 线索，不能替代预注册三种子结论。
3. τ 比敏感性筛查：2000例、同一个种子，用 τ=(1,5,25) 跑 A/B 各一次（沿用同一评测/预处理；预计约5–10分钟）。失败条件：B−A 仍≤−0.01，则当前负方向不支持“仅 τ=1:10:100 特异”的解释；若转正，只能形成配置交互假说，需另行预注册多种子验证。
4. 扩大验证样本：待在盘真实语料至少达到 D9 的8,000档后，同配置跑 A/B seed0 一对（约30分钟内，沿用相同划分纪律并报告验证阳性数）。失败条件：增量样本下 B−A 仍≤−0.02，则“2,000例/9阳性造成方向性负差”的解释被削弱；正向结果只说明样本规模敏感，不能追认本次 H1。

下一步选择

优先方向1：只修复 D7 日志并补跑 B×3，预计约6分钟，目标是关闭 H2(a) 证据缺口；不追加 H1 重跑、不继续优化分阶段指标。H1 主结果维持 falsified。研究者求助事件已登记，供下游 reviewer 对 §二/§三 判据冲突和补测优先级复核。

复现入口与证据

- 独立复算：`python results/_s6_recompute.py`（仅标准库，读原始 full_runs JSON、CSV、progress 和 D7 JSON，不训练）。
- 原始结果：`results/full_joint_vs_staged.csv`、`results/full_bnocal.csv`、`results/full_joint_vs_staged.md`、`results/full_runs/full_*.json`、`results/full_gradient_conflict.json`、`results/full_progress.json`、`results/_full_summary.json`、`results/_full_tier_decision.json`。
- 前序独立审计与代码根因：`notes/researcher-interventions/f6ba4c1b7c024eda8d912348c0875fca.md`；预注册门线：`design.md` §二、§六；门控交接：`notes/handoff/S5-full-experiment.md`。
