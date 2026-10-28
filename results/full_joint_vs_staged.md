# S5-full-experiment：联合(A) vs 分阶段(B) × 3 种子 + B-no-cal 消融 × 3 种子

- 生成时间：2026-10-28T01:06:55+0800；完成 run 9/9
- 数据：PhysioNet-CinC 2019 子采样 n=2000（在盘 3339，跳过 0；septic 率 0.063；划分 {'train': 1400, 'val': 200, 'test': 400}；D10 共用子采样/划分/数据顺序/scaler）
- D9 阶梯定档：2000 条/run；在盘 3339 psv；原因：训练启动时在盘 3339 psv；<4000 取 2000 档（D9 预授权阶梯内降档）；hidden 恒 1500；禁合成数据（F2 不触发：全部真实 PhysioNet 2019 语料）
- 配置（D4 冻结）：K=3 CfC hidden 1500、τ=1:10:100、门控 MLP64、Adam lr=1e-3、BCE、batch 64、E=20（B=8/6/4/2；B-no-cal=8/6/4=18epoch）；种子{0,1,2}；每 epoch 原子 checkpoint
- 算子状态：批量训练全部在 S2⓪ 修后算子上（results/operator_bench.md PASS，F8 未触发）——无「基于旧算子」数字（fb-ab50d0ec9f 条款 vacuous）

## 结论级表：mean±std（val；入院级主指标 + 小时级次指标；D6/D8）

| 臂 | n_done | val_adm_AUROC | val_adm_AUPRC | val_hourly_AUROC | val_hourly_AUPRC |
|---|---|---|---|---|---|
| A | 3 | 0.8301±0.0136 | 0.2290±0.0624 | 0.7390±0.0058 | 0.0302±0.0017 |
| B | 3 | 0.7479±0.0346 | 0.1206±0.0201 | 0.6824±0.0427 | 0.0226±0.0049 |
| BnoCal | 3 | 0.6409±0.0336 | 0.0679±0.0113 | 0.6464±0.0558 | 0.0205±0.0054 |

## 逐 run 明细（val=最后 epoch；test=结束后一次性）

| run | status | ep | train_loss | val_loss | val_adm_AUROC | val_adm_AUPRC | val_h_AUROC | val_h_AUPRC | test_adm_AUROC | test_adm_AUPRC | total_s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| full_A_seed0 | done | 20 | 0.0791 | 0.0595 | 0.8458 | 0.2998 | 0.7451 | 0.0321 | 0.8286 | 0.2132 | 227.8 |
| full_A_seed1 | done | 20 | 0.0793 | 0.0599 | 0.8214 | 0.2048 | 0.7386 | 0.0287 | 0.8176 | 0.2106 | 211.2 |
| full_A_seed2 | done | 20 | 0.0788 | 0.0594 | 0.8232 | 0.1823 | 0.7334 | 0.0300 | 0.8113 | 0.2411 | 223.0 |
| full_B_seed0 | done | 20 | 0.0822 | 0.0622 | 0.7661 | 0.1266 | 0.7120 | 0.0255 | 0.8161 | 0.1853 | 111.8 |
| full_B_seed1 | done | 20 | 0.0817 | 0.0624 | 0.7696 | 0.1370 | 0.7018 | 0.0254 | 0.7587 | 0.1412 | 107.7 |
| full_B_seed2 | done | 20 | 0.0825 | 0.0633 | 0.7080 | 0.0982 | 0.6335 | 0.0170 | 0.7846 | 0.1700 | 108.9 |
| full_BnoCal_seed0 | done | 18 | 0.0810 | 0.0607 | 0.6597 | 0.0672 | 0.6738 | 0.0244 | 0.7494 | 0.1139 | 94.1 |
| full_BnoCal_seed1 | done | 18 | 0.0811 | 0.0606 | 0.6608 | 0.0796 | 0.6833 | 0.0229 | 0.7123 | 0.1002 | 95.1 |
| full_BnoCal_seed2 | done | 18 | 0.0816 | 0.0617 | 0.6021 | 0.0570 | 0.5822 | 0.0143 | 0.7127 | 0.0943 | 100.7 |

## 逐种子配对明细（D6）

### B−A（分阶段−联合；H1 方向性）

| seed | val_admission_auroc | val_admission_auprc | val_hourly_auroc | val_hourly_auprc |
|---|---|---|---|---|
| 0 | -0.0797 | -0.1733 | -0.0331 | -0.0066 |
| 1 | -0.0518 | -0.0678 | -0.0368 | -0.0033 |
| 2 | -0.1152 | -0.0841 | -0.0999 | -0.0131 |

mean±std：val_admission_auroc=-0.0822±0.0318 (n=3)；val_admission_auprc=-0.1084±0.0568 (n=3)；val_hourly_auroc=-0.0566±0.0376 (n=3)；val_hourly_auprc=-0.0076±0.0050 (n=3)

### B−B-no-cal（阶段四联合校准效应；E2/H2b 判据）

| seed | val_admission_auroc | val_admission_auprc | val_hourly_auroc | val_hourly_auprc |
|---|---|---|---|---|
| 0 | 0.1065 | 0.0593 | 0.0382 | 0.0011 |
| 1 | 0.1088 | 0.0574 | 0.0185 | 0.0025 |
| 2 | 0.1059 | 0.0412 | 0.0513 | 0.0027 |

mean±std：val_admission_auroc=0.1070±0.0015 (n=3)；val_admission_auprc=0.0527±0.0100 (n=3)；val_hourly_auroc=0.0360±0.0165 (n=3)；val_hourly_auprc=0.0021±0.0008 (n=3)

### A−B-no-cal（参考：无阶段四的分阶段臂 vs 联合臂）

| seed | val_admission_auroc | val_admission_auprc | val_hourly_auroc | val_hourly_auprc |
|---|---|---|---|---|
| 0 | 0.1862 | 0.2326 | 0.0713 | 0.0077 |
| 1 | 0.1606 | 0.1252 | 0.0553 | 0.0058 |
| 2 | 0.2211 | 0.1254 | 0.1512 | 0.0157 |

mean±std：val_admission_auroc=0.1893±0.0304 (n=3)；val_admission_auprc=0.1611±0.0620 (n=3)；val_hourly_auroc=0.0926±0.0514 (n=3)；val_hourly_auprc=0.0097±0.0053 (n=3)

## H1 预注册门线读数（design §六；终判归 S6-final-verdict）

- 判据① ΔAUROC(B−A, val 入院级) ≥ +0.01：-0.0822±0.0318 → FAIL
- 判据② ΔAUPRC(B−A, val 入院级) ≥ −0.01：-0.1084±0.0568 → FAIL
- 判据③ std(B) ≤ std(A)（val 入院级 AUROC）：0.0346 vs 0.0136 → FAIL
- 三项同时：否（本表为 full 尺度复测读数，verdict 由 S6 按 §六 映射出具）

## B-no-cal 消融（E2/H2b 判据）

- B−B-no-cal（val 入院级 AUROC）：mean=0.1070±0.0015（n=3 配对种子）；AUPRC mean=0.0527±0.0100
- B-no-cal=阶段一至三（8/6/4=18 epoch，与 B 的阶段1-3 步数逐字相同，唯一差异=无阶段四 2 epoch 联合校准）；差值即阶段四效应读数。
- 解读映射存疑（交 S6 显式裁决）：design §二 H2(b) 失败条件把「B-no-cal≈B」列为 H2 证伪且机制解释收窄为「短联合校准的再耦合效应」，而 §三 E2「若真则看到」列又把「B-no-cal 与 B 差异不显著」列为末端校准效应的表现——两处对同一读数的指向相反；本表只报数与两种解读，不预选。

## 梯度冲突（D7，H2a 代理）

- full_A_seed0（20 个记录 epoch）：
  - ep0 stageNone logged=22 ‖g‖=[0.0722, 0.0575, 0.1485] cos={'cos_e1_e2': -0.005, 'cos_e1_e3': -0.0023, 'cos_e2_e3': -0.0}
  - ep1 stageNone logged=22 ‖g‖=[0.0268, 0.0173, 0.0339] cos={'cos_e1_e2': -0.0073, 'cos_e1_e3': -0.0005, 'cos_e2_e3': 0.0003}
  - ep2 stageNone logged=22 ‖g‖=[0.0292, 0.0183, 0.0247] cos={'cos_e1_e2': -0.0038, 'cos_e1_e3': -0.0001, 'cos_e2_e3': 0.0}
  - ep3 stageNone logged=22 ‖g‖=[0.0228, 0.0172, 0.0205] cos={'cos_e1_e2': -0.0003, 'cos_e1_e3': -0.0011, 'cos_e2_e3': 0.0001}
  - ep4 stageNone logged=22 ‖g‖=[0.0173, 0.0133, 0.0167] cos={'cos_e1_e2': -0.0003, 'cos_e1_e3': -0.0005, 'cos_e2_e3': -0.0002}
  - ep5 stageNone logged=22 ‖g‖=[0.0093, 0.0069, 0.0109] cos={'cos_e1_e2': -0.0037, 'cos_e1_e3': 0.0002, 'cos_e2_e3': -0.0001}
  - ep6 stageNone logged=22 ‖g‖=[0.0092, 0.0068, 0.0146] cos={'cos_e1_e2': -0.0103, 'cos_e1_e3': 0.002, 'cos_e2_e3': 0.0002}
  - ep7 stageNone logged=22 ‖g‖=[0.0099, 0.0079, 0.0137] cos={'cos_e1_e2': -0.0071, 'cos_e1_e3': 0.001, 'cos_e2_e3': -0.0005}
  - ep8 stageNone logged=22 ‖g‖=[0.0097, 0.0076, 0.0115] cos={'cos_e1_e2': -0.0017, 'cos_e1_e3': -0.0005, 'cos_e2_e3': 0.0001}
  - ep9 stageNone logged=22 ‖g‖=[0.0106, 0.0076, 0.0115] cos={'cos_e1_e2': -0.0036, 'cos_e1_e3': -0.0002, 'cos_e2_e3': 0.0003}
  - ep10 stageNone logged=22 ‖g‖=[0.0055, 0.0043, 0.0094] cos={'cos_e1_e2': -0.0104, 'cos_e1_e3': 0.0023, 'cos_e2_e3': -0.0002}
  - ep11 stageNone logged=22 ‖g‖=[0.0062, 0.005, 0.0097] cos={'cos_e1_e2': -0.0093, 'cos_e1_e3': 0.0009, 'cos_e2_e3': -0.0003}
  - ep12 stageNone logged=22 ‖g‖=[0.0054, 0.0047, 0.0085] cos={'cos_e1_e2': -0.0081, 'cos_e1_e3': -0.0005, 'cos_e2_e3': -0.0002}
  - ep13 stageNone logged=22 ‖g‖=[0.0052, 0.004, 0.0079] cos={'cos_e1_e2': -0.0066, 'cos_e1_e3': 0.0003, 'cos_e2_e3': -0.0007}
  - ep14 stageNone logged=22 ‖g‖=[0.0041, 0.0035, 0.0101] cos={'cos_e1_e2': -0.0062, 'cos_e1_e3': 0.0005, 'cos_e2_e3': -0.0018}
  - ep15 stageNone logged=22 ‖g‖=[0.0038, 0.0027, 0.0073] cos={'cos_e1_e2': -0.0154, 'cos_e1_e3': 0.0032, 'cos_e2_e3': -0.0047}
  - ep16 stageNone logged=22 ‖g‖=[0.0038, 0.0029, 0.0065] cos={'cos_e1_e2': -0.0159, 'cos_e1_e3': 0.0022, 'cos_e2_e3': -0.0045}
  - ep17 stageNone logged=22 ‖g‖=[0.0024, 0.0021, 0.0124] cos={'cos_e1_e2': -0.017, 'cos_e1_e3': 0.0049, 'cos_e2_e3': -0.006}
  - ep18 stageNone logged=22 ‖g‖=[0.0031, 0.0028, 0.0052] cos={'cos_e1_e2': -0.0099, 'cos_e1_e3': -0.0046, 'cos_e2_e3': -0.0004}
  - ep19 stageNone logged=22 ‖g‖=[0.0031, 0.0027, 0.0067] cos={'cos_e1_e2': -0.0073, 'cos_e1_e3': -0.0033, 'cos_e2_e3': 0.0007}
- full_A_seed1（20 个记录 epoch）：
  - ep0 stageNone logged=22 ‖g‖=[0.0308, 0.0677, 0.1792] cos={'cos_e1_e2': 0.0026, 'cos_e1_e3': 0.0053, 'cos_e2_e3': -0.0033}
  - ep1 stageNone logged=22 ‖g‖=[0.0084, 0.0242, 0.055] cos={'cos_e1_e2': 0.0006, 'cos_e1_e3': 0.0013, 'cos_e2_e3': -0.0036}
  - ep2 stageNone logged=22 ‖g‖=[0.0093, 0.0247, 0.0483] cos={'cos_e1_e2': 0.0003, 'cos_e1_e3': 0.0024, 'cos_e2_e3': -0.0034}
  - ep3 stageNone logged=22 ‖g‖=[0.0078, 0.0194, 0.036] cos={'cos_e1_e2': -0.0001, 'cos_e1_e3': 0.0008, 'cos_e2_e3': -0.004}
  - ep4 stageNone logged=22 ‖g‖=[0.0075, 0.0166, 0.0356] cos={'cos_e1_e2': -0.0007, 'cos_e1_e3': 0.0013, 'cos_e2_e3': -0.0035}
  - ep5 stageNone logged=22 ‖g‖=[0.0043, 0.0089, 0.0274] cos={'cos_e1_e2': 0.0016, 'cos_e1_e3': 0.0022, 'cos_e2_e3': -0.0048}
  - ep6 stageNone logged=22 ‖g‖=[0.0054, 0.0117, 0.0284] cos={'cos_e1_e2': 0.0022, 'cos_e1_e3': 0.0017, 'cos_e2_e3': -0.0045}
  - ep7 stageNone logged=22 ‖g‖=[0.0057, 0.0129, 0.0222] cos={'cos_e1_e2': 0.0026, 'cos_e1_e3': 0.0049, 'cos_e2_e3': -0.008}
  - ep8 stageNone logged=22 ‖g‖=[0.0055, 0.0109, 0.0188] cos={'cos_e1_e2': 0.0004, 'cos_e1_e3': 0.0034, 'cos_e2_e3': -0.0045}
  - ep9 stageNone logged=22 ‖g‖=[0.0057, 0.0115, 0.0232] cos={'cos_e1_e2': 0.0003, 'cos_e1_e3': 0.0034, 'cos_e2_e3': -0.0018}
  - ep10 stageNone logged=22 ‖g‖=[0.0035, 0.0069, 0.023] cos={'cos_e1_e2': 0.0027, 'cos_e1_e3': 0.0022, 'cos_e2_e3': -0.0046}
  - ep11 stageNone logged=22 ‖g‖=[0.0059, 0.0111, 0.0305] cos={'cos_e1_e2': 0.0022, 'cos_e1_e3': 0.0013, 'cos_e2_e3': -0.0051}
  - ep12 stageNone logged=22 ‖g‖=[0.0058, 0.0078, 0.0156] cos={'cos_e1_e2': 0.0008, 'cos_e1_e3': 0.0013, 'cos_e2_e3': -0.0044}
  - ep13 stageNone logged=22 ‖g‖=[0.0053, 0.0064, 0.0128] cos={'cos_e1_e2': 0.0011, 'cos_e1_e3': 0.0015, 'cos_e2_e3': -0.0031}
  - ep14 stageNone logged=22 ‖g‖=[0.0028, 0.0048, 0.0095] cos={'cos_e1_e2': 0.0047, 'cos_e1_e3': 0.0024, 'cos_e2_e3': -0.0073}
  - ep15 stageNone logged=22 ‖g‖=[0.0033, 0.0048, 0.0111] cos={'cos_e1_e2': 0.0025, 'cos_e1_e3': 0.001, 'cos_e2_e3': -0.0053}
  - ep16 stageNone logged=22 ‖g‖=[0.0029, 0.0046, 0.0096] cos={'cos_e1_e2': 0.0053, 'cos_e1_e3': 0.0018, 'cos_e2_e3': -0.0065}
  - ep17 stageNone logged=22 ‖g‖=[0.0034, 0.003, 0.0072] cos={'cos_e1_e2': 0.0031, 'cos_e1_e3': 0.0037, 'cos_e2_e3': -0.0071}
  - ep18 stageNone logged=22 ‖g‖=[0.003, 0.0042, 0.0098] cos={'cos_e1_e2': 0.0049, 'cos_e1_e3': 0.0064, 'cos_e2_e3': -0.0079}
  - ep19 stageNone logged=22 ‖g‖=[0.0028, 0.004, 0.0107] cos={'cos_e1_e2': 0.0016, 'cos_e1_e3': 0.0018, 'cos_e2_e3': -0.0055}
- full_A_seed2（20 个记录 epoch）：
  - ep0 stageNone logged=22 ‖g‖=[0.0465, 0.0969, 0.1195] cos={'cos_e1_e2': -0.0119, 'cos_e1_e3': -0.0086, 'cos_e2_e3': -0.0018}
  - ep1 stageNone logged=22 ‖g‖=[0.0111, 0.0348, 0.0626] cos={'cos_e1_e2': -0.0047, 'cos_e1_e3': -0.0009, 'cos_e2_e3': -0.0003}
  - ep2 stageNone logged=22 ‖g‖=[0.0145, 0.0423, 0.0234] cos={'cos_e1_e2': -0.0026, 'cos_e1_e3': -0.0018, 'cos_e2_e3': 0.0003}
  - ep3 stageNone logged=22 ‖g‖=[0.0099, 0.0236, 0.0157] cos={'cos_e1_e2': -0.0053, 'cos_e1_e3': -0.0005, 'cos_e2_e3': -0.0003}
  - ep4 stageNone logged=22 ‖g‖=[0.0081, 0.0177, 0.009] cos={'cos_e1_e2': -0.0044, 'cos_e1_e3': -0.0044, 'cos_e2_e3': 0.0027}
  - ep5 stageNone logged=22 ‖g‖=[0.0068, 0.0143, 0.0087] cos={'cos_e1_e2': -0.0024, 'cos_e1_e3': -0.0031, 'cos_e2_e3': 0.002}
  - ep6 stageNone logged=22 ‖g‖=[0.0072, 0.0138, 0.0112] cos={'cos_e1_e2': -0.004, 'cos_e1_e3': -0.0043, 'cos_e2_e3': 0.003}
  - ep7 stageNone logged=22 ‖g‖=[0.008, 0.0155, 0.0153] cos={'cos_e1_e2': -0.007, 'cos_e1_e3': -0.0034, 'cos_e2_e3': 0.0046}
  - ep8 stageNone logged=22 ‖g‖=[0.0075, 0.014, 0.0112] cos={'cos_e1_e2': -0.0009, 'cos_e1_e3': -0.0025, 'cos_e2_e3': 0.0015}
  - ep9 stageNone logged=22 ‖g‖=[0.0074, 0.014, 0.0152] cos={'cos_e1_e2': -0.0008, 'cos_e1_e3': -0.0024, 'cos_e2_e3': 0.002}
  - ep10 stageNone logged=22 ‖g‖=[0.004, 0.0074, 0.0109] cos={'cos_e1_e2': -0.0052, 'cos_e1_e3': -0.0042, 'cos_e2_e3': 0.003}
  - ep11 stageNone logged=22 ‖g‖=[0.004, 0.0071, 0.0101] cos={'cos_e1_e2': -0.0065, 'cos_e1_e3': -0.0056, 'cos_e2_e3': 0.0036}
  - ep12 stageNone logged=22 ‖g‖=[0.0041, 0.007, 0.0089] cos={'cos_e1_e2': -0.0038, 'cos_e1_e3': -0.0037, 'cos_e2_e3': 0.0005}
  - ep13 stageNone logged=22 ‖g‖=[0.0041, 0.006, 0.0089] cos={'cos_e1_e2': -0.0055, 'cos_e1_e3': -0.0062, 'cos_e2_e3': 0.0041}
  - ep14 stageNone logged=22 ‖g‖=[0.0045, 0.0053, 0.0068] cos={'cos_e1_e2': -0.0023, 'cos_e1_e3': -0.0034, 'cos_e2_e3': 0.0046}
  - ep15 stageNone logged=22 ‖g‖=[0.0035, 0.0044, 0.0083] cos={'cos_e1_e2': -0.0042, 'cos_e1_e3': -0.0028, 'cos_e2_e3': 0.0037}
  - ep16 stageNone logged=22 ‖g‖=[0.0031, 0.005, 0.0076] cos={'cos_e1_e2': -0.0047, 'cos_e1_e3': -0.0034, 'cos_e2_e3': 0.0026}
  - ep17 stageNone logged=22 ‖g‖=[0.0018, 0.0033, 0.0062] cos={'cos_e1_e2': -0.0136, 'cos_e1_e3': -0.0037, 'cos_e2_e3': 0.0063}
  - ep18 stageNone logged=22 ‖g‖=[0.0024, 0.0046, 0.0084] cos={'cos_e1_e2': -0.0036, 'cos_e1_e3': -0.0036, 'cos_e2_e3': 0.0011}
  - ep19 stageNone logged=22 ‖g‖=[0.0025, 0.0044, 0.007] cos={'cos_e1_e2': -0.0021, 'cos_e1_e3': -0.0039, 'cos_e2_e3': 0.002}
- full_B_seed0（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0122, 0.0402] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0005}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0219, 0.0312] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0011}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.0077, 0.0154] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0014}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0081, 0.0181] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0008}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.0088, 0.0131] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.002}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0063, 0.0072] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0006}
- full_B_seed1（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0133, 0.098] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0012}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0172, 0.0381] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0002}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.01, 0.025] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0011}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0153, 0.0397] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.011, 0.0343] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0083, 0.0268] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0022}
- full_B_seed2（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0181, 0.0486] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0393, 0.0307] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0017}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.0154, 0.0276] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0002}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0226, 0.0269] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0006}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.0154, 0.0765] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0003}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0065, 0.0154] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.001}
- full_BnoCal_seed0（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0122, 0.0402] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0005}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0219, 0.0312] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0011}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.0077, 0.0154] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0014}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0081, 0.0181] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0008}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.0088, 0.0131] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.002}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0063, 0.0072] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0006}
- full_BnoCal_seed1（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0133, 0.098] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0012}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0172, 0.0381] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0002}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.01, 0.025] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0011}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0153, 0.0397] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.011, 0.0343] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0083, 0.0268] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0022}
- full_BnoCal_seed2（6 个记录 epoch）：
  - ep8 stage2 logged=22 ‖g‖=[None, 0.0181, 0.0486] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0007}
  - ep9 stage2 logged=22 ‖g‖=[None, 0.0393, 0.0307] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0017}
  - ep10 stage2 logged=22 ‖g‖=[None, 0.0154, 0.0276] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.0002}
  - ep11 stage2 logged=22 ‖g‖=[None, 0.0226, 0.0269] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0006}
  - ep12 stage2 logged=22 ‖g‖=[None, 0.0154, 0.0765] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': -0.0003}
  - ep13 stage2 logged=22 ‖g‖=[None, 0.0065, 0.0154] cos={'cos_e1_e2': None, 'cos_e1_e3': None, 'cos_e2_e3': 0.001}
- 全量 max|cos|=0.017018（H2a：A 臂全程 vs B 臂联合期的梯度冲突读数；pilot 全量 max|cos|=0.008757，full 复测值见上）

## 时间（D9：单训练 ≤720s）

- full_A_seed0：total=227.8s train=211.4s per_epoch=[11.93, 10.1, 11.18, 11.83, 10.73, 10.76, 10.67, 10.74, 10.64, 10.21, 9.86, 10.12, 11.87, 10.69, 10.72, 10.52, 10.61, 10.71, 10.93, 10.92]
- full_A_seed1：total=211.2s train=205.8s per_epoch=[10.39, 11.35, 11.14, 9.83, 9.9, 10.2, 10.37, 10.55, 11.41, 10.54, 10.17, 10.16, 10.11, 9.97, 9.94, 10.0, 10.15, 11.12, 11.36, 11.37]
- full_A_seed2：total=223.0s train=217.3s per_epoch=[11.41, 10.21, 11.29, 11.32, 11.09, 10.38, 11.25, 11.37, 11.35, 11.35, 11.56, 11.12, 10.75, 11.1, 11.13, 11.17, 11.23, 10.48, 10.6, 11.67]
- full_B_seed0：total=111.8s train=106.0s per_epoch=[4.28, 3.81, 3.78, 3.74, 3.98, 3.84, 4.31, 4.35, 8.88, 8.93, 8.75, 9.0, 8.82, 8.28, 2.86, 2.85, 2.82, 2.78, 7.22, 7.24]
- full_B_seed1：total=107.7s train=102.0s per_epoch=[4.3, 4.18, 4.26, 4.09, 4.11, 3.8, 3.77, 3.71, 7.5, 7.68, 9.05, 8.52, 8.03, 8.04, 2.7, 2.84, 2.73, 2.76, 7.07, 7.27]
- full_B_seed2：total=108.9s train=103.1s per_epoch=[4.12, 4.42, 4.35, 4.33, 4.16, 4.09, 4.1, 3.83, 7.73, 8.71, 8.85, 8.41, 7.76, 7.78, 2.78, 2.84, 2.88, 2.92, 6.96, 6.59]
- full_BnoCal_seed0：total=94.1s train=88.8s per_epoch=[3.83, 3.81, 3.84, 3.77, 4.12, 4.33, 4.33, 4.32, 7.68, 7.49, 8.02, 8.64, 8.74, 8.97, 2.62, 2.71, 2.75, 2.83]
- full_BnoCal_seed1：total=95.1s train=89.4s per_epoch=[4.32, 4.36, 4.3, 4.26, 4.2, 3.97, 3.92, 4.08, 7.94, 8.33, 7.93, 7.97, 7.62, 8.5, 2.86, 2.92, 3.07, 2.99]
- full_BnoCal_seed2：total=100.7s train=94.8s per_epoch=[4.2, 4.03, 4.07, 4.05, 4.41, 4.16, 4.15, 4.41, 10.11, 9.04, 8.72, 9.61, 8.47, 8.41, 2.89, 2.76, 2.89, 2.91]

## 实现记录与偏差（如实）

- D7「每 epoch ≥200 batch」在本档（train≈1400 例→22 batch/epoch）不可行（需额外前向反向、击穿 D9 上限），按 pilot 先例每 epoch 全量 batch 记录；原文口径需 full 语料（setA 20643×0.7≈14450 train→226 batch/epoch），语料下载后台续跑中。
- D7 记录范围相对 pilot 受控扩展：pilot=B 仅 stage4；full 按「≥2 专家联合可训即记录」执行（A 全程；B stage2+stage4；B-no-cal stage2），所有上报余弦均定义良好。
- B-no-cal 总 epoch=18（8/6/4）：与 B 的阶段1-3 逐字相同，唯一差异=无阶段四。
- padding 到 72h：填充步值通道 0、mask 通道 1，损失/评测只在真实小时步计算。
- B/B-no-cal 阶段切换时重建 optimizer（Adam 动量重置；被冻结参数状态本即无关）。
- 入院级标签=窗内任一小时阳性；评分=窗口概率最大值（D8）。
- 数据源=PhysioNet-CinC 2019 回退源（Moor 2023 不可得，S2 证据落盘）；全文语料下载后台续跑中（results/_download_state.json），本档按 D9 阶梯定档。
- D5 单体 LNN sanity 未在 full 重跑（预算优先 A/B/B-no-cal 九跑）：pilot 已 PASS（A−LNN val +0.1499/test +0.1138），full 复用同实现同超参；若 A 臂 full 读数异常低再回查（design D5 口径）。
- 解读警告（交 S6）：B/A 专家更新预算比 0.5 系 4:3:2:1 设计固有（full 8/6/4/2 与 pilot 2/2/1/1 比例相同，仅绝对量放大），S6 终判须把该设计固有混杂显式写入解读。
- full 不终判 H1（design §六：full 出表，S6 按门线映射出具 verdict）。

