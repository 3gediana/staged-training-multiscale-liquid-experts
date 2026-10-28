#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_d7fix_recompute.py — S6-final-verdict-redesign 独立复算（仅标准库，不训练）。

输入（全部只读，不改写任何 S5 产物）：
  - results/full_runs/full_B_seed{0,1,2}_d7fix.json   # 新 tag 补跑（修复后）
  - results/full_runs/full_B_seed{0,1,2}.json        # S5 旧 B 臂（stage4 遥测全缺失）
  - results/full_runs/full_A_seed{0,1,2}.json        # S5 A 臂全程（D7 日志完整）
  - results/full_gradient_conflict.json              # S5 D7 旧日志（修复前对照）

复算内容：
  1. 通过条件（预注册）：每个 B seed 的两个 stage4 epoch 均 grad_logged_batches=22
     （=n_batches）且三对专家余弦全部非 null 且有限；stage2 行为与 S5 一致
     （22/22、cos_e2_e3 非 null）；stage1/3 仍不记录。
  2. 配置不变性（D10 单变量纪律）：d7fix 与 S5 的 config/data 逐字段相等（除
     build_sec），唯一变量=D7 遥测修复；性能指标（val/test 入院级+小时级
     AUROC/AUPRC）逐 seed 对齐差（预期 GPU 非确定性量级）。
  3. D7 修复前后对照：stage4 六 epoch 的 logged batches 与定义良好余弦数。
  4. H2(a) 预注册方向核对（design §二）：A 臂全程（60 epoch）专家对余弦 vs
     B 臂 stage4 校准期（6 epoch）专家对余弦，逐种子配对差（A−B-stage4<0
     ⇒A 更负⇒H2(a) 支持方向）；附 A 末 2 epoch 稳健性视图（非预注册，标注）。
  5. 边界：|cos| 量级 descriptive；H2 overall 维持 inconclusive（H1 前提已
     falsified，本补测只服务 H2(a) 机制证据，不重开 H1 门线）。

输出：results/_d7fix_recompute.json + results/full_gradient_conflict_d7fix.json
（后者与 S5 full_gradient_conflict.json 同 schema，只含新 tag 三 run，不覆盖旧件）。
"""
import json
import math
import os
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PAIRS = ("cos_e1_e2", "cos_e1_e3", "cos_e2_e3")
SEEDS = (0, 1, 2)


def mean_std(vals):
    n = len(vals)
    m = sum(vals) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"mean": m, "std": sd, "n": n}


def load(run_id):
    with open(os.path.join(BASE, "full_runs", run_id + ".json"), encoding="utf-8") as f:
        return json.load(f)


def finite(v):
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


report = {
    "schema": "d7fix-recompute/v1",
    "stage": "S6-final-verdict-redesign",
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "method": ("仅标准库；读 full_runs 原始 JSON（新 tag full_B_seed*_d7fix 与 S5 "
               "full_A/B_seed*）+ full_gradient_conflict.json；std=ddof=1；不训练、"
               "不改写 S5 产物。"),
    "source_files": ["results/full_runs/full_B_seed{0,1,2}_d7fix.json",
                     "results/full_runs/full_B_seed{0,1,2}.json",
                     "results/full_runs/full_A_seed{0,1,2}.json",
                     "results/full_gradient_conflict.json"],
}

# ---------------------------------------------------------------- 1. 遥测通过条件
d7fix = {s: load(f"full_B_seed{s}_d7fix") for s in SEEDS}
s5_b = {s: load(f"full_B_seed{s}") for s in SEEDS}
s5_a = {s: load(f"full_A_seed{s}") for s in SEEDS}

telemetry = {"per_seed": {}, "all_pass": True}
for s in SEEDS:
    r = d7fix[s]
    eps = r["epochs_records"]
    st4 = [e for e in eps if e["stage"] == 4]
    st2 = [e for e in eps if e["stage"] == 2]
    st13 = [e for e in eps if e["stage"] in (1, 3)]
    checks = {
        "status_done": r["status"] == "done",
        "n_epochs_20": len(eps) == 20,
        "stage4_epochs_2": len(st4) == 2,
        "stage4_epochs_are_18_19": [e["epoch"] for e in st4] == [18, 19],
        "stage4_logged_22_each": all(e["grad_logged_batches"] == 22 for e in st4),
        "stage4_logged_eq_n_batches": all(
            e["grad_logged_batches"] == e["n_batches"] == 22 for e in st4),
        "stage4_three_cos_nonnull_finite": all(
            all(finite((e.get("grad_cos_mean") or {}).get(k)) for k in PAIRS)
            for e in st4),
        "stage4_expert_grad_norms_3": all(
            len([v for v in (e.get("expert_grad_norm_mean") or []) if finite(v)]) == 3
            for e in st4),
        "stage2_6_epochs_logged_22": len(st2) == 6 and all(
            e["grad_logged_batches"] == 22 for e in st2),
        "stage2_cos_e2_e3_nonnull": all(
            finite((e.get("grad_cos_mean") or {}).get("cos_e2_e3")) for e in st2),
        "stage13_not_logged": all(e["grad_logged_batches"] == 0 for e in st13)
        and all((e.get("grad_cos_mean") is None) for e in st13),
    }
    telemetry["per_seed"][f"seed{s}"] = {
        **checks,
        "stage4_detail": [{"epoch": e["epoch"], "grad_logged_batches": e["grad_logged_batches"],
                           "n_batches": e["n_batches"],
                           "grad_cos_mean": e.get("grad_cos_mean")} for e in st4],
    }
    telemetry["all_pass"] &= all(checks.values())
report["telemetry_pass_conditions"] = telemetry

# ---------------------------------------------------------------- 2. 配置不变性 + 性能对齐
cfg_keys = ("n_patients", "epochs", "lr", "batch", "hidden", "taus", "gate_hidden",
            "seq_len", "stage_schedule", "log_grad_every")
inv = {"per_seed": {}, "all_equal": True}
perf = {"per_seed": {}, "max_abs_diff": 0.0}
metric_paths = [("final_val", "val_admission", "auroc"), ("final_val", "val_admission", "auprc"),
                ("final_val", "val_hourly", "auroc"), ("final_val", "val_hourly", "auprc"),
                ("test", "admission", "auroc"), ("test", "admission", "auprc"),
                ("test", "hourly", "auroc"), ("test", "hourly", "auprc")]
for s in SEEDS:
    new, old = d7fix[s], s5_b[s]
    cfg_new = {k: new["config"][k] for k in cfg_keys}
    cfg_old = {k: old["config"][k] for k in cfg_keys}
    meta_new = {k: v for k, v in new["data"].items() if k != "build_sec"}
    meta_old = {k: v for k, v in old["data"].items() if k != "build_sec"}
    cfg_equal = cfg_new == cfg_old and meta_new == meta_old
    inv["per_seed"][f"seed{s}"] = {
        "config_equal": cfg_new == cfg_old, "data_meta_equal": meta_new == meta_old,
        "config": cfg_new}
    inv["all_equal"] &= cfg_equal
    diffs = {}
    for top, grp, key in metric_paths:
        nv = ((new.get(top) or {}).get(grp) or {}).get(key)
        ov = ((old.get(top) or {}).get(grp) or {}).get(key)
        d = abs(float(nv) - float(ov))
        diffs[f"{top}.{grp}.{key}"] = {"d7fix": nv, "s5": ov, "abs_diff": d}
        perf["max_abs_diff"] = max(perf["max_abs_diff"], d)
    perf["per_seed"][f"seed{s}"] = diffs
report["config_invariance_d10"] = inv
report["performance_alignment_d7fix_vs_s5"] = perf

# ---------------------------------------------------------------- 3. D7 修复前后对照
with open(os.path.join(BASE, "full_gradient_conflict.json"), encoding="utf-8") as f:
    grad_s5 = json.load(f)
before = {"stage4_epochs": 0, "stage4_logged_batches": 0, "stage4_defined_cosines": 0}
for s in SEEDS:
    for ep in grad_s5["runs"][f"full_B_seed{s}"]["per_epoch"]:
        if str(ep.get("stage")) == "4":
            before["stage4_epochs"] += 1
            before["stage4_logged_batches"] += int(ep.get("grad_logged_batches") or 0)
            before["stage4_defined_cosines"] += sum(
                v is not None for v in (ep.get("grad_cos_mean") or {}).values())
after = {"stage4_epochs": 0, "stage4_logged_batches": 0, "stage4_defined_cosines": 0}
for s in SEEDS:
    for ep in d7fix[s]["epochs_records"]:
        if ep["stage"] == 4:
            after["stage4_epochs"] += 1
            after["stage4_logged_batches"] += int(ep.get("grad_logged_batches") or 0)
            after["stage4_defined_cosines"] += sum(
                v is not None for v in (ep.get("grad_cos_mean") or {}).values())
report["d7_before_after_stage4"] = {
    "before_s5": before, "after_d7fix": after,
    "note": ("修复前 stage4 六 epoch 遥测全缺失（S5 根因：组名前缀计数 0）；"
             "修复后 6×22=132 个 logged batch、6×3=18 条定义良好余弦。")}

# ---------------------------------------------------------------- 4. H2(a) 预注册方向核对
def arm_epoch_cos(runs, stage=None, last_n=None):
    """返回 {seed: {pair: [每 epoch 值]}}；stage=None 取全部 epoch（A 全程）。"""
    out = {}
    for s in SEEDS:
        eps = runs[s]["epochs_records"]
        if stage is not None:
            eps = [e for e in eps if e["stage"] == stage]
        if last_n is not None:
            eps = eps[-last_n:]
        vals = {k: [] for k in PAIRS}
        for e in eps:
            cos = e.get("grad_cos_mean") or {}
            for k in PAIRS:
                if finite(cos.get(k)):
                    vals[k].append(float(cos[k]))
        out[s] = vals
    return out


a_all = arm_epoch_cos(s5_a)                       # A 全程（预注册口径）
a_last2 = arm_epoch_cos(s5_a, last_n=2)           # 稳健性视图（非预注册）
b_st4 = arm_epoch_cos(d7fix, stage=4)             # B stage4 校准期（d7fix）
b_st2_s5 = arm_epoch_cos(s5_b, stage=2)           # S5 B stage2（旧对照，参考）

h2a = {"primary_A_full_vs_B_stage4": {}, "robustness_A_last2_vs_B_stage4": {},
       "reference_B_stage2_s5": {}}
for k in PAIRS:
    diffs, a_means, b_means = [], [], []
    for s in SEEDS:
        am = sum(a_all[s][k]) / len(a_all[s][k])
        bm = sum(b_st4[s][k]) / len(b_st4[s][k])
        a_means.append(am)
        b_means.append(bm)
        diffs.append(am - bm)
    h2a["primary_A_full_vs_B_stage4"][k] = {
        "A_mean_per_seed": a_means, "B_stage4_mean_per_seed": b_means,
        "paired_diff_A_minus_Bstage4_per_seed": diffs,
        "paired_diff_mean_std": mean_std(diffs),
        "n_seeds_A_more_negative": sum(1 for d in diffs if d < 0)}
    rd = []
    for s in SEEDS:
        am = sum(a_last2[s][k]) / len(a_last2[s][k])
        bm = sum(b_st4[s][k]) / len(b_st4[s][k])
        rd.append(am - bm)
    h2a["robustness_A_last2_vs_B_stage4"][k] = {
        "paired_diff_A_last2_minus_Bstage4_mean_std": mean_std(rd)}
    ref_vals = [v for s in SEEDS for v in b_st2_s5[s][k]]
    if ref_vals:  # stage2 仅 τ₂/τ₃ 联合对有定义，e1 对为 None（S5 行为不变）
        h2a["reference_B_stage2_s5"][k] = mean_std(ref_vals)
    else:
        h2a["reference_B_stage2_s5"][k] = {"note": "stage2 无定义（专家未联合可训）"}

pooled_a = {k: [v for s in SEEDS for v in a_all[s][k]] for k in PAIRS}
pooled_b4 = {k: [v for s in SEEDS for v in b_st4[s][k]] for k in PAIRS}
h2a["pooled_epoch_level"] = {
    "A_full_60_epochs": {k: {**mean_std(pooled_a[k]),
                             "min": min(pooled_a[k]), "max": max(pooled_a[k]),
                             "max_abs": max(abs(v) for v in pooled_a[k])} for k in PAIRS},
    "B_stage4_6_epochs": {k: {**mean_std(pooled_b4[k]),
                              "min": min(pooled_b4[k]), "max": max(pooled_b4[k]),
                              "max_abs": max(abs(v) for v in pooled_b4[k])} for k in PAIRS},
    "note": ("A 臂每 epoch 值为 22 个 batch 均值的 epoch 级 cos；B stage4 同口径。"
             "max|cos| 量级对照=S5 曾报 A 0.017018 vs B-stage2 0.002229（不同阶段/"
             "对照，不作替代）。")}

n_cells_neg = sum(h2a["primary_A_full_vs_B_stage4"][k]["n_seeds_A_more_negative"]
                  for k in PAIRS)
all_pairs_negative = all(
    h2a["primary_A_full_vs_B_stage4"][k]["paired_diff_mean_std"]["mean"] < 0
    for k in PAIRS)
h2a["direction_summary"] = {
    "cells_A_more_negative": f"{n_cells_neg}/9（3 种子×3 专家对）",
    "all_three_pairs_mean_diff_negative": bool(all_pairs_negative),
    "preregistered_direction": ("design §二 H2(a)：A 臂训练期专家对梯度余弦比 B 臂阶段四"
                                "校准期更负（A−B-stage4<0）")}
report["h2a_adjudication"] = h2a

# ---------------------------------------------------------------- 5. verdict
telemetry_ok = telemetry["all_pass"] and inv["all_equal"]
if not telemetry_ok:
    h2a_verdict = "inconclusive"
    h2a_note = ("通过条件未满足（遥测不完整或配置漂移）→ 修复未生效/数据无效，"
                "不带证据下方向结论；缺失不当零值。")
elif all_pairs_negative:
    h2a_verdict = "supported"
    h2a_note = ("A 全程专家对余弦三对均比 B-stage4 更负（预注册方向成立）；"
                "但 |cos| 量级≈0（见 pooled_epoch_level.max_abs），为方向性机制证据，"
                "效应量小。H2 overall 仍 inconclusive（H1 前提已 falsified；"
                "本补测不重开 H1 门线、不追认预注册 H1）。")
else:
    h2a_verdict = "inconclusive"
    h2a_note = ("遥测已补齐但 A vs B-stage4 不呈预注册「A 更负」方向 → H2(a) 不支持；"
                "按预注册边界记 inconclusive，不把缺失/无冲突当零值。")
report["verdict"] = {
    "telemetry_complete": bool(telemetry_ok),
    "h2a_direction_supported": bool(all_pairs_negative) if telemetry_ok else None,
    "h2a": h2a_verdict,
    "h2_overall": ("inconclusive / not established（H1 已 falsified，H2 前提不成立；"
                   "H2(b) 读数 +0.107033±0.001550 作为机制事实记录，不提升为 H2 支持）"),
    "h2a_note": h2a_note,
    "h1_gates": "不重开（S6 已终判 falsified；本单元只补 H2(a) 机制证据）",
}

# ---------------------------------------------------------------- 6. 新 tag D7 日志（不覆盖 S5）
grad_out = {
    "schema": "full-gradient-conflict/v1",
    "stage": "S6-final-verdict-redesign",
    "note": ("D7 修复后补跑（tag=d7fix，run_id=full_B_seed{0,1,2}_d7fix）：stage4 六 epoch "
             "全量 22 batch 记录、三对 cos 非 null。与 S5 full_gradient_conflict.json "
             "（修复前，stage4 全 null）并存，不覆盖旧件。"),
    "runs": {}}
for s in SEEDS:
    r = d7fix[s]
    grad_out["runs"][r["run_id"]] = {
        "arm": r["arm"], "seed": r["seed"], "status": r["status"],
        "per_epoch": [{"epoch": e["epoch"], "stage": e["stage"],
                       "grad_logged_batches": e.get("grad_logged_batches"),
                       "expert_grad_norm_mean": e.get("expert_grad_norm_mean"),
                       "grad_cos_mean": e.get("grad_cos_mean")}
                      for e in r["epochs_records"]]}
with open(os.path.join(BASE, "full_gradient_conflict_d7fix.json"), "w", encoding="utf-8") as f:
    json.dump(grad_out, f, ensure_ascii=False, indent=2)

out_path = os.path.join(BASE, "_d7fix_recompute.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(json.dumps({
    "telemetry_all_pass": telemetry["all_pass"],
    "config_invariance_all_equal": inv["all_equal"],
    "performance_max_abs_diff_d7fix_vs_s5": perf["max_abs_diff"],
    "d7_before_after": report["d7_before_after_stage4"],
    "h2a_direction": h2a["direction_summary"],
    "paired_diff_per_pair": {k: h2a["primary_A_full_vs_B_stage4"][k]["paired_diff_mean_std"]
                             for k in PAIRS},
    "pooled_max_abs": {"A": h2a["pooled_epoch_level"]["A_full_60_epochs"],
                       "B_stage4": h2a["pooled_epoch_level"]["B_stage4_6_epochs"]},
    "verdict": report["verdict"],
}, ensure_ascii=False, indent=2))
print("WROTE results/_d7fix_recompute.json + results/full_gradient_conflict_d7fix.json")
