#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6 independent adjudication recompute from S5 raw run JSON + CSV + D7 logs.
Std is sample SD (ddof=1); no model training or result generation is performed.
"""
import csv
import json
import math
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def read_csv(name):
    with open(os.path.join(BASE, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean_std(values):
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {"mean": mean, "std": sd, "n": n}


def get_metric(run, metric, split="val"):
    if split == "val":
        if metric.startswith("adm_"):
            node = run["final_val"]["val_admission"]
        else:
            node = run["final_val"]["val_hourly"]
        key = metric.split("_", 1)[1]
        return float(node[key])
    if metric.startswith("adm_"):
        node = run["test"]["admission"]
    else:
        node = run["test"]["hourly"]
    key = metric.split("_", 1)[1]
    return float(node[key])


def get_csv_metric(row, metric, split="val"):
    prefix = "val_" if split == "val" else "test_"
    column = prefix + metric
    return float(row[column])


rows_ab = read_csv("full_joint_vs_staged.csv")
rows_bn = read_csv("full_bnocal.csv")
all_rows = rows_ab + rows_bn
assert len(rows_ab) == 6 and len(rows_bn) == 3, (len(rows_ab), len(rows_bn))
assert len({r["run_id"] for r in all_rows}) == 9

raw = {}
for row in all_rows:
    run_id = row["run_id"]
    with open(os.path.join(BASE, "full_runs", run_id + ".json"), encoding="utf-8") as f:
        run = json.load(f)
    assert run["run_id"] == run_id and run["status"] == "done"
    assert int(run["config"]["epochs"]) == int(row["epochs_done"])
    assert int(run["data"]["n_used"]) == int(row["n_patients_used"]) == 2000
    assert run["final_val"]["val_admission"]["n_pos"] == 9
    raw[run_id] = run

arms = {
    "A": sorted([r for r in all_rows if r["arm"] == "A"], key=lambda r: int(r["seed"])),
    "B": sorted([r for r in all_rows if r["arm"] == "B"], key=lambda r: int(r["seed"])),
    "BnoCal": sorted([r for r in all_rows if r["arm"] == "BnoCal"], key=lambda r: int(r["seed"])),
}
assert all(len(arms[a]) == 3 for a in arms)

metric_names = ["adm_auroc", "adm_auprc", "hourly_auroc", "hourly_auprc"]
arm_metrics = {}
for arm, arm_rows in arms.items():
    arm_metrics[arm] = {}
    for metric in metric_names:
        vals = [get_csv_metric(r, metric, "val") for r in arm_rows]
        arm_metrics[arm][metric] = {**mean_std(vals), "per_seed": vals}

paired = {}
raw_arm_metrics = {}
for label, left, right in [("B_minus_A", "B", "A"),
                           ("B_minus_BnoCal", "B", "BnoCal"),
                           ("A_minus_BnoCal", "A", "BnoCal")]:
    paired[label] = {}
    right_by_seed = {int(r["seed"]): r for r in arms[right]}
    for metric in metric_names:
        raw_vals = [get_metric(raw[r["run_id"]], metric, "val") for r in arms[left]]
        raw_right = {int(r["seed"]): get_metric(raw[r["run_id"]], metric, "val")
                     for r in arms[right]}
        raw_diffs = [value - raw_right[int(r["seed"])]
                     for value, r in zip(raw_vals, arms[left])]
        diffs = []
        for r in arms[left]:
            seed = int(r["seed"])
            diffs.append(get_csv_metric(r, metric, "val") -
                         get_csv_metric(right_by_seed[seed], metric, "val"))
        paired[label][metric] = {**mean_std(diffs), "per_seed": diffs}
        raw_arm_metrics.setdefault(label, {})[metric] = {**mean_std(raw_diffs), "per_seed": raw_diffs}

raw_marginal_metrics = {}
for arm, arm_rows in arms.items():
    raw_marginal_metrics[arm] = {}
    for metric in metric_names:
        vals = [get_metric(raw[r["run_id"]], metric, "val") for r in arm_rows]
        raw_marginal_metrics[arm][metric] = {**mean_std(vals), "per_seed": vals}

# CSV is the ground-truth table required by the stage; verify each displayed value
# against its raw full-run JSON value (four-decimal CSV export tolerance).
csv_checks = []
for row in all_rows:
    run = raw[row["run_id"]]
    mapping = {
        "val_adm_auroc": get_metric(run, "adm_auroc", "val"),
        "val_adm_auprc": get_metric(run, "adm_auprc", "val"),
        "val_hourly_auroc": get_metric(run, "hourly_auroc", "val"),
        "val_hourly_auprc": get_metric(run, "hourly_auprc", "val"),
        "test_adm_auroc": get_metric(run, "adm_auroc", "test"),
        "test_adm_auprc": get_metric(run, "adm_auprc", "test"),
        "test_hourly_auroc": get_metric(run, "hourly_auroc", "test"),
        "test_hourly_auprc": get_metric(run, "hourly_auprc", "test"),
    }
    for key, value in mapping.items():
        delta = abs(float(row[key]) - value)
        csv_checks.append({"run_id": row["run_id"], "metric": key, "abs_diff": delta})
assert max(x["abs_diff"] for x in csv_checks) <= 0.000051

h1 = {
    "gate_1_delta_adm_auroc_ge_0_01": raw_arm_metrics["B_minus_A"]["adm_auroc"]["mean"] >= 0.01,
    "gate_2_delta_adm_auprc_ge_minus_0_01": raw_arm_metrics["B_minus_A"]["adm_auprc"]["mean"] >= -0.01,
    "gate_3_std_B_le_std_A": raw_marginal_metrics["B"]["adm_auroc"]["std"] <= raw_marginal_metrics["A"]["adm_auroc"]["std"],
}
h1["all_gates_pass"] = all(h1.values())

with open(os.path.join(BASE, "full_gradient_conflict.json"), encoding="utf-8") as f:
    grad = json.load(f)
grad_counts = {}
for run_id, run in grad["runs"].items():
    arm = run["arm"]
    a = grad_counts.setdefault(arm, {"runs": 0, "epochs": 0, "defined_cosines": 0,
                                     "stage4_epochs": 0, "stage4_logged_batches": 0,
                                     "stage4_defined_cosines": 0, "stage2_epochs": 0,
                                     "stage2_defined_cosines": 0})
    a["runs"] += 1
    for ep in run.get("per_epoch", []):
        a["epochs"] += 1
        cos = ep.get("grad_cos_mean") or {}
        defined = sum(v is not None for v in cos.values())
        a["defined_cosines"] += defined
        stage = str(ep.get("stage"))
        if arm == "B" and stage == "4":
            a["stage4_epochs"] += 1
            a["stage4_logged_batches"] += int(ep.get("grad_logged_batches") or 0)
            a["stage4_defined_cosines"] += defined
        if arm == "B" and stage == "2":
            a["stage2_epochs"] += 1
            a["stage2_defined_cosines"] += defined
assert grad_counts["A"]["defined_cosines"] == 180
assert grad_counts["B"]["stage4_epochs"] == 6
assert grad_counts["B"]["stage4_logged_batches"] == 0
assert grad_counts["B"]["stage4_defined_cosines"] == 0
assert grad_counts["B"]["stage2_epochs"] == 18
assert grad_counts["B"]["stage2_defined_cosines"] == 18

with open(os.path.join(BASE, "full_progress.json"), encoding="utf-8") as f:
    progress = json.load(f)
# 2026-10-29 S7-repo-package 兼容性修复（不改变任何数值）：
# d7fix 补跑（tag=full_B_seed{0,1,2}_d7fix）与 S5 共用 full_progress.json，文件现有
# 12 键。H1 复算作用域=CSV 主表内的 9 个 S5 run；d7fix 条目同数据同划分（其遥测复算在
# _d7fix_recompute.py），不参与 H1 主表。原断言 len(progress["runs"]) == 9 在此变为
# 过强，改为按 CSV run 集核对并记录文件总键数。
progress_check = {"n_runs_in_file": len(progress["runs"]), "n_runs_h1": len(all_rows),
                  "all_done": True, "n_pos_by_run": {}, "epochs_by_run": {}}
for row in all_rows:
    run_id = row["run_id"]
    run = progress["runs"][run_id]
    progress_check["all_done"] &= run["status"] == "done"
    progress_check["epochs_by_run"][run_id] = run["n_epochs_done"]
    npos = sorted({e["val_admission"]["n_pos"] for e in run["epochs"]})
    progress_check["n_pos_by_run"][run_id] = npos
assert progress_check["n_runs_h1"] == 9 and progress_check["n_runs_in_file"] >= 9
assert progress_check["all_done"]
assert all(v == [9] for v in progress_check["n_pos_by_run"].values())

# Test-set paired AUROC is descriptive only; the pre-registered H1 primary gates use validation.
by_seed_a = {int(r["seed"]): r for r in arms["A"]}
test_diffs = [get_csv_metric(r, "adm_auroc", "test") -
              get_csv_metric(by_seed_a[int(r["seed"])], "adm_auroc", "test")
              for r in arms["B"]]
test_delta = mean_std(test_diffs)

result = {
    "schema": "s6-independent-recompute/v1",
    "method": "Read raw results/full_runs/*.json and CSV table; sample SD ddof=1; assert CSV within 5.1e-5; no training rerun.",
    "source_files": ["results/full_joint_vs_staged.csv", "results/full_bnocal.csv",
                     "results/full_runs/full_*.json", "results/full_gradient_conflict.json",
                     "results/full_progress.json"],
    "configuration": {"n_runs": 9, "all_done": progress_check["all_done"],
                       "patients_per_run": 2000, "epochs": {"A": 20, "B": 20, "BnoCal": 18},
                       "val_admissions": 200, "val_positive_admissions_per_run": 9},
    "arm_metrics_val": arm_metrics,
    "paired_deltas_val": paired,
    "arm_metrics_val_raw_json": raw_marginal_metrics,
    "paired_deltas_val_raw_json": raw_arm_metrics,
    "h1_gates": h1,
    "test_b_minus_a_admission_auroc_descriptive": test_delta,
    "d7_counts": grad_counts,
    "progress_check": progress_check,
    "csv_raw_json_check": {"n_values": len(csv_checks), "max_abs_diff": max(x["abs_diff"] for x in csv_checks),
                            "all_within_5_1e-5": True},
}
output = os.path.join(BASE, "_s6_recompute.json")
with open(output, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
print("WROTE results/_s6_recompute.json")
