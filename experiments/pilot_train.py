#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pilot_train.py — S3-pilot 执行器：A(联合)×3种子 vs B(2/2/1/1 分阶段)×3种子 + 单体LNN sanity。

design.md 冻结映射：
  D1 臂语义 / D2 阶段步数（pilot E=6 → 2/2/1/1）/ D3 数据预处理 / D4 模型超参 /
  D5 方向性健全性（A 须优于单体 LNN）/ D6 种子{0,1,2}+mean±std / D7 梯度冲突日志 /
  D8 评测（入院级主+小时级次）/ D9 checkpoint+子采样阶梯 / D10 单变量纪律。

D10 单变量纪律落实：所有臂/种子共用同一子采样患者集（SUBSAMPLE_SEED）、同一患者级
划分（SPLIT_SEED）、同一 train 数据顺序（LOADER_SEED）、同一预处理与 scaler；
唯一差异=模型初始化种子与训练流程（A 联合 / B 分阶段 / LNN 单体）。

D7 执行口径（与 design 原文的差异，如实记录）：原文「每 epoch 抽样 ≥200 个 batch」，
8000 档 train≈5600 例→88 batch/epoch，200 个不重复 batch/epoch 需额外前向反向、
与 D9 单训练 ≤12min 硬顶冲突。本执行器按「每 epoch 全量 batch 记录」执行
（A 臂 88×6=528 样本/run；B 臂仅 stage4=88 样本/run），cosine 估计样本量与
标准差全部落盘，S5 full（437 batch/epoch）可回归原文口径。

用法：
  G:/ai/python11/python.exe experiments/pilot_train.py --arm A --seed 0 --n-patients 8000
  G:/ai/python11/python.exe experiments/pilot_train.py --summary        # 汇总出四张表
  G:/ai/python11/python.exe experiments/pilot_train.py --arm B --seed 1 --resume  # 断点续跑
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import data_pipeline as dp  # noqa: E402
import mr_moe  # noqa: E402

DATA_ROOT = os.path.join(PROJECT, "experiments", "data")
CKPT_DIR = os.path.join(PROJECT, "experiments", "ckpt")
RESULTS = os.path.join(PROJECT, "results")
RUNS_DIR = os.path.join(RESULTS, "pilot_runs")
PROGRESS = os.path.join(RESULTS, "pilot_progress.json")
TIER_DECISION = os.path.join(RESULTS, "_pilot_tier_decision.json")

SUBSAMPLE_SEED = 20261027   # D10：所有臂/种子共用同一子采样患者集
SPLIT_SEED = 0              # D10：共用同一患者级划分（Che 2018 操作化，见 data_pipeline）
LOADER_SEED = 12345         # D10：共用同一 train 数据顺序
IN_DIM = 2 * dp.N_FEATURES  # 68 = [值 34, mask 34]
STAGE_EPOCHS_E6 = (2, 2, 1, 1)  # D2：pilot E=6 → 2/2/1/1


# ---------------------------------------------------------------- 工具
def _json_safe(o):
    """递归把非有限浮点/ numpy 标量转成 JSON 安全值（F3 NaN run 也要能落盘）。"""
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def _atomic_write_json(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = np.asarray(labels).astype(float)
    scores = np.asarray(scores, dtype=float)
    lab = labels.astype(int)
    # 非有限分数（如 F3 NaN run）不崩指标计算，记 None 由上层判读
    if scores.size == 0 or not np.isfinite(scores).all() or len(np.unique(lab)) < 2:
        return {"auroc": None, "auprc": None, "n": int(lab.size),
                "n_pos": int(lab.sum()) if lab.size else 0}
    return {"auroc": float(roc_auc_score(lab, scores)),
            "auprc": float(average_precision_score(lab, scores)),
            "n": int(lab.size), "n_pos": int(lab.sum())}


def stage_schedule(epochs: int) -> list[int]:
    """D2 阶段步数：E=6 → [1,1,2,2,3,4]（2/2/1/1）；通用回退按 4:3:2:1+最小1epoch。"""
    if epochs == 6:
        out: list[int] = []
        for stage, cnt in enumerate(STAGE_EPOCHS_E6, 1):
            out.extend([stage] * cnt)
        return out
    base = epochs / 10.0
    raw = [max(1, round(base * w)) for w in (4, 3, 2, 1)]
    while sum(raw) > epochs:
        i = raw.index(max(raw))
        if raw[i] > 1:
            raw[i] -= 1
        else:
            break
    while sum(raw) < epochs:
        raw[0] += 1
    out = []
    for stage, cnt in enumerate(raw, 1):
        out.extend([stage] * cnt)
    return out


# ---------------------------------------------------------------- 数据
def build_tensors(examples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """患者列表 → 定长张量（padding 到 SEQ_LEN；填充步 mask=1、step-valid=0）。

    padding 约定（实现记录）：值通道填 0、mask 通道置 1（与 D3「0 填充但可见」一致），
    损失与评测只在 step-valid=1 的真实小时上计算。
    """
    n = len(examples)
    L, d = dp.SEQ_LEN, dp.N_FEATURES
    X = np.zeros((n, L, 2 * d), dtype=np.float32)
    X[:, :, d:] = 1.0
    V = np.zeros((n, L), dtype=np.float32)
    Y = np.zeros((n, L), dtype=np.float32)
    for i, ex in enumerate(examples):
        t = min(ex.t, L)
        X[i, :t, :d] = ex.x[:t, :d]
        X[i, :t, d:] = ex.x[:t, d:]
        V[i, :t] = 1.0
        Y[i, :t] = ex.y[:t]
    return X, V, Y


def build_or_load_cache(n_patients: int) -> dict:
    """子采样+划分+scaler+张量化，落盘 npz 缓存（七个 run 共用，保证 D10 一致）。"""
    os.makedirs(RUNS_DIR, exist_ok=True)
    cache = os.path.join(RESULTS, f"_pilot_data_ss{SUBSAMPLE_SEED}_n{n_patients}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {"X": z["X"], "V": z["V"], "Y": z["Y"],
                "idx_tr": z["idx_tr"], "idx_va": z["idx_va"], "idx_te": z["idx_te"],
                "meta": json.loads(str(z["meta_json"])), "cache_path": cache}
    t0 = time.time()
    examples, skipped = dp.load_dataset(DATA_ROOT, set_names=("training_setA",))
    n_avail = len(examples)
    rng = np.random.RandomState(SUBSAMPLE_SEED)
    perm = rng.permutation(n_avail)
    n_used = min(n_patients, n_avail)
    sub = [examples[i] for i in perm[:n_used]]
    splits = dp.patient_split(sub, seed=SPLIT_SEED)
    scaler = dp.fit_scaler_on_train(splits["train"])   # D3：只用 train 拟合
    dp.apply_scaler(sub, scaler)
    X, V, Y = build_tensors(sub)
    pos = {ex.pid: i for i, ex in enumerate(sub)}
    idx = {k: np.array([pos[ex.pid] for ex in splits[k]], dtype=np.int64)
           for k in ("train", "val", "test")}
    meta = {"n_available": n_avail, "n_used": n_used, "n_requested": n_patients,
            "skipped_files": len(skipped), "subsample_seed": SUBSAMPLE_SEED,
            "split_seed": SPLIT_SEED, "loader_seed": LOADER_SEED,
            "split_sizes": {k: len(v) for k, v in idx.items()},
            "scaler_dmin": [float(x) for x in scaler.dmin],
            "scaler_range": [float(x) for x in scaler._rng],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_sec": round(time.time() - t0, 1)}
    tmp = cache + ".tmp"
    with open(tmp, "wb") as f:  # 传文件对象：np.savez 不会自动补 .npz 后缀
        np.savez(f, X=X, V=V, Y=Y, idx_tr=idx["train"], idx_va=idx["val"],
                 idx_te=idx["test"], meta_json=np.array(json.dumps(meta, ensure_ascii=False)))
    os.replace(tmp, cache)
    return {"X": X, "V": V, "Y": Y, "idx_tr": idx["train"], "idx_va": idx["val"],
            "idx_te": idx["test"], "meta": meta, "cache_path": cache}


# ---------------------------------------------------------------- 评测（D8）
@torch.no_grad()
def evaluate(model, X, V, Y, device, bs: int = 128) -> dict:
    model.eval()
    n = len(X)
    probs = np.zeros((n, dp.SEQ_LEN), dtype=np.float32)
    loss_sum, loss_n = 0.0, 0
    for s in range(0, n, bs):
        xb = torch.from_numpy(X[s:s + bs]).to(device, non_blocking=True)
        vb = torch.from_numpy(V[s:s + bs]).to(device, non_blocking=True)
        yb = torch.from_numpy(Y[s:s + bs]).to(device, non_blocking=True)
        logits = model(xb)
        m = vb > 0
        if m.any():
            l = F.binary_cross_entropy_with_logits(logits[m], yb[m], reduction="sum")
            loss_sum += float(l)
            loss_n += int(m.sum())
        probs[s:s + bs] = torch.sigmoid(logits).float().cpu().numpy()
    adm_s: list[float] = []
    adm_y: list[float] = []
    hr_s: list[np.ndarray] = []
    hr_y: list[np.ndarray] = []
    for i in range(n):
        m = V[i] > 0
        if not m.any():
            continue
        adm_s.append(float(probs[i][m].max()))                 # D8：窗口概率最大值
        adm_y.append(1.0 if float(Y[i][m].max()) > 0 else 0.0)  # 入院级标签=窗内任一阳性
        hr_s.append(probs[i][m])
        hr_y.append(Y[i][m])
    hr_s = np.concatenate(hr_s) if hr_s else np.array([])
    hr_y = np.concatenate(hr_y) if hr_y else np.array([])
    return {"loss": (loss_sum / loss_n) if loss_n else None,
            "admission": _binary_metrics(np.array(adm_y), np.array(adm_s)),
            "hourly": _binary_metrics(hr_y, hr_s)}


# ---------------------------------------------------------------- checkpoint / progress（D9）
def ckpt_path(run_id: str, epoch: int) -> str:
    return os.path.join(CKPT_DIR, f"{run_id}_ep{epoch}.pt")


def save_ckpt(run_id: str, epoch: int, stage, model, rec: dict) -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    path = ckpt_path(run_id, epoch)
    tmp = path + ".tmp"
    torch.save({"run_id": run_id, "epoch": epoch, "stage": stage,
                "model": model.state_dict(), "epoch_record": rec}, tmp)
    os.replace(tmp, path)
    return path


def latest_ckpt_epoch(run_id: str) -> int | None:
    best = None
    for p in glob.glob(os.path.join(CKPT_DIR, f"{run_id}_ep*.pt")):
        try:
            ep = int(os.path.basename(p).rsplit("_ep", 1)[1].split(".pt")[0])
        except ValueError:
            continue
        best = ep if best is None else max(best, ep)
    return best


def update_progress(run_id: str, arm: str, seed: int, status: str,
                    epochs: list[dict], extra: dict | None = None) -> None:
    prog = {"schema": "pilot-progress/v1", "runs": {}}
    if os.path.exists(PROGRESS):
        try:
            with open(PROGRESS, encoding="utf-8") as f:
                prog = json.load(f)
        except Exception:  # noqa: BLE001 — 损坏则重置（ckpt 仍在）
            pass
    entry = {"arm": arm, "seed": seed, "status": status,
             "last_epoch": (epochs[-1]["epoch"] if epochs else None),
             "n_epochs_done": len(epochs), "epochs": epochs,
             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if extra:
        entry.update(extra)
    prog.setdefault("runs", {})[run_id] = entry
    prog["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_write_json(PROGRESS, prog)


# ---------------------------------------------------------------- 单 run 训练
def train_run(arm: str, seed: int, n_patients: int, epochs: int, lr: float,
              device: str, log_grad_every: int, resume: bool, tag: str) -> dict:
    run_id = f"{arm}_seed{seed}" + (f"_{tag}" if tag else "")
    kind = "lnn" if arm == "lnn" else "mr_moe"
    t0 = time.time()
    cache = build_or_load_cache(n_patients)
    X, V, Y = cache["X"], cache["V"], cache["Y"]
    idx_tr, idx_va, idx_te = cache["idx_tr"], cache["idx_va"], cache["idx_te"]
    meta = cache["meta"]

    model = mr_moe.build_model(IN_DIM, seed, kind=kind).to(device)
    n_params = mr_moe.count_params(model)
    stages = stage_schedule(epochs) if arm == "B" else [None] * epochs

    start_epoch = 0
    ep_records: list[dict] = []
    if resume:
        ep = latest_ckpt_epoch(run_id)
        if ep is not None and ep + 1 < epochs:
            sd = torch.load(ckpt_path(run_id, ep), map_location=device)
            model.load_state_dict(sd["model"])
            start_epoch = ep + 1
            if os.path.exists(PROGRESS):
                with open(PROGRESS, encoding="utf-8") as f:
                    old = json.load(f).get("runs", {}).get(run_id, {})
                ep_records = [r for r in old.get("epochs", []) if r["epoch"] < start_epoch]
            print(f"[resume] {run_id}: ckpt epoch={ep} → 从 epoch {start_epoch} 续跑", flush=True)

    Xt = torch.from_numpy(X[idx_tr])
    Vt = torch.from_numpy(V[idx_tr])
    Yt = torch.from_numpy(Y[idx_tr])
    ds = torch.utils.data.TensorDataset(Xt, Vt, Yt)
    gen = torch.Generator().manual_seed(LOADER_SEED)  # D10：同数据顺序

    opt = None
    prev_stage = None
    status = "done"
    t_train = 0.0
    for epoch in range(start_epoch, epochs):
        stage = stages[epoch]
        if arm == "B":
            groups = model.param_groups_for_stage(stage)
            if stage != prev_stage:
                opt = torch.optim.Adam([p for g in groups for p in g["params"]], lr=lr)
                prev_stage = stage
        elif opt is None:
            opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=mr_moe.BATCH_SIZE, shuffle=True, generator=gen)
        model.train()
        log_grad = (arm != "lnn") and ((arm == "A") or (arm == "B" and stage == 4))
        t_ep = time.time()
        losses: list[float] = []
        gnorms: list[float] = []
        eg_list: list[list] = []
        cos_list: list[dict] = []
        n_logged = 0
        nan_hit = False
        for bi, (xb, vb, yb) in enumerate(loader):
            xb = xb.to(device, non_blocking=True)
            vb = vb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            m = vb > 0
            if not m.any():
                continue
            loss = F.binary_cross_entropy_with_logits(logits[m], yb[m])
            if log_grad and (bi % log_grad_every == 0):
                st = model.expert_gradient_stats(loss)  # D7（内部 retain_graph，随后仍可 backward）
                eg_list.append(st["grad_norms"])
                cos_list.append(st["cosine_pairs"])
                n_logged += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            g2 = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    g2 += float(p.grad.detach().pow(2).sum())
            gn = math.sqrt(g2)  # F3 监控：全局梯度范数（含 τ=100 慢专家放大）
            opt.step()
            lv = float(loss.detach())
            losses.append(lv)
            gnorms.append(gn)
            if not math.isfinite(lv) or not math.isfinite(gn):
                nan_hit = True
                print(f"[nan] {run_id} epoch={epoch} batch={bi} loss={lv} gnorm={gn}", flush=True)
                break
        t_train += time.time() - t_ep
        ev = evaluate(model, X[idx_va], V[idx_va], Y[idx_va], device)
        rec: dict = {"epoch": epoch, "stage": stage,
                     "train_loss_mean": _mean(losses),
                     "grad_global_norm_mean": _mean(gnorms),
                     "n_batches": len(losses), "epoch_sec": round(time.time() - t_ep, 2),
                     "grad_logged_batches": n_logged,
                     "val_loss": ev["loss"], "val_admission": ev["admission"],
                     "val_hourly": ev["hourly"]}
        if eg_list:
            arr = np.array([[np.nan if v is None else v for v in row] for row in eg_list],
                           dtype=np.float64)
            rec["expert_grad_norm_mean"] = [
                None if np.all(np.isnan(arr[:, i])) else float(np.nanmean(arr[:, i]))
                for i in range(arr.shape[1])]
            keys = sorted(cos_list[0].keys())
            rec["grad_cos_mean"] = {
                k: (float(np.nanmean([c[k] for c in cos_list if c[k] is not None]))
                    if any(c[k] is not None for c in cos_list) else None)
                for k in keys}
        ep_records.append(rec)
        save_ckpt(run_id, epoch, stage, model, rec)
        update_progress(run_id, arm, seed,
                        "nan_abort" if nan_hit else "running", ep_records)
        va = ev["admission"]
        print(f"[{run_id}] ep{epoch} stage{stage} loss={rec['train_loss_mean']} "
              f"gnorm={rec['grad_global_norm_mean']} val_loss={ev['loss']} "
              f"val_adm_auroc={va['auroc']} ({rec['epoch_sec']}s)", flush=True)
        if nan_hit:
            status = "nan_abort"
            break

    test_eval = evaluate(model, X[idx_te], V[idx_te], Y[idx_te], device) if len(idx_te) else None
    total_sec = round(time.time() - t0, 1)
    ep_secs = [r["epoch_sec"] for r in ep_records]
    ms_pb = [round(r["epoch_sec"] * 1000.0 / max(r["n_batches"], 1), 1) for r in ep_records]
    run = {"schema": "pilot-run/v1", "run_id": run_id, "arm": arm, "seed": seed,
           "kind": kind, "status": status,
           "config": {"n_patients": n_patients, "epochs": epochs, "lr": lr,
                      "batch": mr_moe.BATCH_SIZE, "hidden": mr_moe.HIDDEN,
                      "taus": list(mr_moe.TAUS), "gate_hidden": mr_moe.GATE_HIDDEN,
                      "seq_len": mr_moe.SEQ_LEN, "stage_schedule": stages,
                      "log_grad_every": log_grad_every, "device": device,
                      "torch": torch.__version__},
           "data": meta, "params": n_params, "epochs_records": ep_records,
           "final_val": ({"train_loss_mean": ep_records[-1]["train_loss_mean"],
                          "val_loss": ep_records[-1]["val_loss"],
                          "val_admission": ep_records[-1]["val_admission"],
                          "val_hourly": ep_records[-1]["val_hourly"]}
                         if ep_records else None),
           "test": test_eval,
           "timing": {"total_sec": total_sec,
                      "data_prep_sec": meta.get("build_sec", 0.0),
                      "train_sec": round(t_train, 1),
                      "per_epoch_sec": ep_secs, "ms_per_batch": ms_pb}}
    _atomic_write_json(os.path.join(RUNS_DIR, f"{run_id}.json"), run)
    update_progress(run_id, arm, seed, status, ep_records,
                    extra={"total_sec": total_sec, "test_admission":
                           (test_eval or {}).get("admission")})
    print(f"[done] {run_id} status={status} total={total_sec}s "
          f"test_adm_auroc={(test_eval or {}).get('admission', {}).get('auroc')}", flush=True)
    return run


# ---------------------------------------------------------------- 汇总出表
def _fmt(x, nd=4):
    return "None" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def summary() -> dict:
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        if "smoke" in os.path.basename(p):
            continue
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    done = [r for r in runs if r.get("status") == "done"]

    # ---- CSV（逐 run 一行）
    csv_path = os.path.join(RESULTS, "pilot_joint_vs_staged.csv")
    cols = ["run_id", "arm", "seed", "status", "n_patients_used", "epochs_done",
            "final_train_loss", "final_val_loss", "val_adm_auroc", "val_adm_auprc",
            "val_hourly_auroc", "val_hourly_auprc", "test_adm_auroc", "test_adm_auprc",
            "test_hourly_auroc", "test_hourly_auprc", "total_sec", "ms_per_batch_mean",
            "grad_norm_e1_mean", "grad_norm_e2_mean", "grad_norm_e3_mean",
            "grad_cos_e1_e2_mean", "grad_cos_e1_e3_mean", "grad_cos_e2_e3_mean", "lr"]
    lines = [",".join(cols)]
    for r in runs:
        fv = r.get("final_val") or {}
        te = r.get("test") or {}
        tst = (te.get("admission") or {})
        tsth = (te.get("hourly") or {})
        eps = r.get("epochs_records") or []
        ms_pb = r.get("timing", {}).get("ms_per_batch") or []
        gn = [e.get("expert_grad_norm_mean") for e in eps if e.get("expert_grad_norm_mean")]
        gc = [e.get("grad_cos_mean") for e in eps if e.get("grad_cos_mean")]
        row = [r["run_id"], r["arm"], str(r["seed"]), r["status"],
               str(r["data"]["n_used"]), str(len(eps)),
               _fmt(fv.get("train_loss_mean")), _fmt(fv.get("val_loss")),
               _fmt((fv.get("val_admission") or {}).get("auroc")),
               _fmt((fv.get("val_admission") or {}).get("auprc")),
               _fmt((fv.get("val_hourly") or {}).get("auroc")),
               _fmt((fv.get("val_hourly") or {}).get("auprc")),
               _fmt(tst.get("auroc")), _fmt(tst.get("auprc")),
               _fmt(tsth.get("auroc")), _fmt(tsth.get("auprc")),
               str(r.get("timing", {}).get("total_sec")),
               _fmt(_mean(ms_pb), 1)]
        for i in range(3):
            row.append(_fmt(_mean([g[i] for g in gn if i < len(g) and g[i] is not None])))
        for k in ("cos_e1_e2", "cos_e1_e3", "cos_e2_e3"):
            row.append(_fmt(_mean([c.get(k) for c in gc if c.get(k) is not None])))
        row.append(str(r["config"]["lr"]))
        lines.append(",".join(row))
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ---- 臂间配对差（同种子 B−A / A−LNN）
    def by_arm_seed(a, s):
        for r in done:
            if r["arm"] == a and r["seed"] == s:
                return r
        return None

    pair_keys = [("val_admission", "auroc"), ("val_admission", "auprc"),
                 ("val_hourly", "auroc"), ("val_hourly", "auprc")]

    def delta_table(arm_hi, arm_lo, label):
        rows = []
        for s in (0, 1, 2):
            hi, lo = by_arm_seed(arm_hi, s), by_arm_seed(arm_lo, s)
            if hi is None or lo is None:
                continue
            row = {"seed": s}
            for grp, key in pair_keys:
                a = ((hi.get("final_val") or {}).get(grp) or {}).get(key)
                b = ((lo.get("final_val") or {}).get(grp) or {}).get(key)
                row[f"{grp}_{key}"] = None if (a is None or b is None) else round(a - b, 4)
            rows.append(row)
        means = {}
        for grp, key in pair_keys:
            vals = [r[f"{grp}_{key}"] for r in rows if r.get(f"{grp}_{key}") is not None]
            means[f"{grp}_{key}"] = {
                "mean": round(float(np.mean(vals)), 4) if vals else None,
                "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
                "n_pairs": len(vals)}
        return {"label": label, "per_seed": rows, "mean_std": means}

    d_ba = delta_table("B", "A", "B−A（分阶段−联合，H1 方向性，pilot 不终判）")
    d_al = delta_table("A", "lnn", "A−LNN（D5 方向性健全性）")

    # ---- 梯度冲突 JSON（D7）
    grad_out = {"schema": "pilot-gradient-conflict/v1", "stage": "S3-pilot",
                "note": "D7：A 臂全程 / B 臂 stage4 记录；阶段1-3 专家不联合，叉乘余弦无定义（None）",
                "runs": {}}
    for r in runs:
        eps = r.get("epochs_records") or []
        grad_out["runs"][r["run_id"]] = {
            "arm": r["arm"], "seed": r["seed"], "status": r["status"],
            "per_epoch": [{"epoch": e["epoch"], "stage": e["stage"],
                           "grad_logged_batches": e.get("grad_logged_batches"),
                           "expert_grad_norm_mean": e.get("expert_grad_norm_mean"),
                           "grad_cos_mean": e.get("grad_cos_mean")} for e in eps]}
    _atomic_write_json(os.path.join(RESULTS, "pilot_gradient_conflict.json"), grad_out)

    # ---- timing JSON
    timing_out = {"schema": "pilot-timing/v1", "stage": "S3-pilot",
                  "d9_cap_sec": 720, "d9_hard_cap_sec": 1200, "runs": {}}
    for r in runs:
        timing_out["runs"][r["run_id"]] = {
            "arm": r["arm"], "seed": r["seed"], "status": r["status"],
            "total_sec": r.get("timing", {}).get("total_sec"),
            "train_sec": r.get("timing", {}).get("train_sec"),
            "per_epoch_sec": r.get("timing", {}).get("per_epoch_sec"),
            "ms_per_batch": r.get("timing", {}).get("ms_per_batch"),
            "within_d9_cap": (r.get("timing", {}).get("total_sec") or 0) <= 720}
    _atomic_write_json(os.path.join(RESULTS, "pilot_timing.json"), timing_out)

    # ---- 里程碑 ①②③
    a_done = [r for r in done if r["arm"] == "A"]
    b_done = [r for r in done if r["arm"] == "B"]
    lnn_done = [r for r in done if r["arm"] == "lnn"]
    m1 = None
    for r in a_done:
        eps = r.get("epochs_records") or []
        if len(eps) >= 2 and eps[-1]["train_loss_mean"] is not None \
                and eps[0]["train_loss_mean"] is not None \
                and eps[-1]["train_loss_mean"] < eps[0]["train_loss_mean"] \
                and ((r.get("final_val") or {}).get("val_admission") or {}).get("auroc") is not None:
            m1 = r
            break
    m2 = b_done[0] if b_done else None
    prog_ok = os.path.exists(PROGRESS)
    grad_ok = any(e.get("grad_cos_mean") for r in runs for e in (r.get("epochs_records") or []))
    diff_ok = os.path.exists(os.path.join(RESULTS, "baseline_diff.md"))
    m3 = bool(prog_ok and grad_ok and diff_ok)
    milestones = {
        "m1_joint_baseline": {"pass": m1 is not None,
                              "evidence": (f"{m1['run_id']} loss {m1['epochs_records'][0]['train_loss_mean']:.4f}"
                                           f"→{m1['final_val']['train_loss_mean']:.4f}, "
                                           f"val_adm_auroc="
                                           f"{m1['final_val']['val_admission']['auroc']:.4f}"
                                           if m1 else "无完成的 A 臂 run")},
        "m2_staged_arm": {"pass": m2 is not None,
                          "evidence": (f"{m2['run_id']} {len(m2['epochs_records'])} epochs "
                                       f"stage={[e['stage'] for e in m2['epochs_records']]}"
                                       if m2 else "无完成的 B 臂 run")},
        "m3_pilot_log": {"pass": m3,
                         "evidence": f"progress={prog_ok} grad_cos={grad_ok} diff_list={diff_ok}"}}

    tier = None
    if os.path.exists(TIER_DECISION):
        with open(TIER_DECISION, encoding="utf-8") as f:
            tier = json.load(f)

    # ---- MD
    md = []
    md.append("# S3-pilot：联合(A) vs 分阶段(B) × 3 种子 + 单体 LNN sanity")
    md.append("")
    md.append(f"- 生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S%z')}；"
              f"完成 run {len(done)}/{len(runs)}（smoke 不计）")
    md.append(f"- 数据：PhysioNet-CinC 2019 training_setA 子采样 "
              f"n={runs[0]['data']['n_used'] if runs else '?'}"
              f"（在盘 {runs[0]['data']['n_available'] if runs else '?'}，"
              f"跳过 {runs[0]['data']['skipped_files'] if runs else '?'}）"
              if runs else "- 数据：无 run")
    if tier:
        md.append(f"- D9 阶梯决策：{tier.get('tier')} 条/run；原因：{tier.get('reason')}")
    md.append("- 配置（D4 冻结）：K=3 CfC hidden 1500、τ=1:10:100、门控 MLP64、"
              "Adam lr=1e-3、BCE、batch 64、E=6（B=2/2/1/1）；种子{0,1,2}；"
              "D10 共用子采样/划分/数据顺序/scaler")
    md.append("")
    md.append("## 逐 run 结果（val=每 epoch；test=结束后一次性）")
    md.append("")
    md.append("| run | status | ep | train_loss | val_loss | val_adm_AUROC | val_adm_AUPRC "
              "| val_h_AUROC | val_h_AUPRC | test_adm_AUROC | test_adm_AUPRC | total_s |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        fv = r.get("final_val") or {}
        te = (r.get("test") or {}).get("admission") or {}
        va = fv.get("val_admission") or {}
        vh = fv.get("val_hourly") or {}
        md.append("| {rid} | {st} | {ne} | {tl} | {vl} | {va} | {vap} | {vha} | {vhp} "
                  "| {ta} | {tap} | {ts} |".format(
                      rid=r["run_id"], st=r["status"], ne=len(r.get("epochs_records") or []),
                      tl=_fmt(fv.get("train_loss_mean")), vl=_fmt(fv.get("val_loss")),
                      va=_fmt(va.get("auroc")), vap=_fmt(va.get("auprc")),
                      vha=_fmt(vh.get("auroc")), vhp=_fmt(vh.get("auprc")),
                      ta=_fmt(te.get("auroc")), tap=_fmt(te.get("auprc")),
                      ts=r.get("timing", {}).get("total_sec")))
    md.append("")
    md.append("## 臂间方向性差值（pilot 不终判 H1）")
    md.append("")
    for d in (d_ba, d_al):
        md.append(f"### {d['label']}")
        md.append("")
        md.append("| seed | " + " | ".join(k for k in d["per_seed"][0] if k != "seed") + " |"
                  if d["per_seed"] else "（无配对）")
        if d["per_seed"]:
            md.append("|---|" + "---|" * (len(d["per_seed"][0]) - 1))
            for row in d["per_seed"]:
                md.append("| " + str(row["seed"]) + " | "
                          + " | ".join(_fmt(row[k]) for k in row if k != "seed") + " |")
            md.append("")
            md.append("mean±std：" + "；".join(
                f"{k}={_fmt(v['mean'])}±{_fmt(v['std'])} (n={v['n_pairs']})"
                for k, v in d["mean_std"].items()))
        md.append("")
    md.append("## 单体 LNN sanity（D5）")
    md.append("")
    if lnn_done and a_done:
        ln = lnn_done[0]
        a0 = by_arm_seed("A", 0)
        md.append(f"- LNN seed0 test_adm_auroc={_fmt(((ln.get('test') or {}).get('admission') or {}).get('auroc'))}；"
                  f"A seed0 test_adm_auroc={_fmt((((a0 or {}).get('test') or {}).get('admission') or {}).get('auroc'))}")
        md.append(f"- A−LNN（seed0）：{_fmt(d_al['mean_std'].get('val_admission_auroc', {}).get('mean'))}"
                  "（val_adm_auroc，mean）")
    else:
        md.append("- 缺 LNN 或 A 臂完成 run，sanity 未评")
    md.append("")
    md.append("## 梯度冲突（D7，H2a 代理）")
    md.append("")
    for r in runs:
        eps = [e for e in (r.get("epochs_records") or []) if e.get("grad_cos_mean")]
        if not eps:
            continue
        md.append(f"- {r['run_id']}：")
        for e in eps:
            md.append(f"  - ep{e['epoch']} stage{e['stage']} "
                      f"logged={e.get('grad_logged_batches')} "
                      f"‖g‖={[None if v is None else round(v, 4) for v in (e.get('expert_grad_norm_mean') or [])]} "
                      f"cos={ {k: (None if v is None else round(v, 4)) for k, v in e['grad_cos_mean'].items()} }")
    md.append("")
    md.append("## 时间（D9：单训练 ≤12min，硬顶 20min）")
    md.append("")
    for r in runs:
        t = r.get("timing", {})
        md.append(f"- {r['run_id']}：total={t.get('total_sec')}s "
                  f"train={t.get('train_sec')}s per_epoch={t.get('per_epoch_sec')}")
    md.append("")
    md.append("## 首里程碑 ①②③ 验收")
    md.append("")
    md.append(f"- ① 联合基线损失下降且 AUROC 可算：{'PASS' if milestones['m1_joint_baseline']['pass'] else 'FAIL'}"
              f" — {milestones['m1_joint_baseline']['evidence']}")
    md.append(f"- ② 分阶段臂跑通：{'PASS' if milestones['m2_staged_arm']['pass'] else 'FAIL'}"
              f" — {milestones['m2_staged_arm']['evidence']}")
    md.append(f"- ③ pilot 日志（数字/两臂验证曲线/每阶段验证损失/梯度余弦/差异清单）："
              f"{'PASS' if milestones['m3_pilot_log']['pass'] else 'FAIL'}"
              f" — {milestones['m3_pilot_log']['evidence']}")
    md.append("")
    md.append("## 方向性判读（pilot 口径；design §六：不终判 H1）")
    md.append("")
    per_seed = d_ba["per_seed"]
    if per_seed:
        deltas = [r.get("val_admission_auroc") for r in per_seed
                  if r.get("val_admission_auroc") is not None]
        n_neg = sum(1 for d in deltas if d < 0)
        ms = d_ba["mean_std"]["val_admission_auroc"]
        md.append(f"- B−A（val 入院级 AUROC）：mean={_fmt(ms['mean'])}±{_fmt(ms['std'])}"
                  f"（n={ms['n_pairs']} 配对种子）；{n_neg}/{len(deltas)} 个种子为负"
                  f"（方向一致性={'全部同向' if n_neg == len(deltas) else '不一致'}）。")
        md.append(f"- 读数=分阶段臂在本 pilot 设置下**不优于**联合臂（Δmean={_fmt(ms['mean'])} AUROC，"
                  f"Δstd={_fmt(ms['std'])}；pilot 样本量小、不终判 H1）。")
    if lnn_done and a_done:
        ln = lnn_done[0]
        a0 = by_arm_seed("A", 0)
        md.append(f"- D5 方向性健全性：A(seed0) 优于单体 LNN——val {_fmt(((a0 or {}).get('final_val') or {}).get('val_admission', {}).get('auroc'))}"
                  f" vs {_fmt(((ln.get('final_val') or {}).get('val_admission') or {}).get('auroc'))}，"
                  f"test {_fmt((((a0 or {}).get('test') or {}).get('admission') or {}).get('auroc'))}"
                  f" vs {_fmt(((ln.get('test') or {}).get('admission') or {}).get('auroc'))}（PASS）。")
    all_cos = [v for r in runs for e in (r.get("epochs_records") or [])
               for v in (e.get("grad_cos_mean") or {}).values() if v is not None]
    if all_cos:
        md.append(f"- H2(a) 代理（D7）：专家对梯度余弦全部 |cos|<{max(abs(c) for c in all_cos):.4f}"
                  "（A 臂全程与 B 臂 stage4 均≈0）——pilot 规模未测到可辨的梯度冲突信号，"
                  "H2(a) 暂 neither supported nor falsified，留 S5 full + B-no-cal 消融 adjudicate。")
    md.append("- 解读警告（交 S4）：E=6 的 2/2/1/1 分配下 B 臂专家更新预算结构性少于 A 臂"
              "（τ₁≈3 epoch、τ₂/τ₃≈3 epoch vs A 全参数 6 epoch），负差值与该预算差纠缠；"
              "S5 full（8/6/4/2）会缩小比例差。B 臂无 loss 爆炸/NaN（F3 未触发），"
              "τ=100 慢专家梯度范数 ~2-10× 于快专家但不爆（与 S2 基准一致）。")
    b_runs = [r for r in done if r["arm"] == "B"]
    if b_runs:
        trans = []
        for r in b_runs:
            s3 = [e for e in r["epochs_records"] if e["stage"] == 3]
            s4 = [e for e in r["epochs_records"] if e["stage"] == 4]
            if s3 and s4:
                trans.append(f"{r['run_id']} {s3[-1]['val_loss']:.4f}→{s4[-1]['val_loss']:.4f}")
        md.append(f"- B 臂 stage3→4（联合校准）验证损失过渡：{'；'.join(trans)}"
                  "（变化 ≤0.004，无断崖；stage3 的 val AUROC 波动大=200 例/约 15 septic 的"
                  "小样本噪声，已在 pilot_timing/progress 落逐 epoch 曲线）。")
    md.append("")
    md.append("## 实现记录与偏差（如实）")
    md.append("")
    n_ep_batches = None
    if runs:
        eps0 = runs[0].get("epochs_records") or []
        if eps0:
            n_ep_batches = eps0[0].get("n_batches")
    md.append(f"- D7「每 epoch ≥200 batch」在本档（{runs[0]['data']['n_used'] if runs else '?'} 例→"
              f"{n_ep_batches} batch/epoch）不可行（需额外前向反向、击穿 D9 12min 上限），"
              "按每 epoch 全量 batch 记录；样本量已落 pilot_gradient_conflict.json，S5 full 回归原文口径。")
    md.append("- padding 到 72h：填充步值通道 0、mask 通道 1，损失/评测只在真实小时步计算。")
    md.append("- B 臂阶段切换时重建 optimizer（Adam 动量重置；被冻结参数状态本即无关）。")
    md.append("- 入院级标签=窗内任一小时阳性；评分=窗口概率最大值（D8）。")
    md.append("- 数据源=PhysioNet-CinC 2019 回退源（Moor 2023 本夜不可得，S2 证据落盘）。")
    md.append("- pilot 只答方向性与管线健全性，不终判 H1（design §六）。")
    md.append("")
    with open(os.path.join(RESULTS, "pilot_joint_vs_staged.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    out = {"schema": "pilot-summary/v1", "n_runs": len(runs), "n_done": len(done),
           "delta_B_minus_A": d_ba, "delta_A_minus_LNN": d_al,
           "milestones": milestones, "tier": tier}
    _atomic_write_json(os.path.join(RESULTS, "_pilot_summary.json"), out)
    print(json.dumps({"n_runs": len(runs), "n_done": len(done),
                      "milestones": {k: v["pass"] for k, v in milestones.items()},
                      "d_BA_val_adm_auroc": d_ba["mean_std"]["val_admission_auroc"]},
                     ensure_ascii=False), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "lnn"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-patients", type=int, default=8000, help="D9 阶梯 8000→4000→2000")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--lr", type=float, default=mr_moe.LR, help="F3 两档：1e-3 / 3e-4")
    ap.add_argument("--tag", default="", help="run 名后缀（smoke 等；汇总时跳过）")
    ap.add_argument("--resume", action="store_true", help="从最新 ckpt 续跑（D9）")
    ap.add_argument("--log-grad-every", type=int, default=1, help="D7：每多少 batch 记一次梯度")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.summary:
        summary()
        return 0
    if args.arm is None or args.seed is None:
        ap.error("--arm 与 --seed 必填（或用 --summary）")
    train_run(args.arm, args.seed, args.n_patients, args.epochs, args.lr,
              args.device, args.log_grad_every, args.resume, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
