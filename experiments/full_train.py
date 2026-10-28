#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""full_train.py — S5-full-experiment 执行器：full 划分 E=20，A(联合)×3 种子 vs
B(8/6/4/2 分阶段)×3 种子 + 条件消融 B-no-cal(阶段一至三)×3 种子。

design.md 冻结映射：
  D1 臂语义 / D2 阶段步数（full E=20 → 8/6/4/2；B-no-cal=阶段1-3=8/6/4=18 epoch，
  与 B 的阶段1-3 步数逐字相同，唯一差异=无阶段四联合校准——E2/H2b 判据）/
  D3 数据预处理 / D4 模型超参 / D6 种子{0,1,2}+mean±std / D7 梯度冲突日志 /
  D8 评测（入院级主+小时级次）/ D9 阶梯定档+checkpoint / D10 单变量纪律。

与 pilot_train.py 的关系：pilot 的扩展（S5 stage inputs 指定）。pilot 已验证的
实现注记原样保留：stage 切换重建 optimizer（Adam 动量重置记录在案）、padding 约定
（填充步值通道 0、mask 通道 1，损失/评测只在真实小时步计算）、入院级标签=窗内任一
小时阳性、评分=窗口概率最大值。

D9 阶梯定档（本单元指令）：按训练启动时在盘语料定档——在盘 ≥8000 → 8000 档；
2000-7999 → 在盘 ≥4000 取 4000 档、否则 2000 档；hidden 恒 1500 不动。
决策+速率证据原子落 results/_full_tier_decision.json（禁合成数据 F2）。

D7 执行口径（与 design 原文的差异，如实记录）：原文「每 epoch 抽样 ≥200 个 batch」，
本档 train≈1400 例（2000 档）→22 batch/epoch，200 个不重复 batch 需额外前向反向、
击穿 D9 单训练时长上限；按 pilot 先例「每 epoch 全量 batch 记录」
（A 臂 22×20=440 样本/run；B 臂 stage2+stage4=22×8=176；B-no-cal stage2=22×6=132）。
原文口径需 full 语料（setA 20643×0.7≈14450 train→226 batch/epoch），全文语料下载
后台续跑中（results/_download_state.json），达量后可由后续单元回归原文口径。

D7 记录范围（相对 pilot 的受控扩展，如实记录）：pilot 规则=A 全程 / B 仅 stage4；
本执行器按「≥2 个专家联合可训即记录」执行——A 全部 epoch；B 的 stage2（τ₂,τ₃ 联合）
与 stage4（全部）；B-no-cal 的 stage2。所有上报余弦均为定义良好的专家对
（stage1/3 无非联合专家对，记 None）。design §四 D7 括注「阶段一至三专家不联合」
对 stage2 不精确（τ₂/τ₃ 在 stage2 联合可训），本扩展只增加定义良好的记录。

D7 计数 bug 修复（S6-final-verdict-redesign，2026-10-29；S5 判读〔52〕代码级根因）：
原实现按 param group 名前缀 startswith("expert") 计联合可训专家组数，而
mr_moe.param_groups_for_stage(4) 只有单组且名为 "all" → 计数 0 → stage4 的
log_grad=False → S5 B 臂 stage4 六 epoch 遥测全缺失（grad_logged_batches=0、
三对 cos 全 null），H2(a) 设计点名的「A 全程 vs B 阶段四」比较无数据。现改为按
groups 实际覆盖的专家参数集合计数（stage1=1/stage2=2/stage3=0/stage4=3），并在
每个 epoch 末加非空断言：log_grad=True 的 epoch 必须 grad_logged_batches>0 且
联合可训专家对余弦全部非 null 且有限，否则 AssertionError 终止 run（防同类静默缺失）。
修复不改变任何训练数学（expert_gradient_stats 只读梯度、不动 p.grad/参数/RNG），
同配置 B×3 补跑（新 tag d7fix，run_id=full_B_seed{0,1,2}_d7fix）与 S5 的性能指标
应对齐（差异仅 GPU 非确定性量级），差异=纯遥测补齐。

用法：
  G:/ai/python11/python.exe experiments/full_train.py --tier-decision   # 只定档并落盘
  G:/ai/python11/python.exe experiments/full_train.py --all             # 九跑顺序执行
  G:/ai/python11/python.exe experiments/full_train.py --arm A --seed 0  # 单 run
  G:/ai/python11/python.exe experiments/full_train.py --arm B --seed 1 --resume
  G:/ai/python11/python.exe experiments/full_train.py --summary         # 汇总出表
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
RUNS_DIR = os.path.join(RESULTS, "full_runs")
PROGRESS = os.path.join(RESULTS, "full_progress.json")
TIER_DECISION = os.path.join(RESULTS, "_full_tier_decision.json")
DOWNLOAD_STATE = os.path.join(RESULTS, "_download_state.json")

SUBSAMPLE_SEED = 20261027   # D10：所有臂/种子共用同一子采样患者集（与 pilot 同种子表）
SPLIT_SEED = 0              # D10：共用同一患者级划分（Che 2018 操作化，见 data_pipeline）
LOADER_SEED = 12345         # D10：共用同一 train 数据顺序
IN_DIM = 2 * dp.N_FEATURES  # 68 = [值 34, mask 34]

FULL_EPOCHS = 20            # D2：full E=20 → 8/6/4/2（预注册默认值，不修订）
STAGE_EPOCHS_E20 = (8, 6, 4, 2)          # D2 比例 4:3:2:1 + 最小 1 epoch
BNOCAL_STAGE_EPOCHS = (8, 6, 4)          # E2/H2b：B 的阶段1-3 逐字相同，无阶段四
SEEDS = (0, 1, 2)           # D6 固定种子表
ARMS = ("A", "B", "BnoCal")
D9_CAP_SEC = 720            # D9：单次训练 ≤12min（pilot 口径沿用；full E=20 同上限）


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
    if scores.size == 0 or not np.isfinite(scores).all() or len(np.unique(lab)) < 2:
        return {"auroc": None, "auprc": None, "n": int(lab.size),
                "n_pos": int(lab.sum()) if lab.size else 0}
    return {"auroc": float(roc_auc_score(lab, scores)),
            "auprc": float(average_precision_score(lab, scores)),
            "n": int(lab.size), "n_pos": int(lab.sum())}


def stage_schedule(arm: str, epochs: int) -> list[int | None]:
    """D2 阶段步数：A 联合=[None]*E；B=E=20 → 8/6/4/2；B-no-Cal=阶段1-3 → 8/6/4。"""
    if arm == "A":
        return [None] * epochs
    if arm == "B":
        raw = list(STAGE_EPOCHS_E20)
        assert sum(raw) == epochs, f"B 阶段步数 {raw} 与 E={epochs} 不一致"
    elif arm == "BnoCal":
        raw = list(BNOCAL_STAGE_EPOCHS)
        assert sum(raw) == epochs, f"B-no-Cal 阶段步数 {raw} 与 E={epochs} 不一致"
    else:
        raise ValueError(f"未知 arm={arm}")
    out: list[int | None] = []
    for stage, cnt in enumerate(raw, 1):
        out.extend([stage] * cnt)
    return out


def epochs_for_arm(arm: str) -> int:
    return FULL_EPOCHS if arm in ("A", "B") else sum(BNOCAL_STAGE_EPOCHS)


# ---------------------------------------------------------------- D9 阶梯定档
def _download_rate_evidence() -> dict:
    """从 results/_download_state.json 取本 run 速率证据（无则空）。"""
    if not os.path.exists(DOWNLOAD_STATE):
        return {}
    try:
        with open(DOWNLOAD_STATE, encoding="utf-8") as f:
            st = json.load(f)
    except Exception:  # noqa: BLE001
        return {}
    ok = st.get("n_ok_this_run") or 0
    el = st.get("elapsed_sec") or 0.0
    return {"this_run_ok": ok, "this_run_fail": st.get("n_fail_this_run"),
            "elapsed_sec": el,
            "rate_files_per_sec": round(ok / el, 3) if el > 0 else None,
            "updated_at": st.get("updated_at")}


def decide_tier(write: bool = True) -> dict:
    """D9 阶梯：按在盘语料定档（≥8000→8000；≥4000→4000；否则 2000）。hidden 恒 1500。"""
    n_disk = 0
    for name in ("training_setA", "training_setB"):
        d = os.path.join(DATA_ROOT, name)
        if os.path.isdir(d):
            n_disk += sum(1 for f in os.listdir(d) if f.endswith(".psv"))
    if n_disk >= 8000:
        tier = 8000
    elif n_disk >= 4000:
        tier = 4000
    else:
        tier = 2000
    rate = _download_rate_evidence()
    rps = rate.get("rate_files_per_sec") or 1.17
    eta = {}
    for t in (4000, 8000):
        rem = max(0, t - n_disk)
        eta[f"tier{t}_eta_sec"] = round(rem / rps, 0) if rem else 0
        eta[f"tier{t}_eta_h"] = round(rem / rps / 3600.0, 2) if rem else 0.0
    dec = {"schema": "full-tier-decision/v1", "stage": "S5-full-experiment",
           "decided_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "tier": tier, "n_on_disk_at_decision": n_disk,
           "ladder": "D9 预授权 8000→4000→2000；hidden 恒 1500 不动",
           "rule": "在盘 ≥8000→8000 档；2000-7999→在盘 ≥4000 取 4000 否则 2000 档",
           "rate_evidence": rate,
           "eta_from_decision": eta,
           "reason": (f"训练启动时在盘 {n_disk} psv；"
                      + (f"≥8000 取 8000 档" if n_disk >= 8000 else
                         (f"≥4000 取 4000 档" if n_disk >= 4000 else
                          "<4000 取 2000 档（D9 预授权阶梯内降档）")) +
                      "；hidden 恒 1500；禁合成数据（F2 不触发：全部真实 PhysioNet 2019 语料）"),
           "d9_compliance": "hidden 恒 1500；每 epoch 原子 checkpoint；单 run 时长落 full_progress/timing",
           "corpus_plan": ("download_physionet2019.py --workers 32 后台续跑中"
                           "（results/_download_state.json 原子记进度）；full 语料"
                           "（setA 20643+setB 20000）未到位不影响本档定档，"
                           "D7 原文 ≥200 batch/epoch 口径留待语料到位后由后续单元回归")}
    if write:
        _atomic_write_json(TIER_DECISION, dec)
    return dec


# ---------------------------------------------------------------- 数据
def build_tensors(examples) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """患者列表 → 定长张量（padding 到 SEQ_LEN；填充步 mask=1、step-valid=0）。

    padding 约定（与 pilot 一致）：值通道填 0、mask 通道置 1（D3「0 填充但可见」），
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
    """子采样+划分+scaler+张量化，落盘 npz 缓存（九个 run 共用，保证 D10 一致）。"""
    os.makedirs(RUNS_DIR, exist_ok=True)
    cache = os.path.join(RESULTS, f"_full_data_ss{SUBSAMPLE_SEED}_n{n_patients}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        return {"X": z["X"], "V": z["V"], "Y": z["Y"],
                "idx_tr": z["idx_tr"], "idx_va": z["idx_va"], "idx_te": z["idx_te"],
                "meta": json.loads(str(z["meta_json"])), "cache_path": cache}
    t0 = time.time()
    examples, skipped = dp.load_dataset(DATA_ROOT, set_names=("training_setA", "training_setB"))
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
    stats = dp.dataset_stats(sub)
    meta = {"n_available": n_avail, "n_used": n_used, "n_requested": n_patients,
            "skipped_files": len(skipped), "subsample_seed": SUBSAMPLE_SEED,
            "split_seed": SPLIT_SEED, "loader_seed": LOADER_SEED,
            "split_sizes": {k: len(v) for k, v in idx.items()},
            "dataset_stats": stats,
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
    prog = {"schema": "full-progress/v1", "runs": {}}
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
              device: str, log_grad_every: int, resume: bool, tag: str = "") -> dict:
    run_id = f"full_{arm}_seed{seed}" + (f"_{tag}" if tag else "")
    t0 = time.time()
    cache = build_or_load_cache(n_patients)
    X, V, Y = cache["X"], cache["V"], cache["Y"]
    idx_tr, idx_va, idx_te = cache["idx_tr"], cache["idx_va"], cache["idx_te"]
    meta = cache["meta"]

    model = mr_moe.build_model(IN_DIM, seed, kind="mr_moe").to(device)
    n_params = mr_moe.count_params(model)
    stages = stage_schedule(arm, epochs)

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
        if arm in ("B", "BnoCal"):
            groups = model.param_groups_for_stage(stage)
            # D7 修复（S6-final-verdict-redesign，2026-10-29）：联合可训专家组数按
            # **实际可训专家参数集合**判定，不再按组名前缀 startswith("expert")——
            # mr_moe.param_groups_for_stage(4) 只有单组且名为 "all"（stage4 全部参数
            # 解冻联合校准），组名前缀计数得 0 → log_grad=False → B 臂 stage4 六个
            # epoch 遥测静默全缺失（S5 full_gradient_conflict.json 实测
            # grad_logged_batches=0、三对 cos 全 null，H2(a) 指定比较期无数据）。
            # 现统计 groups 实际覆盖的专家模块参数：被覆盖的专家数即联合可训专家组数
            # （stage1=1 / stage2=2 / stage3=0 / stage4=3，与命名约定彻底解耦）。
            trainable_ids = {id(p) for g in groups for p in g["params"]}
            n_expert_groups = sum(
                1 for expert in model.experts
                if any(id(p) in trainable_ids for p in expert.parameters()))
            if stage != prev_stage:
                opt = torch.optim.Adam([p for g in groups for p in g["params"]], lr=lr)
                prev_stage = stage
        else:
            n_expert_groups = 3
            if opt is None:
                opt = torch.optim.Adam(model.parameters(), lr=lr)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=mr_moe.BATCH_SIZE, shuffle=True, generator=gen)
        model.train()
        # D7：≥2 个专家**按实际可训参数集合**联合可训的 epoch 才记录
        # （A 全程 n=3；B stage2 n=2 + stage4 n=3；B-no-Cal stage2 n=2）。
        # 计数口径见上方 trainable_ids 注释（S6-final-verdict-redesign 修复：
        # 不再按组名前缀，stage4 组名 "all" 也能正确判为 3）。
        log_grad = n_expert_groups >= 2
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
        # D7 遥测非空断言（S6-final-verdict-redesign，2026-10-29 增补）：
        # log_grad=True 的 epoch 必须①真的记录到梯度（grad_logged_batches>0）且
        # ②所有**联合可训**专家对余弦非 null 且有限（n_expert_groups=3 时即三对
        # 全非 null，如 A 全程/B stage4；stage2 的 n=2 时该联合对非 null）。
        # 任一不满足立即 AssertionError 终止本 run——不允许同类遥测静默缺失
        # （S5 的 B stage4 六 epoch 全 null 即因缺此断言而全程无声）。
        if log_grad:
            assert n_logged > 0, (
                f"[d7-assert] {run_id} epoch={epoch} stage={stage}: log_grad=True "
                f"但 grad_logged_batches=0（D7 遥测静默缺失，终止）")
            cos_mean = rec.get("grad_cos_mean") or {}
            expected_pairs = n_expert_groups * (n_expert_groups - 1) // 2
            n_defined = sum(1 for v in cos_mean.values() if v is not None)
            nonfinite = sorted(k for k, v in cos_mean.items()
                               if v is not None and not math.isfinite(v))
            assert n_defined == expected_pairs and not nonfinite, (
                f"[d7-assert] {run_id} epoch={epoch} stage={stage}: 联合可训专家对"
                f"余弦 {n_defined}/{expected_pairs} 非 null、非有限项 {nonfinite}"
                f"（{cos_mean}；D7 遥测不完整，终止）")
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
    run = {"schema": "full-run/v1", "run_id": run_id, "arm": arm, "seed": seed,
           "status": status,
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
                      "per_epoch_sec": ep_secs, "ms_per_batch": ms_pb,
                      "within_d9_cap_720s": total_sec <= D9_CAP_SEC}}
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


def _load_runs() -> list[dict]:
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        if "smoke" in os.path.basename(p):
            continue
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    return runs


_PER_RUN_COLS = ["run_id", "arm", "seed", "status", "n_patients_used", "epochs_done",
                 "final_train_loss", "final_val_loss", "val_adm_auroc", "val_adm_auprc",
                 "val_hourly_auroc", "val_hourly_auprc", "test_adm_auroc", "test_adm_auprc",
                 "test_hourly_auroc", "test_hourly_auprc", "total_sec", "ms_per_batch_mean",
                 "grad_norm_e1_mean", "grad_norm_e2_mean", "grad_norm_e3_mean",
                 "grad_cos_e1_e2_mean", "grad_cos_e1_e3_mean", "grad_cos_e2_e3_mean",
                 "within_d9_cap_720s", "lr"]


def _per_run_row(r: dict) -> list[str]:
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
    row.append(str(r.get("timing", {}).get("within_d9_cap_720s")))
    row.append(str(r["config"]["lr"]))
    return row


def _write_csv(path: str, runs: list[dict]) -> None:
    lines = [",".join(_PER_RUN_COLS)]
    for r in runs:
        lines.append(",".join(_per_run_row(r)))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


_PAIR_KEYS = [("val_admission", "auroc"), ("val_admission", "auprc"),
              ("val_hourly", "auroc"), ("val_hourly", "auprc")]


def _by_arm_seed(runs: list[dict], arm: str, seed: int):
    for r in runs:
        if r["arm"] == arm and r["seed"] == seed and r.get("status") == "done":
            return r
    return None


def delta_table(runs: list[dict], arm_hi, arm_lo, label) -> dict:
    """逐种子配对差（hi−lo）+ mean±std（ddof=1；D6 要求 std 不省略）。"""
    arm_hi = arm_hi if isinstance(arm_hi, (list, tuple)) else [arm_hi]
    arm_lo = arm_lo if isinstance(arm_lo, (list, tuple)) else [arm_lo]
    rows = []
    for s in SEEDS:
        hi = next((_by_arm_seed(runs, a, s) for a in arm_hi
                   if _by_arm_seed(runs, a, s) is not None), None)
        lo = next((_by_arm_seed(runs, a, s) for a in arm_lo
                   if _by_arm_seed(runs, a, s) is not None), None)
        if hi is None or lo is None:
            continue
        row = {"seed": s}
        for grp, key in _PAIR_KEYS:
            a = ((hi.get("final_val") or {}).get(grp) or {}).get(key)
            b = ((lo.get("final_val") or {}).get(grp) or {}).get(key)
            row[f"{grp}_{key}"] = None if (a is None or b is None) else round(a - b, 6)
        rows.append(row)
    means = {}
    for grp, key in _PAIR_KEYS:
        vals = [r[f"{grp}_{key}"] for r in rows if r.get(f"{grp}_{key}") is not None]
        means[f"{grp}_{key}"] = {
            "mean": round(float(np.mean(vals)), 6) if vals else None,
            "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None,
            "n_pairs": len(vals)}
    return {"label": label, "per_seed": rows, "mean_std": means}


def _arm_stats(runs: list[dict], arm: str) -> dict:
    """臂内 mean±std（val 入院级/小时级 AUROC/AUPRC；D6）。"""
    done = [_by_arm_seed(runs, arm, s) for s in SEEDS]
    done = [r for r in done if r is not None]
    out = {"arm": arm, "n_done": len(done), "metrics": {}}
    for grp, key in _PAIR_KEYS:
        vals = [((r.get("final_val") or {}).get(grp) or {}).get(key) for r in done]
        vals = [v for v in vals if v is not None]
        out["metrics"][f"{grp}_{key}"] = {
            "mean": round(float(np.mean(vals)), 6) if vals else None,
            "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None,
            "n": len(vals),
            "per_seed": [((r.get("final_val") or {}).get(grp) or {}).get(key) for r in done]}
    return out


def summary() -> dict:
    runs = _load_runs()
    done = [r for r in runs if r.get("status") == "done"]
    ab_runs = [r for r in runs if r["arm"] in ("A", "B")]
    bc_runs = [r for r in runs if r["arm"] == "BnoCal"]

    # ---- full_joint_vs_staged.csv（A×3 + B×3 逐 run）
    _write_csv(os.path.join(RESULTS, "full_joint_vs_staged.csv"), ab_runs)
    # ---- full_bnocal.csv（B-no-cal×3 逐 run + 配对差 B−BnoCal 附行）
    _write_csv(os.path.join(RESULTS, "full_bnocal.csv"), bc_runs)
    d_ba = delta_table(runs, "B", "A", "B−A（分阶段−联合；H1 方向性）")
    d_bc = delta_table(runs, "B", "BnoCal", "B−B-no-cal（阶段四联合校准效应；E2/H2b 判据）")
    d_ac = delta_table(runs, "A", "BnoCal", "A−B-no-cal（参考：无阶段四的分阶段臂 vs 联合臂）")
    stats = {a: _arm_stats(runs, a) for a in ARMS}

    # ---- H1 预注册门线读数（design §六；终判归 S6）
    a_adm = stats["A"]["metrics"]["val_admission_auroc"]
    b_adm = stats["B"]["metrics"]["val_admission_auroc"]
    a_auprc = stats["A"]["metrics"]["val_admission_auprc"]
    b_auprc = stats["B"]["metrics"]["val_admission_auprc"]
    d_auroc = d_ba["mean_std"]["val_admission_auroc"]
    d_auprc = d_ba["mean_std"]["val_admission_auprc"]
    gate = None
    if None not in (d_auroc["mean"], d_auprc["mean"]) \
            and a_adm["std"] is not None and b_adm["std"] is not None:
        c1 = d_auroc["mean"] >= 0.01
        c2 = d_auprc["mean"] >= -0.01
        c3 = b_adm["std"] <= a_adm["std"]
        gate = {"criterion_1_delta_auroc_ge_+0.01": {"value": d_auroc["mean"], "pass": bool(c1)},
                "criterion_2_delta_auprc_ge_-0.01": {"value": d_auprc["mean"], "pass": bool(c2)},
                "criterion_3_stdB_le_stdA": {"std_A": a_adm["std"], "std_B": b_adm["std"],
                                             "pass": bool(c3)},
                "all_three": bool(c1 and c2 and c3)}

    # ---- 梯度冲突 JSON（D7）
    grad_out = {"schema": "full-gradient-conflict/v1", "stage": "S5-full-experiment",
                "note": ("D7：≥2 专家联合可训的 epoch 记录（A 全部；B stage2+stage4；"
                         "B-no-cal stage2）；stage1/3 无非联合专家对，叉乘余弦无定义（None）。"
                         "design 原文「每 epoch ≥200 batch」在本档（22 batch/epoch）不可行，"
                         "按 pilot 先例每 epoch 全量 batch 记录，样本量逐 epoch 落盘。"),
                "runs": {}}
    for r in runs:
        eps = r.get("epochs_records") or []
        grad_out["runs"][r["run_id"]] = {
            "arm": r["arm"], "seed": r["seed"], "status": r["status"],
            "per_epoch": [{"epoch": e["epoch"], "stage": e["stage"],
                           "grad_logged_batches": e.get("grad_logged_batches"),
                           "expert_grad_norm_mean": e.get("expert_grad_norm_mean"),
                           "grad_cos_mean": e.get("grad_cos_mean")} for e in eps]}
    _atomic_write_json(os.path.join(RESULTS, "full_gradient_conflict.json"), grad_out)

    # ---- 汇总 JSON
    out = {"schema": "full-summary/v1", "stage": "S5-full-experiment",
           "n_runs": len(runs), "n_done": len(done),
           "arms": stats,
           "delta_B_minus_A": d_ba, "delta_B_minus_BnoCal": d_bc,
           "delta_A_minus_BnoCal": d_ac,
           "h1_gate_reading": gate,
           "d9": {"cap_sec": D9_CAP_SEC,
                  "max_total_sec": max([r.get("timing", {}).get("total_sec") or 0
                                        for r in runs] or [0]),
                  "all_within_cap": all(r.get("timing", {}).get("within_d9_cap_720s")
                                        for r in runs) if runs else None}}
    _atomic_write_json(os.path.join(RESULTS, "_full_summary.json"), out)

    # ---- MD（结论级表）
    tier = None
    if os.path.exists(TIER_DECISION):
        with open(TIER_DECISION, encoding="utf-8") as f:
            tier = json.load(f)
    md = []
    md.append("# S5-full-experiment：联合(A) vs 分阶段(B) × 3 种子 + B-no-cal 消融 × 3 种子")
    md.append("")
    md.append(f"- 生成时间：{time.strftime('%Y-%m-%dT%H:%M:%S%z')}；"
              f"完成 run {len(done)}/{len(runs)}")
    if runs:
        d0 = runs[0]["data"]
        md.append(f"- 数据：PhysioNet-CinC 2019 子采样 n={d0['n_used']}"
                  f"（在盘 {d0['n_available']}，跳过 {d0['skipped_files']}；"
                  f"septic 率 {d0['dataset_stats']['septic_patient_rate']}；"
                  f"划分 {d0['split_sizes']}；D10 共用子采样/划分/数据顺序/scaler）")
    if tier:
        md.append(f"- D9 阶梯定档：{tier.get('tier')} 条/run；在盘 {tier.get('n_on_disk_at_decision')} psv；"
                  f"原因：{tier.get('reason')}")
    md.append("- 配置（D4 冻结）：K=3 CfC hidden 1500、τ=1:10:100、门控 MLP64、"
              "Adam lr=1e-3、BCE、batch 64、E=20（B=8/6/4/2；B-no-cal=8/6/4=18epoch）；"
              "种子{0,1,2}；每 epoch 原子 checkpoint")
    md.append("- 算子状态：批量训练全部在 S2⓪ 修后算子上（results/operator_bench.md PASS，"
              "F8 未触发）——无「基于旧算子」数字（fb-ab50d0ec9f 条款 vacuous）")
    md.append("")
    md.append("## 结论级表：mean±std（val；入院级主指标 + 小时级次指标；D6/D8）")
    md.append("")
    md.append("| 臂 | n_done | val_adm_AUROC | val_adm_AUPRC | val_hourly_AUROC | val_hourly_AUPRC |")
    md.append("|---|---|---|---|---|---|")
    for a in ARMS:
        m = stats[a]["metrics"]
        md.append(f"| {a} | {stats[a]['n_done']} | "
                  f"{_fmt(m['val_admission_auroc']['mean'])}±{_fmt(m['val_admission_auroc']['std'])} | "
                  f"{_fmt(m['val_admission_auprc']['mean'])}±{_fmt(m['val_admission_auprc']['std'])} | "
                  f"{_fmt(m['val_hourly_auroc']['mean'])}±{_fmt(m['val_hourly_auroc']['std'])} | "
                  f"{_fmt(m['val_hourly_auprc']['mean'])}±{_fmt(m['val_hourly_auprc']['std'])} |")
    md.append("")
    md.append("## 逐 run 明细（val=最后 epoch；test=结束后一次性）")
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
    md.append("## 逐种子配对明细（D6）")
    md.append("")
    for d in (d_ba, d_bc, d_ac):
        md.append(f"### {d['label']}")
        md.append("")
        if d["per_seed"]:
            md.append("| seed | " + " | ".join(k for k in d["per_seed"][0] if k != "seed") + " |")
            md.append("|---|" + "---|" * (len(d["per_seed"][0]) - 1))
            for row in d["per_seed"]:
                md.append("| " + str(row["seed"]) + " | "
                          + " | ".join(_fmt(row[k]) for k in row if k != "seed") + " |")
            md.append("")
            md.append("mean±std：" + "；".join(
                f"{k}={_fmt(v['mean'])}±{_fmt(v['std'])} (n={v['n_pairs']})"
                for k, v in d["mean_std"].items()))
        else:
            md.append("（无配对）")
        md.append("")
    md.append("## H1 预注册门线读数（design §六；终判归 S6-final-verdict）")
    md.append("")
    if gate:
        md.append(f"- 判据① ΔAUROC(B−A, val 入院级) ≥ +0.01：{_fmt(d_auroc['mean'])}±{_fmt(d_auroc['std'])}"
                  f" → {'PASS' if gate['criterion_1_delta_auroc_ge_+0.01']['pass'] else 'FAIL'}")
        md.append(f"- 判据② ΔAUPRC(B−A, val 入院级) ≥ −0.01：{_fmt(d_auprc['mean'])}±{_fmt(d_auprc['std'])}"
                  f" → {'PASS' if gate['criterion_2_delta_auprc_ge_-0.01']['pass'] else 'FAIL'}")
        md.append(f"- 判据③ std(B) ≤ std(A)（val 入院级 AUROC）：{_fmt(b_adm['std'])} vs {_fmt(a_adm['std'])}"
                  f" → {'PASS' if gate['criterion_3_stdB_le_stdA']['pass'] else 'FAIL'}")
        md.append(f"- 三项同时：{'是' if gate['all_three'] else '否'}（本表为 full 尺度复测读数，"
                  "verdict 由 S6 按 §六 映射出具）")
    else:
        md.append("- 缺完成 run，门线未评")
    md.append("")
    md.append("## B-no-cal 消融（E2/H2b 判据）")
    md.append("")
    if d_bc["per_seed"]:
        ms = d_bc["mean_std"]["val_admission_auroc"]
        ms_p = d_bc["mean_std"]["val_admission_auprc"]
        md.append(f"- B−B-no-cal（val 入院级 AUROC）：mean={_fmt(ms['mean'])}±{_fmt(ms['std'])}"
                  f"（n={ms['n_pairs']} 配对种子）；AUPRC mean={_fmt(ms_p['mean'])}±{_fmt(ms_p['std'])}")
        md.append("- B-no-cal=阶段一至三（8/6/4=18 epoch，与 B 的阶段1-3 步数逐字相同，"
                  "唯一差异=无阶段四 2 epoch 联合校准）；差值即阶段四效应读数。")
        md.append("- 解读映射存疑（交 S6 显式裁决）：design §二 H2(b) 失败条件把「B-no-cal≈B」"
                  "列为 H2 证伪且机制解释收窄为「短联合校准的再耦合效应」，而 §三 E2「若真则看到」"
                  "列又把「B-no-cal 与 B 差异不显著」列为末端校准效应的表现——两处对同一读数的"
                  "指向相反；本表只报数与两种解读，不预选。")
    else:
        md.append("- 无 B-no-cal 完成 run，消融未评")
    md.append("")
    md.append("## 梯度冲突（D7，H2a 代理）")
    md.append("")
    for r in runs:
        eps = [e for e in (r.get("epochs_records") or []) if e.get("grad_cos_mean")]
        if not eps:
            continue
        md.append(f"- {r['run_id']}（{len(eps)} 个记录 epoch）：")
        for e in eps:
            md.append(f"  - ep{e['epoch']} stage{e['stage']} "
                      f"logged={e.get('grad_logged_batches')} "
                      f"‖g‖={[None if v is None else round(v, 4) for v in (e.get('expert_grad_norm_mean') or [])]} "
                      f"cos={ {k: (None if v is None else round(v, 4)) for k, v in e['grad_cos_mean'].items()} }")
    all_cos = [v for r in runs for e in (r.get("epochs_records") or [])
               for v in (e.get("grad_cos_mean") or {}).values() if v is not None]
    if all_cos:
        md.append(f"- 全量 max|cos|={max(abs(c) for c in all_cos):.6f}"
                  "（H2a：A 臂全程 vs B 臂联合期的梯度冲突读数；"
                  "pilot 全量 max|cos|=0.008757，full 复测值见上）")
    md.append("")
    md.append("## 时间（D9：单训练 ≤720s）")
    md.append("")
    for r in runs:
        t = r.get("timing", {})
        md.append(f"- {r['run_id']}：total={t.get('total_sec')}s "
                  f"train={t.get('train_sec')}s per_epoch={t.get('per_epoch_sec')}")
    md.append("")
    md.append("## 实现记录与偏差（如实）")
    md.append("")
    n_ep_batches = runs[0]["epochs_records"][0].get("n_batches") if runs and runs[0].get("epochs_records") else None
    md.append(f"- D7「每 epoch ≥200 batch」在本档（train≈{runs[0]['data']['split_sizes']['train'] if runs else '?'} 例→"
              f"{n_ep_batches} batch/epoch）不可行（需额外前向反向、击穿 D9 上限），"
              "按 pilot 先例每 epoch 全量 batch 记录；原文口径需 full 语料"
              "（setA 20643×0.7≈14450 train→226 batch/epoch），语料下载后台续跑中。")
    md.append("- D7 记录范围相对 pilot 受控扩展：pilot=B 仅 stage4；full 按「≥2 专家联合可训即记录」"
              "执行（A 全程；B stage2+stage4；B-no-cal stage2），所有上报余弦均定义良好。")
    md.append("- B-no-cal 总 epoch=18（8/6/4）：与 B 的阶段1-3 逐字相同，唯一差异=无阶段四。")
    md.append("- padding 到 72h：填充步值通道 0、mask 通道 1，损失/评测只在真实小时步计算。")
    md.append("- B/B-no-cal 阶段切换时重建 optimizer（Adam 动量重置；被冻结参数状态本即无关）。")
    md.append("- 入院级标签=窗内任一小时阳性；评分=窗口概率最大值（D8）。")
    md.append("- 数据源=PhysioNet-CinC 2019 回退源（Moor 2023 不可得，S2 证据落盘）；"
              "全文语料下载后台续跑中（results/_download_state.json），本档按 D9 阶梯定档。")
    md.append("- D5 单体 LNN sanity 未在 full 重跑（预算优先 A/B/B-no-cal 九跑）："
              "pilot 已 PASS（A−LNN val +0.1499/test +0.1138），full 复用同实现同超参；"
              "若 A 臂 full 读数异常低再回查（design D5 口径）。")
    md.append("- 解读警告（交 S6）：B/A 专家更新预算比 0.5 系 4:3:2:1 设计固有"
              "（full 8/6/4/2 与 pilot 2/2/1/1 比例相同，仅绝对量放大），"
              "S6 终判须把该设计固有混杂显式写入解读。")
    md.append("- full 不终判 H1（design §六：full 出表，S6 按门线映射出具 verdict）。")
    md.append("")
    with open(os.path.join(RESULTS, "full_joint_vs_staged.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps({"n_runs": len(runs), "n_done": len(done),
                      "delta_B_minus_A_val_adm_auroc": d_auroc,
                      "delta_B_minus_BnoCal_val_adm_auroc":
                          d_bc["mean_std"]["val_admission_auroc"],
                      "h1_gate": gate},
                     ensure_ascii=False), flush=True)
    return out


# ---------------------------------------------------------------- 驱动
def run_all(device: str, lr: float, log_grad_every: int, resume: bool) -> dict:
    """九跑顺序执行：A×3 → B×3 → B-no-cal×3（D9：每 epoch 原子 ckpt，单 run 失败不拖垮其余）。"""
    dec = decide_tier(write=True)
    tier = dec["tier"]
    print(f"[tier] 在盘 {dec['n_on_disk_at_decision']} psv → {tier} 档"
          f"（速率证据 {dec['rate_evidence']}）", flush=True)
    plan = [(arm, seed) for arm in ARMS for seed in SEEDS]
    results = {}
    for arm, seed in plan:
        run_id = f"full_{arm}_seed{seed}"
        try:
            r = train_run(arm, seed, tier, epochs_for_arm(arm), lr, device,
                          log_grad_every, resume)
            results[run_id] = r["status"]
        except Exception as e:  # noqa: BLE001 — 单 run 崩溃留证不拖垮其余（D9 分段交单元）
            results[run_id] = f"error:{type(e).__name__}"
            update_progress(run_id, arm, seed, f"error:{type(e).__name__}", [],
                            extra={"error": str(e)[:500]})
            print(f"[error] {run_id}: {type(e).__name__}: {e}", flush=True)
        # 释放显存再进下一 run
        try:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    print(f"[plan-done] {json.dumps(results, ensure_ascii=False)}", flush=True)
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B", "BnoCal"], default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-patients", type=int, default=0,
                    help="0=按 D9 阶梯定档（默认）；显式值仅调试用")
    ap.add_argument("--epochs", type=int, default=0, help="0=按臂默认（A/B=20，B-no-cal=18）")
    ap.add_argument("--lr", type=float, default=mr_moe.LR, help="F3 两档：1e-3 / 3e-4")
    ap.add_argument("--tag", default="", help="run 名后缀（smoke 等；汇总时跳过）")
    ap.add_argument("--resume", action="store_true", help="从最新 ckpt 续跑（D9）")
    ap.add_argument("--log-grad-every", type=int, default=1, help="D7：每多少 batch 记一次梯度")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tier-decision", action="store_true", help="只定档并落盘")
    ap.add_argument("--all", action="store_true", help="九跑顺序执行")
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    if args.tier_decision:
        print(json.dumps(decide_tier(write=True), ensure_ascii=False), flush=True)
        return 0
    if args.summary:
        summary()
        return 0
    if args.all:
        run_all(args.device, args.lr, args.log_grad_every, args.resume)
        summary()
        return 0
    if args.arm is None or args.seed is None:
        ap.error("--arm 与 --seed 必填（或用 --all / --summary / --tier-decision）")
    n_patients = args.n_patients or decide_tier(write=False)["tier"]
    epochs = args.epochs or epochs_for_arm(args.arm)
    train_run(args.arm, args.seed, n_patients, epochs, args.lr,
              args.device, args.log_grad_every, args.resume, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
