#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""data_pipeline.py — D3 预处理冻结实现（小时网格 / 72h 截断 / ffill+mask / min-max / 标签）。

D3 冻结（design.md §四）：
  - 小时网格重采样（dt=1）；
  - 每条入院截断前 72h；
  - 缺失值前向填充 + 缺失 mask 作为输入通道拼接（禁 0 填充裸奔进前向/反向）；
  - min-max 归一化只用 train 划分拟合；
  - 标签=Sepsis-3 onset，预测目标=未来 6h 内 onset（y(t)=1[onset∈(t,t+6h]]，
    PhysioNet 2019/Che 2018 标准式）。

PhysioNet-CinC 2019 数据形态（S2 data_check 实测记录于 results/data_check.json）：
  - training_setA/ 与 training_setB/ 下每患者一个 .psv（列=40 临床变量 + SepsisLabel）；
  - 行=小时步（ICULOS 1..N）；缺失=空值。

本模块职责：.psv → 模型输入张量（x: (T, 2d) = [值, mask]，y: (T,)）。
标签约定（实测记录项）：直接采用 PhysioNet 2019 随数据提供的 SepsisLabel
（挑战标准 6h-ahead 标注：septic 患者从 onset 前 6h 起 label=1）。与 D3 公式
y(t)=1[onset∈(t,t+6h]] 的关系：onset 之后时段挑战标注为 1 而公式为 0；本管线取
挑战标注（与官方评测口径一致），差异记入 results/baseline_diff.md。

用法（smoke）：G:/ai/python11/python.exe experiments/data_pipeline.py
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# PhysioNet 2019 的 34 个时变临床特征（8 生命体征 + 26 实验室值）。
# 静态/元数据列（Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS）不作输入通道：
# 对每小时预测无增量信息且为常数通道；选择记入 results/baseline_diff.md。
VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
FEATURES: tuple[str, ...] = tuple(VITALS + LABS)
LABEL_COL = "SepsisLabel"
N_FEATURES = len(FEATURES)  # 34

SEQ_LEN = 72          # D3：前 72h 截断
DT_HOURS = 1          # D3：小时网格


@dataclass
class PatientExample:
    pid: str
    x: np.ndarray            # (T, 2*d)：前 d 列=前向填充后的值，后 d 列=缺失 mask(1=缺失)
    y: np.ndarray            # (T,) float32：6h-ahead Sepsis-3 标签
    t: int                   # 实际步数（≤SEQ_LEN）
    n_missing_raw: int = 0   # 截断窗内原始缺失单元数（诊断用）
    onset_hour: int | None = None  # 首个 label=1 的小时（None=non-septic）


@dataclass
class Scaler:
    """min-max 归一化（只用 train 划分拟合；D3）。"""

    dmin: np.ndarray = field(default_factory=lambda: np.zeros(N_FEATURES))
    dmax: np.ndarray = field(default_factory=lambda: np.ones(N_FEATURES))

    def fit(self, values: np.ndarray) -> "Scaler":
        # values: (N, d) 仅用可观测值拟合（NaN 按列忽略）
        # 全缺失列守卫：nanmin/nanmax 对全 NaN 列返回 NaN（S3 smoke 实测 EtCO2 在
        # 96 例子采样 train 中 100% 缺失即触发），该列按 [0,1] 常数列处理——
        # 其所有取值本就被 mask=1 标记为缺失，不参与模型可见信息。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            dmin = np.nanmin(values, axis=0)
            dmax = np.nanmax(values, axis=0)
        bad = ~(np.isfinite(dmin) & np.isfinite(dmax))
        dmin = np.where(bad, 0.0, dmin)
        dmax = np.where(bad, 1.0, dmax)
        rng = np.asarray(dmax - dmin, dtype=np.float64)
        rng = np.where(rng < 1e-8, 1.0, rng)  # 常数列防除零
        self._rng = rng.astype(np.float32)
        self.dmin = dmin.astype(np.float32)
        self.dmax = dmax.astype(np.float32)
        return self

    def transform(self, v: np.ndarray) -> np.ndarray:
        out = (v - self.dmin) / self._rng
        return np.clip(out, 0.0, 1.0)


def load_patient_psv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """读单患者 .psv → (values: (T,d) float32 含 NaN, labels: (T,) float32)。"""
    df = pd.read_csv(path, sep="|")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺列 {missing[:5]}...")
    values = df[list(FEATURES)].to_numpy(dtype=np.float32)
    labels = df[LABEL_COL].to_numpy(dtype=np.float32)
    return values, labels


def build_example(pid: str, values: np.ndarray, labels: np.ndarray,
                  seq_len: int = SEQ_LEN) -> PatientExample:
    """D3 单患者处理：72h 截断 → 前向填充 → 0 填残余首段缺失 → mask 通道。

    禁 0 填充裸奔：所有被 0 填的位置 mask=1（模型可见缺失），ffill 只传播真实观测。
    """
    values = values[:seq_len]
    labels = labels[:seq_len]
    t = values.shape[0]
    mask = np.isnan(values).astype(np.float32)
    n_missing_raw = int(mask.sum())
    # 前向填充（沿时间）；首段残余 NaN 保留为 NaN（Scaler.transform 前替换）
    df = pd.DataFrame(values).ffill()
    filled = df.to_numpy(dtype=np.float32)
    lead_nan = np.isnan(filled)
    filled[lead_nan] = 0.0
    mask[lead_nan] = 1.0  # 首段缺失也标记（0 填充但可见）
    x = np.concatenate([filled, mask], axis=1).astype(np.float32)  # (T, 2d)
    onset = int(np.argmax(labels > 0)) if (labels > 0).any() else None
    return PatientExample(pid=pid, x=x, y=labels.astype(np.float32), t=t,
                          n_missing_raw=n_missing_raw, onset_hour=onset)


def load_dataset(root: str, set_names: Sequence[str] = ("training_setA", "training_setB"),
                 limit: int | None = None) -> tuple[list[PatientExample], list[str]]:
    """从 PhysioNet 2019 目录树加载患者（root/training_setX/*.psv）。

    返回 (examples, skipped_files)：列缺失/解析失败的文件跳过并记录（下载残缺容错）。
    """
    examples: list[PatientExample] = []
    skipped: list[str] = []
    for name in set_names:
        d = os.path.join(root, name)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".psv"))
        if limit is not None:
            files = files[:limit]
        for f in files:
            path = os.path.join(d, f)
            pid = f[:-4]
            try:
                values, labels = load_patient_psv(path)
            except Exception:  # noqa: BLE001 — 残缺文件跳过并记录
                skipped.append(os.path.relpath(path, root))
                continue
            examples.append(build_example(pid, values, labels))
    return examples, skipped


def fit_scaler_on_train(examples: Iterable[PatientExample]) -> Scaler:
    """只用 train 划分拟合 min-max（D3）；按特征列收集全部可观测值。"""
    vals = []
    for ex in examples:
        filled = ex.x[:, :N_FEATURES]              # (T,d)
        mask = ex.x[:, N_FEATURES:]                # (T,d) 1=缺失
        obs_values = np.where(mask == 0.0, filled, np.nan)  # 缺失处置 NaN 后按列忽略
        vals.append(obs_values)
    if not vals:
        return Scaler().fit(np.zeros((1, N_FEATURES), dtype=np.float32))
    return Scaler().fit(np.concatenate(vals, axis=0))


def apply_scaler(examples: Iterable[PatientExample], scaler: Scaler) -> None:
    """就地应用归一化到 x 的值通道（mask 通道不动）。"""
    for ex in examples:
        ex.x[:, :N_FEATURES] = scaler.transform(ex.x[:, :N_FEATURES]).astype(np.float32)


def patient_split(examples: Sequence[PatientExample], seed: int = 0,
                  ratios: tuple[float, float, float] = (0.7, 0.1, 0.2)
                  ) -> dict[str, list[PatientExample]]:
    """患者级随机划分（固定种子；D6/D10 单变量纪律）。

    协议说明：PhysioNet 2019 未随数据提供官方划分文件；Che et al. 2018 划分为
    论文引用的划分依据。本实现用固定种子患者级随机划分作为其操作化，记入
    results/baseline_diff.md。严禁按小时切分（同一患者跨划分=泄漏）。
    """
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(examples))
    n = len(examples)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    return {
        "train": [examples[i] for i in idx[:n_tr]],
        "val": [examples[i] for i in idx[n_tr:n_tr + n_va]],
        "test": [examples[i] for i in idx[n_tr + n_va:]],
    }


def dataset_stats(examples: Sequence[PatientExample]) -> dict:
    """例数 / 序列长度分布 / 标签率（S2 data_check 记录项）。"""
    lens = np.array([ex.t for ex in examples], dtype=np.int64)
    pos_hours = np.array([int((ex.y > 0).sum()) for ex in examples], dtype=np.int64)
    tot_hours = int(lens.sum())
    n_septic = int(sum(1 for ex in examples if ex.onset_hour is not None))
    return {
        "n_patients": len(examples),
        "n_septic_patients": n_septic,
        "septic_patient_rate": round(n_septic / max(len(examples), 1), 4),
        "total_hours": tot_hours,
        "positive_hours": int(pos_hours.sum()),
        "hourly_label_rate": round(float(pos_hours.sum()) / max(tot_hours, 1), 5),
        "seq_len": {
            "min": int(lens.min()) if len(lens) else 0,
            "p25": float(np.percentile(lens, 25)) if len(lens) else 0.0,
            "median": float(np.median(lens)) if len(lens) else 0.0,
            "p75": float(np.percentile(lens, 75)) if len(lens) else 0.0,
            "p95": float(np.percentile(lens, 95)) if len(lens) else 0.0,
            "max": int(lens.max()) if len(lens) else 0,
            "mean": round(float(lens.mean()), 2) if len(lens) else 0.0,
        },
    }


def _smoke() -> None:
    """管线 smoke：合成微型 .psv（仅代码路径单测，不是 benchmark 数据）。"""
    import tempfile

    rng = np.random.RandomState(0)
    with tempfile.TemporaryDirectory() as td:
        d = os.path.join(td, "training_setA")
        os.makedirs(d)
        for i in range(4):
            t = int(rng.randint(20, 72))
            df = pd.DataFrame(rng.randn(t, N_FEATURES).round(3), columns=list(FEATURES))
            df.iloc[::7] = np.nan  # 撒缺失
            labels = np.zeros(t, dtype=int)
            if i % 2 == 0:
                labels[max(0, t // 2):] = 1  # 半程起 septic
            df[LABEL_COL] = labels
            df.to_csv(os.path.join(d, f"p{i:06d}.psv"), sep="|", index=False)
        examples, skipped = load_dataset(td)
        assert len(examples) == 4 and not skipped
        scaler = fit_scaler_on_train(examples)
        apply_scaler(examples, scaler)
        splits = patient_split(examples, seed=0)
        stats = dataset_stats(examples)
        ex = examples[0]
        assert ex.x.shape[1] == 2 * N_FEATURES
        assert set(np.unique(ex.x[:, N_FEATURES:])) <= {0.0, 1.0}
        print(f"[smoke] data_pipeline ok n={len(examples)} "
              f"splits={ {k: len(v) for k, v in splits.items()} }")
        print(f"[smoke] stats={stats}")
        print(f"[smoke] x[0,:6]={ex.x[0, :6].round(3).tolist()} mask[0,:6]={ex.x[0, N_FEATURES:N_FEATURES+6].tolist()}")


if __name__ == "__main__":
    _smoke()
