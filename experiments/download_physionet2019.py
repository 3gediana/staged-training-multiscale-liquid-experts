#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""download_physionet2019.py — PhysioNet-CinC 2019 全文语料断点续跑下载器（keep-alive 并行）。

S2 data_check 定影数据源=PhysioNet 2019（回退源）。S3 pilot 前需完整 A+B 语料。

实测结构（2026-10-27 live，results/data_check.json P1 + 前台边界探针）：
  - training_setA/：p000001.psv .. p020643.psv（二分实测上界 20,643 文件）
  - training_setB/：p100001.psv .. p120000.psv（边界实测 20,000 文件；命名全局连号，
    setB 从 p100001 起——data_check 的 p000001 探针假设对 setB 不适用，已前台修正）
  - 单文件 KB 级，A+B 合计约 300-400MB（夜班 ≤2GB 预算内）

网络事实（2026-10-27 实测）：
  - 本地代理 127.0.0.1:7897 为瓶颈；physionet.org 直连（trust_env=False）稳定；
  - curl 子进程逐文件 ~1.1 文件/s；requests 线程级持久 Session（keep-alive）
    16 工人 2.5 文件/s、32 工人 3.5 文件/s（并发上限 ~4/s，服务端按 IP 排队）。

断点续跑：跳过已存在文件；.part 原子替换；进度原子落盘 results/_download_state.json；
D 盘余 <50GB 自动停。

用法：G:/ai/python11/python.exe experiments/download_physionet2019.py [--workers 32]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT, "experiments", "data")
STATE = os.path.join(PROJECT, "results", "_download_state.json")

BASE = "https://physionet.org/files/challenge-2019/1.0.0/training"
# 实测区间（data_check P1 二分 + 前台边界探针，2026-10-27）
RANGES = {"training_setA": (1, 20643), "training_setB": (100001, 120000)}

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.trust_env = False  # 直连（绕本地代理）
        _local.s = s
    return _local.s


def _reset_session() -> None:
    s = requests.Session()
    s.trust_env = False
    _local.s = s


def dl(name: str, i: int, retries: int = 4) -> bool:
    dst = os.path.join(DATA_DIR, name, f"p{i:06d}.psv")
    tmp = dst + ".part"
    for attempt in range(retries):
        try:
            r = _session().get(f"{BASE}/{name}/p{i:06d}.psv", timeout=60)
            if r.status_code == 200 and len(r.content) > 0:
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, dst)
                return True
            if r.status_code == 404:
                return False  # 不存在（区间外），不重试
        except Exception:  # noqa: BLE001
            _reset_session()
        time.sleep(1 + attempt)
    if os.path.exists(tmp):
        os.remove(tmp)
    return False


def download_range(name: str, lo: int, hi: int, workers: int, batch: int,
                   stop_at: int = 0) -> dict:
    """stop_at>0：该 set 在盘文件数达到 stop_at 即干净停止（S3 pilot 启动条件；

    断点续跑语义不变——跳过已存在，。part 原子替换，进度原子落盘。
    """
    d = os.path.join(DATA_DIR, name)
    os.makedirs(d, exist_ok=True)
    t0 = time.time()

    def _n_on_disk() -> int:
        return sum(1 for f in os.listdir(d) if f.endswith(".psv"))

    missing = [i for i in range(lo, hi + 1)
               if not (os.path.exists(os.path.join(d, f"p{i:06d}.psv"))
                       and os.path.getsize(os.path.join(d, f"p{i:06d}.psv")) > 0)]
    ok = fail = 0
    for s in range(0, len(missing), batch):
        chunk = missing[s:s + batch]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(lambda i: dl(name, i), chunk))
        ok += sum(1 for r in res if r)
        fail += sum(1 for r in res if not r)
        state = {"set": name, "range": [lo, hi], "n_missing_total": len(missing),
                 "n_ok_this_run": ok, "n_fail_this_run": fail,
                 "elapsed_sec": round(time.time() - t0, 1),
                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(STATE + ".tmp", "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False)
        os.replace(STATE + ".tmp", STATE)
        if stop_at:
            n_disk = _n_on_disk()
            if n_disk >= stop_at:
                return {"set": name, "range": [lo, hi],
                        "stop_reason": f"在盘 {n_disk}≥stop_at {stop_at}，按 S3 pilot 启动条件停",
                        "n_on_disk": n_disk, "n_ok_this_run": ok,
                        "n_fail_this_run": fail, "n_missing_total": len(missing),
                        "elapsed_sec": round(time.time() - t0, 1)}
        free = shutil.disk_usage("D:/").free / 1024**3
        if free < 50:
            return {"set": name, "stop_reason": f"D 盘余 {free:.1f}GB<50GB 停",
                    "n_ok_this_run": ok, "n_fail_this_run": fail,
                    "n_missing_total": len(missing)}
    return {"set": name, "stop_reason": "done", "n_missing_total": len(missing),
            "n_ok_this_run": ok, "n_fail_this_run": fail,
            "elapsed_sec": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--stop-after-seta", type=int, default=0,
                    help="setA 在盘文件数达到 N 即停（S3 pilot 启动条件；0=不下发限制）")
    args = ap.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    for si, (name, (lo, hi)) in enumerate(RANGES.items()):
        stop_at = args.stop_after_seta if (si == 0 and args.stop_after_seta) else 0
        r = download_range(name, lo, hi, args.workers, args.batch, stop_at)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        if r.get("stop_reason") != "done":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
