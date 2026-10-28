#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""_d7fix_drive.py — S6-final-verdict-redesign 驱动：同 2000 例配置补跑 B×3（新 tag d7fix）。

背景（S5 判读〔52〕/S6 final_verdict）：full_train.py:398 原按 param group 名前缀
startswith("expert") 计联合可训专家组数，mr_moe stage4 组名为 "all" → 计数 0 →
log_grad=False → B 臂 stage4 六 epoch 遥测全缺失。修复（按实际可训专家参数集合计数
+每 epoch 非空断言）后，本驱动以**与 S5 逐字相同**的 2000 例配置补跑 B×3：

  n_patients=2000（显式钉住：D9 阶梯此时会按在盘 ~3.6 万 psv 选 8000 档，
  显式传参保证 D10 单变量纪律=同子采样/划分/顺序/scaler/τ/hidden/lr/batch 不变，
  唯一变量=D7 遥测修复）、epochs=20（8/6/4/2）、lr=1e-3、batch=64、
  log_grad_every=1、seed 0/1/2、tag=d7fix（run_id=full_B_seed{0,1,2}_d7fix，
  不覆盖不改写任何 S5 full_* 产物；也不跑 --summary，避免汇总 glob 到新 tag
  后重写 S5 的 CSV/JSON）。

仅标准库；子进程用 Windows 原生路径调 G:/ai/python11/python.exe（MSYS exec 墙绕行，
先例见 _pilot_drive.py  docstring）。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

PY = r"G:\ai\python11\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
N = "2000"            # D10：与 S5 full 逐字同档（_full_tier_decision.json 定档 2000）
TAG = "d7fix"         # 新 tag：run_id=full_B_seed{0,1,2}_d7fix
SEEDS = (0, 1, 2)


def main() -> int:
    log_path = os.path.join(PROJECT, "results", "_d7fix_train.log")
    log = open(log_path, "a", encoding="utf-8")

    def say(msg: str) -> None:
        line = f"[d7fix-drive {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    say(f"start n_patients={N} tag={TAG} seeds={SEEDS} py={PY}")
    rc_all = []
    for seed in SEEDS:
        cmd = [PY, os.path.join(HERE, "full_train.py"),
               "--arm", "B", "--seed", str(seed),
               "--n-patients", N, "--tag", TAG,
               "--device", "cuda"]
        say("run " + " ".join(cmd))
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=PROJECT).returncode
        say(f"B_seed{seed} rc={rc} elapsed={round(time.time() - t0, 1)}s")
        rc_all.append(rc)
    say(f"end rc={rc_all}")
    log.close()
    return 0 if all(r == 0 for r in rc_all) else 1


if __name__ == "__main__":
    sys.exit(main())
