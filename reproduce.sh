#!/usr/bin/env bash
# =============================================================================
# reproduce.sh — 一条命令复现本仓库全部结果表读数
#
# 做什么：从仓库内原始 run JSON（results/full_runs/*.json 等）独立复算
#   H1 终判（三臂 mean±std、配对差、预注册三门线）与 D7 stage4 修复复算
#   （遥测通过条件、配置不变性、修复前后对照、H2(a) 配对余弦）。
# 不做什么：不训练、不下载数据、不联网、不需要 GPU。复算脚本仅用 Python 标准库。
#
# 用法：
#   bash reproduce.sh            # 任意 Python 3.8+ 即可
#   PYTHON=/path/to/python bash reproduce.sh   # 指定解释器
#
# 退出码：0 = 复算通过且关键读数与冻结值一致；非 0 = 复算失败（断言/读数不符）。
# 注意：复跑会刷新 results/_d7fix_recompute.json 的 generated_at 时间戳
#   （数值不变，幂等；S6 判读单元已复跑验证）。
# =============================================================================
set -eu
cd "$(dirname "$0")"

# --- 1. 定位 Python 3（优先项目环境 G:/ai/python11，其次 PATH 上的 python/python3/py）---
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for cand in python python3 py "G:/ai/python11/python.exe"; do
    case "$cand" in
      */*)
        if [ -x "$cand" ]; then PY="$cand"; break; fi
        ;;
      *)
        if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
        ;;
    esac
  done
fi
if [ -z "$PY" ]; then
  echo "[reproduce] ERROR: 找不到 Python 3。请安装 Python 3.8+ 或用 PYTHON=/path/to/python 指定。" >&2
  exit 1
fi
echo "[reproduce] Python: $PY -> $("$PY" --version 2>&1)"
echo "[reproduce] cwd: $(pwd)"

# --- 2. H1 终判复算（S6-final-verdict 独立复算脚本；读原始 JSON+CSV，不训练）---
echo
echo "[reproduce] === 1/2 H1 终判复算：results/_s6_recompute.py ==="
"$PY" results/_s6_recompute.py

# --- 3. D7 stage4 修复复算（S6-final-verdict-redesign 独立复算脚本）---
echo
echo "[reproduce] === 2/2 D7 stage4 修复复算：results/_d7fix_recompute.py ==="
"$PY" results/_d7fix_recompute.py

# --- 4. 关键读数与本次提交的冻结值核对（防悄悄漂移；数值均来自上述两个 json）---
echo
echo "[reproduce] === 关键读数核对（vs 本提交冻结值）==="
"$PY" - <<'PYEOF'
import json

def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)

s6 = load("results/_s6_recompute.json")
d7 = load("results/_d7fix_recompute.json")

ok = True
def expect(label, got, want, tol=1e-6):
    global ok
    good = abs(float(got) - float(want)) <= tol
    ok = ok and good
    print("  [%s] %s = %.6f（冻结 %.6f）" % ("OK  " if good else "FAIL", label, got, want))

def expect_true(label, cond):
    global ok
    ok = ok and bool(cond)
    print("  [%s] %s" % ("OK  " if cond else "FAIL", label))

# H1：CSV 口径（4dp 显示层）复算值
a = s6["arm_metrics_val"]["A"]["adm_auroc"]
b = s6["arm_metrics_val"]["B"]["adm_auroc"]
expect("A val 入院级 AUROC mean", a["mean"], 0.830133)
expect("A val 入院级 AUROC std", a["std"], 0.013598)
expect("B val 入院级 AUROC mean", b["mean"], 0.747900)
expect("B val 入院级 AUROC std", b["std"], 0.034599)
bma = s6["paired_deltas_val"]["B_minus_A"]["adm_auroc"]
expect("B-A val 入院级 AUROC mean", bma["mean"], -0.082233)
expect("B-A val 入院级 AUROC std", bma["std"], 0.031776)
expect("B-A val 入院级 AUROC 负种子数", sum(1 for v in bma["per_seed"] if v < 0), 3, 0)
bmn = s6["paired_deltas_val"]["B_minus_BnoCal"]["adm_auroc"]
expect("B-BnoCal val 入院级 AUROC mean", bmn["mean"], 0.107033)
expect("B-BnoCal val 入院级 AUROC std", bmn["std"], 0.001550)
g = s6["h1_gates"]
expect_true("H1 三门线全 FAIL（all_gates_pass=False）", g["all_gates_pass"] is False)
expect_true("H1 门线①ΔAUROC≥+0.01 = False", g["gate_1_delta_adm_auroc_ge_0_01"] is False)
expect_true("H1 门线②ΔAUPRC≥-0.01 = False", g["gate_2_delta_adm_auprc_ge_minus_0_01"] is False)
expect_true("H1 门线③std(B)≤std(A) = False", g["gate_3_std_B_le_std_A"] is False)
expect_true("九 run 全 done", s6["progress_check"]["all_done"] is True)
expect_true("每 run val 阳性入院 = 9",
            all(v == [9] for v in s6["progress_check"]["n_pos_by_run"].values()))

# D7 修复：通过条件 + 前后对照 + bit 级复现 + H2(a) 配对差
expect_true("D7 遥测通过条件 all_pass", d7["telemetry_pass_conditions"]["all_pass"] is True)
expect_true("D10 配置不变性 all_equal", d7["config_invariance_d10"]["all_equal"] is True)
expect("d7fix vs S5 性能最大绝对差（bit 级=0）",
       d7["performance_alignment_d7fix_vs_s5"]["max_abs_diff"], 0.0)
ba = d7["d7_before_after_stage4"]
expect("修复前 stage4 logged batches", ba["before_s5"]["stage4_logged_batches"], 0)
expect("修复前 stage4 定义余弦", ba["before_s5"]["stage4_defined_cosines"], 0)
expect("修复后 stage4 logged batches", ba["after_d7fix"]["stage4_logged_batches"], 132)
expect("修复后 stage4 定义余弦", ba["after_d7fix"]["stage4_defined_cosines"], 18)
h2a = d7["h2a_adjudication"]["primary_A_full_vs_B_stage4"]
expect("H2(a) 配对差 cos_e1_e2 mean", h2a["cos_e1_e2"]["paired_diff_mean_std"]["mean"], -0.00486814342772656)
expect("H2(a) 配对差 cos_e1_e3 mean", h2a["cos_e1_e3"]["paired_diff_mean_std"]["mean"], 0.0012015587290327607)
expect("H2(a) 配对差 cos_e2_e3 mean", h2a["cos_e2_e3"]["paired_diff_mean_std"]["mean"], -0.0018929816832248681)
expect_true("H2(a) 三对非全负（预注册方向不成立）",
            d7["h2a_adjudication"]["direction_summary"]["all_three_pairs_mean_diff_negative"] is False)
expect_true("H2(a) verdict = inconclusive", d7["verdict"]["h2a"] == "inconclusive")

print()
if ok:
    print("[reproduce] ALL CHECKS PASS —— 复算结果与冻结读数一致。")
    print("[reproduce] 结论口径：H1 falsified（三门线全 FAIL）；H2(a) inconclusive（不支持）；"
          "H2 overall inconclusive；负结果不包装成支持性结论。")
else:
    print("[reproduce] CHECK FAILED —— 复算读数与冻结值不一致，请检查 results/ 原始文件是否被改动。")
    raise SystemExit(1)
PYEOF

echo
echo "[reproduce] DONE（rc=0）。输出 json：results/_s6_recompute.json、results/_d7fix_recompute.json"
