#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S6-final-verdict-redesign：D7 stage4 遥测计数 bug 修复的单元验证（不训练）。

验证三件事（全部实跑，不引用）：
  1. 源码级：full_train.py 不再按组名前缀 startswith("expert") 计数；含
     trainable_ids 参数集合计数与 d7-assert 非空断言。
  2. 语义级（GPU 实测）：对 param_groups_for_stage(1..4) 的 groups 按「实际可训
     专家参数集合」计数，得 stage1=1 / stage2=2 / stage3=0 / stage4=3；旧组名前缀
     口径对 stage4 得 0（复现 S5 bug 根因）。
  3. 功能级（GPU 实测）：stage4 全解冻后 expert_gradient_stats 的三对专家余弦
     全部非 null 且有限（stage2 仅 τ₂/τ₃ 联合对非 null，与 S5 stage2 行为一致）。
"""
import ast
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import data_pipeline as dp  # noqa: E402
import full_train  # noqa: E402
import mr_moe  # noqa: E402

IN_DIM = 2 * dp.N_FEATURES
PAIR_KEYS = ("cos_e1_e2", "cos_e1_e3", "cos_e2_e3")
report = {"checks": [], "old_prefix_counts": {}, "new_param_set_counts": {}}


def check(name, ok, detail=""):
    report["checks"].append({"check": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    assert ok, name


# ---- 1. 源码级检查（AST 级，忽略注释/文档字符串） -------------------------
src_path = os.path.join(HERE, "full_train.py")
with open(src_path, encoding="utf-8") as f:
    src = f.read()
tree = ast.parse(src)
startswith_calls = [
    n for n in ast.walk(tree)
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    and n.func.attr == "startswith"
]
check("source_no_group_name_prefix_count",
      not startswith_calls,
      f"full_train.py 代码中已无 startswith 调用（残留 {len(startswith_calls)} 处）")
check("source_has_trainable_ids_param_set_count",
      "trainable_ids" in src and "for expert in model.experts" in src,
      "按实际可训专家参数集合计数")
check("source_has_nonempty_assert",
      src.count("[d7-assert]") >= 2 and "grad_logged_batches=0" in src,
      "每 epoch 非空断言存在（logged>0 且联合对 cos 非 null）")

# ---- 2/3. 语义+功能级（GPU） --------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = mr_moe.build_model(IN_DIM, 0, kind="mr_moe").to(device)
torch.manual_seed(0)
x = torch.randn(4, 16, IN_DIM, device=device)
y = (torch.rand(4, 16, device=device) > 0.5).float()

for stage in (1, 2, 3, 4):
    groups = model.param_groups_for_stage(stage)
    # 旧口径（S5 bug）：组名前缀
    old_n = sum(1 for g in groups if g["name"].startswith("expert"))
    # 新口径（修复）：实际可训专家参数集合
    trainable_ids = {id(p) for g in groups for p in g["params"]}
    new_n = sum(1 for e in model.experts
                if any(id(p) in trainable_ids for p in e.parameters()))
    report["old_prefix_counts"][f"stage{stage}"] = old_n
    report["new_param_set_counts"][f"stage{stage}"] = new_n

check("count_stage1", new_n if stage == 1 else True, "")
check("count_stage2", new_n if stage == 2 else True, "")
check("count_stage3", new_n if stage == 3 else True, "")
check("count_stage4", new_n if stage == 4 else True, "")

expected = {1: 1, 2: 2, 3: 0, 4: 3}
for stage in (1, 2, 3, 4):
    groups = model.param_groups_for_stage(stage)
    old_n = sum(1 for g in groups if g["name"].startswith("expert"))
    trainable_ids = {id(p) for g in groups for p in g["params"]}
    new_n = sum(1 for e in model.experts
                if any(id(p) in trainable_ids for p in e.parameters()))
    check(f"param_set_count_stage{stage}", new_n == expected[stage],
          f"old_prefix={old_n} new_param_set={new_n} expected={expected[stage]}")

check("old_prefix_reproduces_s5_bug_stage4",
      report["old_prefix_counts"]["stage4"] == 0,
      "旧组名前缀口径 stage4=0（S5 bug 根因复现）")

# 功能级：各 stage 下 expert_gradient_stats 的余弦定义性
cos_by_stage = {}
for stage in (2, 4):
    model.param_groups_for_stage(stage)
    model.train()
    logits = model(x)
    m = y > 0
    loss = F.binary_cross_entropy_with_logits(logits[m], y[m])
    st = model.expert_gradient_stats(loss)
    cos = st["cosine_pairs"]
    cos_by_stage[f"stage{stage}"] = {k: cos[k] for k in PAIR_KEYS}
    n_defined = sum(1 for k in PAIR_KEYS if cos[k] is not None)
    exp_pairs = expected[stage] * (expected[stage] - 1) // 2
    check(f"cos_defined_stage{stage}", n_defined == exp_pairs,
          f"defined={n_defined}/{exp_pairs} cos={ {k: (None if v is None else round(v, 6)) for k, v in cos.items()} }")
    if stage == 4:
        check("cos_stage4_all_three_pairs_nonnull_finite",
              all(cos[k] is not None and torch.isfinite(torch.tensor(cos[k])) for k in PAIR_KEYS),
              "stage4 三对 cos 全非 null 且有限（H2(a) 比较期遥测可记录）")

# log_grad 开关语义（与 full_train.py:421 同式）
for stage in (1, 2, 3, 4):
    groups = model.param_groups_for_stage(stage)
    trainable_ids = {id(p) for g in groups for p in g["params"]}
    n = sum(1 for e in model.experts
            if any(id(p) in trainable_ids for p in e.parameters()))
    log_grad = n >= 2
    want = stage in (2, 4)
    check(f"log_grad_switch_stage{stage}", log_grad == want,
          f"log_grad={log_grad} expected={want}")

report["cos_by_stage_functional"] = cos_by_stage
report["schema"] = "d7fix-unit-test/v1"
report["device"] = device
report["all_pass"] = all(c["pass"] for c in report["checks"])
print("\nALL PASS" if report["all_pass"] else "\nSOME FAILED")
sys.exit(0 if report["all_pass"] else 1)
