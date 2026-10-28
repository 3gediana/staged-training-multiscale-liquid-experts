#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mr_moe.py — MR-MoE 冻结实现（design.md D4 + D1 + D5 + D7）。

论文对应：arXiv:2606.12240v1 §2.4（Eq.8-11：K 个固定时间常数 LNN 专家 + softmax 门控
MLP + 读出，无 attention 的基础 MR-MoE）；论文无官方代码（checklist #5 "Not yet"），
本文件按 Eqs.1-18 自行实现。与论文的差异全部记入 results/baseline_diff.md。

冻结配置（D4）：
  - K=3 CfC 专家，各 hidden=1500（论文 §3.3「1500 neurons」）；
  - 固定 τ 比 1:10:100（dt=1h 网格下 τ=1h/10h/100h；论文未给数值，设计冻结取值）；
  - 门控 = softmax(MLP hidden 64)（论文「small MLP」未给尺寸，设计冻结 64）；
  - 读出 = 每专家线性 + 末端 sigmoid（论文 C_k 线性读出；sigmoid 为 BCE 所需）；
  - f_k = ncps 0.0.2 CfC 标准胞体（default 模式，论文未给 f_k 形式）。

ncps 0.0.2 接口事实（S2 ⓪/① 实测，证据见 results/operator_bench.md 与
results/ncps_tau_probe.json）：
  - CfC.forward(input, hx, timespans) 在 batch_first=True 且 timespans 形状 (B,L)、
    B>1 时直接 RuntimeError（cfc.py:166 `timespans[:, t].squeeze()` 得 (B,) 无法与
    (B,H) 广播）；
  - 固定 τ 注入走库路径的可用形态：timespans 形状 (1,L) 常向量（squeeze 后为 0 维
    标量，可与 (B,H) 广播）——本实现专家默认走该路径，无需 manual-loop、不改
    site-packages；
  - timespans=None ≡ 全 1（均匀网格 dt=1，实测 max_abs_diff=0.0）；
  - manual-loop 路径（ts 以 (B,1) 进 cell）与库路径逐样本一致（兄弟课题实测
    ~3.3e-07；本课题 probe ① 在真实配置复验）。

阶段训练（D1）：被冻结参数 requires_grad=False 且不进 optimizer。
  stage1 只训快专家 τ₁；stage2 冻结快专家训 τ₂,τ₃；stage3 冻结全部专家只训门控+读出；
  stage4 全部解冻短联合校准。
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
from ncps.torch import CfC

# D4 冻结超参（与 design.md §四 D4 一致；改这里=改冻结设计，须走 DECISIONS）
K_EXPERTS = 3
HIDDEN = 1500
TAUS: tuple[float, ...] = (1.0, 10.0, 100.0)  # 固定 τ 比 1:10:100
GATE_HIDDEN = 64
SEQ_LEN = 72  # D3：每条入院截断前 72h
BATCH_SIZE = 64  # D4（论文未给）
LR = 1e-3  # 论文 §3.3


class FixedTauCfCExpert(nn.Module):
    """固定 τ 的 CfC 专家（ncps 0.0.2 库路径，(1,L) 常向量 timespans 注入）。

    forward(x: (B,L,d)) -> (output: (B,L,H), h_final: (B,H))
    """

    def __init__(self, input_size: int, hidden: int = HIDDEN, tau: float = 1.0,
                 mode: str = "default") -> None:
        super().__init__()
        self.tau = float(tau)
        self.cfc = CfC(input_size, hidden, mode=mode)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, seq_len = x.size(0), x.size(1)
        # (1,L) 常向量：库内 squeeze() 得 0 维标量 τ，与 (B,H) 广播（B>1 可用形态）
        ts = x.new_full((1, seq_len), self.tau)
        out, h = self.cfc(x, timespans=ts)
        return out, h

    def forward_manual_loop(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """manual-loop 备选路径（probe ① 恒等性验证用；ts 以 (B,1) 进 cell）。"""
        b, seq_len = x.size(0), x.size(1)
        h_state = torch.zeros((b, self.cfc.state_size), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(seq_len):
            ts = x.new_full((b, 1), self.tau)
            h_out, h_state = self.cfc.rnn_cell.forward(x[:, t], h_state, ts)
            outs.append(h_out)
        return torch.stack(outs, dim=1), h_state


class MRMoE(nn.Module):
    """K=3 固定 τ CfC 专家 + softmax 门控 MLP + 每专家线性读出的 MR-MoE（Eq.8-11）。

    forward(x: (B,L,d)) -> logits: (B,L)（未过 sigmoid；损失用 BCEWithLogits）
    """

    def __init__(self, input_size: int, k: int = K_EXPERTS, hidden: int = HIDDEN,
                 taus: Sequence[float] = TAUS, gate_hidden: int = GATE_HIDDEN) -> None:
        super().__init__()
        assert k == len(taus), "专家数必须与 τ 个数一致"
        self.k, self.hidden, self.taus = k, hidden, tuple(float(t) for t in taus)
        self.experts = nn.ModuleList(
            [FixedTauCfCExpert(input_size, hidden, tau) for tau in self.taus]
        )
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hidden), nn.Tanh(), nn.Linear(gate_hidden, k)
        )
        self.readouts = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(k)])

    # ---------- 前向 ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expert_logits = []
        for expert, readout in zip(self.experts, self.readouts):
            out, _ = expert(x)  # (B,L,H)
            expert_logits.append(readout(out))  # (B,L,1)
        y = torch.cat(expert_logits, dim=-1)  # (B,L,K)
        pi = torch.softmax(self.gate(x), dim=-1)  # (B,L,K)
        return (y * pi).sum(dim=-1)  # (B,L)

    def forward_with_details(self, x: torch.Tensor) -> dict:
        """带中间量的前向（梯度冲突日志 / 分析用）。"""
        details = {"expert_out": [], "expert_logit": [], "pi": None, "logits": None}
        for expert, readout in zip(self.experts, self.readouts):
            out, _ = expert(x)
            details["expert_out"].append(out.detach())
            details["expert_logit"].append(readout(out).detach())
        details["pi"] = torch.softmax(self.gate(x), dim=-1).detach()
        details["logits"] = self.forward(x).detach()
        return details

    # ---------- D1 阶段训练参数组 ----------
    def param_groups_for_stage(self, stage: int) -> list[dict]:
        """D1 四阶段可训参数组（被冻结参数 requires_grad=False 且不进 optimizer）。

        stage1: 只训快专家（τ₁）
        stage2: 冻结快专家，训 τ₂/τ₃
        stage3: 冻结全部专家，只训门控 MLP 与读出层
        stage4: 全部参数（短联合校准）
        """
        if stage == 1:
            groups = [{"params": list(self.experts[0].parameters()), "name": "expert_tau1"}]
        elif stage == 2:
            groups = [{"params": list(self.experts[i].parameters()), "name": f"expert_tau{i+1}"}
                      for i in (1, 2)]
        elif stage == 3:
            groups = [
                {"params": list(self.gate.parameters()), "name": "gate"},
                {"params": list(self.readouts.parameters()), "name": "readouts"},
            ]
        elif stage == 4:
            groups = [{"params": list(self.parameters()), "name": "all"}]
        else:
            raise ValueError(f"未知 stage={stage}（D1 只定义 1-4）")
        for g in groups:
            for p in g["params"]:
                p.requires_grad_(True)
        trainable = {id(p) for g in groups for p in g["params"]}
        for p in self.parameters():
            if id(p) not in trainable:
                p.requires_grad_(False)
        return groups

    # ---------- D7 梯度冲突代理 ----------
    @torch.enable_grad()
    def expert_gradient_stats(self, loss: torch.Tensor) -> dict:
        """各专家参数梯度 g_k=∇_{θ_k}L 的范数与三对专家余弦相似度（D7）。

        阶段一至三专家不联合优化，叉乘余弦无定义——调用方负责只在联合训练
        （A 臂全程 / B 臂 stage4）时解释本输出。
        """
        flats = []
        for expert in self.experts:
            params = [p for p in expert.parameters() if p.requires_grad]
            if not params:
                flats.append(None)
                continue
            grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
            flat = torch.cat([(g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1))
                              for g, p in zip(grads, params)])
            flats.append(flat)
        norms = [None if f is None else float(f.norm().item()) for f in flats]
        cos = {}
        idx = [(i, j) for i in range(self.k) for j in range(i + 1, self.k)]
        for i, j in idx:
            if flats[i] is None or flats[j] is None:
                cos[f"cos_e{i+1}_e{j+1}"] = None
            else:
                cos[f"cos_e{i+1}_e{j+1}"] = float(
                    torch.nn.functional.cosine_similarity(flats[i], flats[j], dim=0).item()
                )
        return {"grad_norms": norms, "cosine_pairs": cos}


class MonolithicLNN(nn.Module):
    """单体 LNN 基线（D5 方向性健全性：A 臂须优于本模型）。

    单 CfC hidden=1500，均匀网格（timespans=None ≡ dt=1）+ 线性读出 + sigmoid。
    论文 monolithic LNN 同为单动力学系统 hidden 1500（§3.5.2）；其时间常数未给，
    本实现取均匀 dt=1（与论文 A 臂 core(x) 不传 timespans 的语义一致），
    记入 results/baseline_diff.md。
    """

    def __init__(self, input_size: int, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.hidden = hidden
        self.cfc = CfC(input_size, hidden, mode="default")
        self.readout = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.cfc(x, timespans=None)  # (B,L,H)
        return self.readout(out).squeeze(-1)  # (B,L)


def build_model(input_size: int, seed: int, kind: str = "mr_moe") -> torch.nn.Module:
    """按固定种子表构建模型（D10 单变量纪律：同初始化种子）。"""
    torch.manual_seed(seed)
    if kind == "mr_moe":
        return MRMoE(input_size)
    if kind == "lnn":
        return MonolithicLNN(input_size)
    raise ValueError(f"未知 kind={kind}")


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def _smoke() -> None:
    """最小前向/反向 smoke（小尺寸；真实尺寸基准见 experiments/operator_bench.py）。"""
    torch.manual_seed(0)
    d, b, seq = 8, 4, 6
    model = MRMoE(d, k=3, hidden=32, taus=(1.0, 10.0, 100.0), gate_hidden=16)
    x = torch.randn(b, seq, d, requires_grad=True)
    logits = model(x)
    assert logits.shape == (b, seq), logits.shape
    assert torch.isfinite(logits).all(), "前向出现非有限值"
    target = torch.randint(0, 2, (b, seq)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"
        assert torch.isfinite(p.grad).all(), f"{name} 梯度非有限"
    # manual-loop 与库路径恒等（固定 τ）
    e = model.experts[0]
    with torch.no_grad():
        o1, h1 = e(x)
        o2, h2 = e.forward_manual_loop(x)
    print(f"[smoke] mr_moe ok loss={loss.item():.4f} "
          f"manual_vs_lib_max_diff={(o1 - o2).abs().max().item():.2e} "
          f"params={count_params(model)}")
    # 单体 LNN smoke
    lnn = MonolithicLNN(d, hidden=32)
    ll = lnn(x)
    assert ll.shape == (b, seq)
    torch.nn.functional.binary_cross_entropy_with_logits(ll, target).backward()
    print(f"[smoke] lnn ok params={count_params(lnn)}")
    # D7 梯度冲突助手 smoke
    model.zero_grad()
    logits = model(x)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    stats = model.expert_gradient_stats(loss)
    print(f"[smoke] grad stats norms={[round(n, 4) for n in stats['grad_norms']]} "
          f"cos={ {k: round(v, 4) for k, v in stats['cosine_pairs'].items()} }")
    # D1 阶段参数组 smoke
    for stage in (1, 2, 3, 4):
        groups = model.param_groups_for_stage(stage)
        n_train = sum(p.numel() for g in groups for p in g["params"])
        print(f"[smoke] stage{stage} trainable={n_train} groups={[g['name'] for g in groups]}")


if __name__ == "__main__":
    _smoke()
