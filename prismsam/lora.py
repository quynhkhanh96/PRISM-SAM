"""
LoRA modules.

`LoRALinear` is kept BYTE-COMPATIBLE with the pre-revision lora.py
(submodule names `.base`, `.A`, `.B`, same init, same dropout placement on
the low-rank branch output) so that every pre-revision `final.pth` loads
into a model built by the new code without key remapping. Do not rename.

`LoRAFusedQKV` is new (R5 `enc-lora`): SAM1's ViT attention computes
q, k, v with ONE fused `nn.Linear(dim, 3*dim)`. To keep the adapter
structure identical to the TWT recipe (`qkvo` = four independent adapters),
it attaches three independent rank-r adapters (q, k, v) to the fused layer
and concatenates their outputs. Output layout [q | k | v] matches SAM's
`qkv(x).reshape(B, HW, 3, heads, -1)`.
"""

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=8, dropout=0.05):
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.B.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + self.drop(self.B(self.A(x))) * self.scale


class LoRAFusedQKV(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=8, dropout=0.05):
        super().__init__()
        if base.out_features % 3 != 0:
            raise ValueError(f"fused qkv expects out_features % 3 == 0, got {base.out_features}")
        d_out = base.out_features // 3
        self.base = base
        self.A = nn.ModuleList([nn.Linear(base.in_features, rank, bias=False) for _ in range(3)])
        self.B = nn.ModuleList([nn.Linear(rank, d_out, bias=False) for _ in range(3)])
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        for a, b in zip(self.A, self.B):
            nn.init.kaiming_uniform_(a.weight, a=5 ** 0.5)
            nn.init.zeros_(b.weight)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        delta = torch.cat([b(a(x)) for a, b in zip(self.A, self.B)], dim=-1)
        return self.base(x) + self.drop(delta) * self.scale
