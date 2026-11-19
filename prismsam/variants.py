"""
Canonical adapt-configs for the R1/R5 grid (TMLR Review 1) and the
`{site}-{param}[-{lora spec}]__tok{0|1}` part of experiment names.

EXPECTED_TRAINABLE: parameter counts for SAM1 ViT-L, mask tokens included
(1,024). Checked by tests/test_variants.py; quoted in the paper.
"""

VARIANTS = {
    "tok": {"site": "tok", "param": None, "tok": True},
    "twt-lora": {"site": "twt", "param": "lora", "tok": True,
                 "lora": {"rank": 8, "alpha": 8, "dropout": 0.05, "scope": "full", "proj": "qkvo"}},
    "twt-full": {"site": "twt", "param": "full", "tok": True},
    "nontwt-full": {"site": "nontwt", "param": "full", "tok": True},
    "nontwt-full-tok0": {"site": "nontwt", "param": "full", "tok": False},   # optional (R2)
    "nontwt-lora": {"site": "nontwt", "param": "lora", "tok": True,
                    "lora": {"rank": 18, "alpha": 18, "dropout": 0.05}},
    "enc-lora": {"site": "enc", "param": "lora", "tok": True,
                 "lora": {"rank": 6, "alpha": 6, "dropout": 0.05, "proj": "qkvo", "last_k": 2}},
    "dec-full": {"site": "dec", "param": "full", "tok": True},
}

EXPECTED_TRAINABLE = {
    "tok": 1_024,
    "twt-lora": 95_232,          # 94,208 adapter + 1,024
    "twt-full": 3_292_288,       # 3,291,264 + 1,024
    "nontwt-full": 634_208,      # 559,232 hypernet + 73,952 upscaling + 1,024
    "nontwt-full-tok0": 633_184,
    "nontwt-lora": 95_488,       # 18 * 5,248 = 94,464 + 1,024
    "enc-lora": 99_328,          # 2 blocks * 4 * 6 * 2,048 = 98,304 + 1,024
    "dec-full": 4_058_340,       # of which 132,612 (IoU head) never receive gradient
}


def variant_name(cfg):
    """e.g. 'twt-lora-scfull-qkvo-r8a8__tok1', 'enc-lora-last2-qkvo-r6a6__tok1'."""
    site, param = cfg["site"], cfg.get("param")
    if site == "tok":
        return "tok__tok1"
    parts = [site, param]
    if param == "lora":
        lc = cfg["lora"]
        if site == "twt":
            parts += [f"sc{lc.get('scope', 'full')}", lc.get("proj", "qkvo")]
        elif site == "enc":
            parts += [f"last{lc['last_k']}", lc.get("proj", "qkvo")]
        parts.append(f"r{lc['rank']}a{lc['alpha']}")
    return "-".join(parts) + f"__tok{int(bool(cfg.get('tok')))}"
