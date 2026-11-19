"""
Machine-specific paths, read from `local_paths.yaml` at the repo root
(git-ignored). Copy `local_paths.example.yaml` and edit it on each machine.
No hostname or absolute path is hard-coded anywhere else in the repo.
"""

import os

import yaml

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FILE = os.environ.get("PRISMSAM_PATHS", os.path.join(_REPO, "local_paths.yaml"))
_cache = None

SAM1_CKPT_NAMES = {"vit_b": "sam_vit_b_01ec64.pth",
                   "vit_l": "sam_vit_l_0b3195.pth",
                   "vit_h": "sam_vit_h_4b8939.pth"}


def _load():
    global _cache
    if _cache is None:
        if not os.path.exists(_FILE):
            raise FileNotFoundError(
                f"{_FILE} not found. Copy local_paths.example.yaml to local_paths.yaml "
                f"and fill in this machine's paths (or set PRISMSAM_PATHS).")
        with open(_FILE) as f:
            _cache = yaml.safe_load(f) or {}
    return _cache


def get(key):
    cfg = _load()
    if key not in cfg or cfg[key] in (None, ""):
        raise KeyError(f"'{key}' missing in {_FILE}")
    return cfg[key]


def sam1_checkpoint(model_type="vit_l"):
    return os.path.join(get("sam_ckpt_dir"), SAM1_CKPT_NAMES[model_type])