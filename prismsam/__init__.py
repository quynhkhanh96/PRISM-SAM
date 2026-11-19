"""PRISM-SAM library. Makes third_party/ importable (segment_anything, sam2, ...)."""
import os as _os
import sys as _sys

_TP = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "third_party")
if _TP not in _sys.path:
    _sys.path.insert(0, _TP)
