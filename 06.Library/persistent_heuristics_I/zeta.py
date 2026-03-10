"""
persistent_heuristics_I.zeta
-----------------------------
Re-exports the public API of zeta_zeros.py.

The zero data lives in _data/zeta_zeros.py (6000 LMFDB zeros of the
Riemann zeta function at 31 decimal places, imported as a trusted input).
"""

import importlib.util
import pathlib

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros" / "zeta_zeros.py"

_spec = importlib.util.spec_from_file_location("_zeta_zeros", _DATA)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

available = _mod.available
info      = _mod.info
get_zero  = _mod.get_zero
get_zeros = _mod.get_zeros

__all__ = ["available", "info", "get_zero", "get_zeros"]
