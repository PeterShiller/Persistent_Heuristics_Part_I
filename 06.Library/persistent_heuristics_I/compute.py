"""
persistent_heuristics_I.compute
---------------------------------
Exposes the zero computation pipeline as an importable module and
as the CLI entry point `ph1-compute-zeros`.

Usage (CLI after pip install):
    ph1-compute-zeros --d 5 --nzeros 1000 --high-precision

Usage (Python):
    from persistent_heuristics_I.compute import compute_zeros
    compute_zeros(d=5, nzeros=20, high_precision=True)
"""

import importlib.util
import pathlib
import sys

# Load compute_Lfunc_zeros.py from its original location in the repo,
# which lives two levels up from this package directory.
_HERE = pathlib.Path(__file__).resolve().parent          # .../06.Library/persistent_heuristics_I/
_REPO = _HERE.parent.parent                              # repo root
_COMPUTE_DIR = _REPO / "00.Computing L(s, \u03c7) Zeros"
_COMPUTE     = _COMPUTE_DIR / "compute_Lfunc_zeros.py"

if str(_COMPUTE_DIR) not in sys.path:
    sys.path.insert(0, str(_COMPUTE_DIR))

_spec = importlib.util.spec_from_file_location("_compute_Lfunc_zeros", _COMPUTE)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

compute_zeros = _mod.compute_zeros
main          = _mod.main

__all__ = ["compute_zeros", "main"]
