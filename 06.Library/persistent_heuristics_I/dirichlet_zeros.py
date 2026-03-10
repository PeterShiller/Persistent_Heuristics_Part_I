"""
dirichlet_zeros.py  —  Public data API for ARB-certified zeros of quadratic Dirichlet L-functions
==================================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

persistent_heuristics_I.dirichlet_zeros
-----------------------------------------
Re-exports the public data API of L_function_zeros.py: precomputed
ARB-certified zeros of quadratic Dirichlet L-functions at 70 decimal places.

This module uses runtime path resolution to locate L_function_zeros.py
in the repository tree.  It requires an editable install (pip install -e)
from a cloned copy of the repository; a standalone wheel is not supported.
"""
import importlib.util
import pathlib

# persistent_heuristics_I/ -> 06.Library/ -> repo root
_REPO = pathlib.Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros" / "L_function_zeros.py"

if not _DATA.exists():
    raise FileNotFoundError(
        f"L_function_zeros.py not found at expected path:\n  {_DATA}\n"
        "This package must be installed in editable mode (pip install -e) "
        "from a cloned copy of the Persistent_Heuristics_Part_I repository."
    )

_spec = importlib.util.spec_from_file_location("_L_function_zeros", _DATA)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

available_characters = _mod.available_characters
info                 = _mod.info
get_zero             = _mod.get_zero
get_zeros            = _mod.get_zeros
get_bound            = _mod.get_bound
get_bounds           = _mod.get_bounds
get_seal             = _mod.get_seal
get_bound_stats      = _mod.get_bound_stats

__all__ = [
    "available_characters", "info", "get_zero", "get_zeros",
    "get_bound", "get_bounds", "get_seal", "get_bound_stats",
]
