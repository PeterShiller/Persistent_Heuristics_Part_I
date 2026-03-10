"""
zeta.py  —  Public data API for Riemann zeta zeros (LMFDB, 6000 zeros at 31 decimal places)
============================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

persistent_heuristics_I.zeta
-----------------------------
Re-exports the public API of zeta_zeros.py (6000 LMFDB zeros of the
Riemann zeta function at 31 decimal places, imported as a trusted input).

This module uses runtime path resolution to locate zeta_zeros.py in the
repository tree.  It requires an editable install (pip install -e) from
a cloned copy of the repository; a standalone wheel is not supported.
"""
import importlib.util as _importlib_util
import pathlib as _pathlib

# persistent_heuristics_I/ -> 06.Library/ -> repo root
_REPO = _pathlib.Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros" / "zeta_zeros.py"

if not _DATA.exists():
    raise FileNotFoundError(
        f"zeta_zeros.py not found at expected path:\n  {_DATA}\n"
        "This package must be installed in editable mode (pip install -e) "
        "from a cloned copy of the Persistent_Heuristics_Part_I repository."
    )

_spec = _importlib_util.spec_from_file_location("_zeta_zeros", _DATA)
_mod  = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

available = _mod.available
info      = _mod.info
get_zero  = _mod.get_zero
get_zeros = _mod.get_zeros

# Note: get_zeros accepts an optional start parameter (default 1) allowing
# retrieval of any contiguous slice, e.g. get_zeros(n=100, start=500) returns
# zeros 500 through 599.  get_zero(k) is 1-indexed.  Both return Decimal objects.

__all__ = ["available", "info", "get_zero", "get_zeros"]
