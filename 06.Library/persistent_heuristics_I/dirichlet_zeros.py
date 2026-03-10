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
import warnings

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

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_d(d):
    """Require d to be a plain int.  Reject float, str, etc."""
    if not isinstance(d, int):
        raise TypeError(
            f"d must be an int, got {type(d).__name__!r}.  "
            f"Valid values: {_mod.available_characters()}"
        )

def _warn_if_unsealed(d):
    """Emit a warning if the zero table for d has not been globally sealed."""
    seal = _mod.get_seal(d)
    if seal is None:
        warnings.warn(
            f"The zero table for chi_{d} has not been globally sealed: "
            "completeness by argument-principle winding number has not been "
            "verified.  Individual zeros are Newton-certified but the table "
            "may be incomplete.  Use get_seal({d}) to check status.".format(d=d),
            stacklevel=3,
        )

# ---------------------------------------------------------------------------
# Public API (wraps _mod with type-checking and unsealed warnings)
# ---------------------------------------------------------------------------

def available_characters():
    """Return the list of available squarefree discriminants."""
    return _mod.available_characters()

def info(d):
    """Print metadata for character chi_d."""
    _check_d(d)
    return _mod.info(d)

def get_zero(d, k, as_string=False):
    """Return the k-th zero ordinate of L(s, chi_d) as a Decimal (1-indexed)."""
    _check_d(d)
    _warn_if_unsealed(d)
    return _mod.get_zero(d, k, as_string=as_string)

def get_zeros(d, n=None, dp=70, as_strings=False):
    """Return the first n zero ordinates of L(s, chi_d) as a list of Decimals."""
    _check_d(d)
    _warn_if_unsealed(d)
    return _mod.get_zeros(d, n=n, dp=dp, as_strings=as_strings)

def get_bound(d, k):
    """Return the certified |L(1/2+i*gamma_k)| bound as (mantissa, exponent)."""
    _check_d(d)
    return _mod.get_bound(d, k)

def get_bounds(d, n=None):
    """Return certified bounds for the first n zeros as a list of (mantissa, exponent)."""
    _check_d(d)
    return _mod.get_bounds(d, n=n)

def get_seal(d):
    """Return seal metadata dict for chi_d, or None if not yet sealed."""
    _check_d(d)
    return _mod.get_seal(d)

def get_bound_stats(d):
    """Return summary statistics for the certified bounds of chi_d."""
    _check_d(d)
    return _mod.get_bound_stats(d)

__all__ = [
    "available_characters", "info", "get_zero", "get_zeros",
    "get_bound", "get_bounds", "get_seal", "get_bound_stats",
]
