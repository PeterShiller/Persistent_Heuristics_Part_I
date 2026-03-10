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
import importlib.util as _importlib_util
import pathlib as _pathlib
import warnings as _warnings

# persistent_heuristics_I/ -> 06.Library/ -> repo root
_REPO = _pathlib.Path(__file__).resolve().parent.parent.parent
_DATA = _REPO / "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros" / "L_function_zeros.py"

if not _DATA.exists():
    raise FileNotFoundError(
        f"L_function_zeros.py not found at expected path:\n  {_DATA}\n"
        "This package must be installed in editable mode (pip install -e) "
        "from a cloned copy of the Persistent_Heuristics_Part_I repository."
    )

_spec = _importlib_util.spec_from_file_location("_L_function_zeros", _DATA)
_mod  = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Fixed set of available squarefree discriminants.  Defined here as a constant
# so that _check_d does not depend on _mod being healthy.
_VALID_D = (2, 3, 5, 6, 7, 10, 11, 13)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_d(d):
    """Require d to be a plain int in _VALID_D (not bool, not numpy.int64, etc.).
    Users with numpy scalars should pass int(d)."""
    if isinstance(d, bool) or not isinstance(d, int):
        raise TypeError(
            f"d must be a Python int, got {type(d).__name__!r}.  "
            f"Valid values: {list(_VALID_D)}.  "
            f"If d comes from numpy, pass int(d)."
        )

def _warn_if_unsealed(d):
    """Emit a warning if the zero table for d has not been globally sealed."""
    seal = _mod.get_seal(d)
    if seal is None:
        _warnings.warn(
            f"The zero table for chi_{d} has not been globally sealed: "
            f"completeness by argument-principle winding number has not been "
            f"verified.  Individual zeros are Newton-certified but the table "
            f"may be incomplete.  Use get_seal({d}) to check status.",
            stacklevel=3,
        )

# ---------------------------------------------------------------------------
# Public API (wraps _mod with type-checking and unsealed warnings)
# ---------------------------------------------------------------------------

def available_characters():
    """Return the list of available squarefree discriminants."""
    return _mod.available_characters()

def info(d):
    """Return a metadata dict for character chi_d (conductor, field, zero count, precision)."""
    _check_d(d)
    return _mod.info(d)

def get_zero(d, k, as_string=False):
    """Return the k-th zero ordinate of L(s, chi_d) as a Decimal (1-indexed).
    Emits a warning if the table for d has not been globally sealed."""
    _check_d(d)
    _warn_if_unsealed(d)
    return _mod.get_zero(d, k, as_string=as_string)

def get_zeros(d, n=None, dp=70, as_strings=False):
    """Return the first n zero ordinates of L(s, chi_d) as a list of Decimals.
    dp controls decimal places (0..70, default 70); n=None returns all zeros.
    Emits a warning if the table for d has not been globally sealed."""
    _check_d(d)
    _warn_if_unsealed(d)
    if dp is None:
        dp = 70
    return _mod.get_zeros(d, n=n, dp=dp, as_strings=as_strings)

def get_bound(d, k):
    """Return the certified |L(1/2+i*gamma_k)| bound as a (float, int) tuple
    (mantissa, exponent), meaning |L(1/2+i*gamma_k)| < mantissa * 10^exponent.
    The bound holds for the individual zero regardless of seal status, but
    the table may be incomplete if the character is not yet sealed.  Emits
    a warning if the table for d has not been globally sealed."""
    _check_d(d)
    _warn_if_unsealed(d)
    return _mod.get_bound(d, k)

def get_bounds(d, n=None):
    """Return certified bounds for the first n zeros as a list of (float, int) tuples.
    Each tuple (mantissa, exponent) means |L(1/2+i*gamma)| < mantissa * 10^exponent.
    n=None returns bounds for all stored zeros.
    Emits a warning if the table for d has not been globally sealed."""
    _check_d(d)
    _warn_if_unsealed(d)
    total = _mod.info(d)["num_zeros"]
    if n is not None:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError(f"n must be an int or None, got {type(n).__name__!r}")
        if not (1 <= n <= total):
            raise ValueError(f"n={n} out of range [1, {total}]")
    return _mod.get_bounds(d, n=n)

def get_seal(d):
    """Return seal metadata dict for chi_d, or None if not yet sealed.
    A None return means global completeness has not been verified: individual
    zeros are Newton-certified but the table may be incomplete."""
    _check_d(d)
    return _mod.get_seal(d)

def get_bound_stats(d):
    """Return summary statistics for the certified bounds of chi_d.
    Returns a dict with keys: n, min_bound, max_bound, mean_bound, min_index, max_index.
    Bounds are stored as the exponent e in |L| < m * 10^e (so more negative = tighter).
    min_bound is the most negative exponent (best/tightest certification);
    max_bound is the least negative exponent (worst/loosest certification).
    min_index and max_index are the 1-based zero indices achieving these extremes.
    Emits a warning if the table for d has not been globally sealed."""
    _check_d(d)
    _warn_if_unsealed(d)
    return _mod.get_bound_stats(d)

__all__ = [
    "available_characters", "info", "get_zero", "get_zeros",
    "get_bound", "get_bounds", "get_seal", "get_bound_stats",
]
