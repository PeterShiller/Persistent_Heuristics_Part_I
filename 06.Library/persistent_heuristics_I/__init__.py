"""
persistent_heuristics_I
=======================
ARB-certified zero tables and theorem scripts for:

    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

Installation
------------
The data API (get_zeros, get_bound, etc.) requires no binary dependencies:

    git clone https://github.com/PeterShiller/Persistent_Heuristics_Part_I
    pip install -e Persistent_Heuristics_Part_I/06.Library

To also use the computation pipeline (ph1-compute-zeros, compute_zeros()),
python-flint must be installed:

    pip install -e 'Persistent_Heuristics_Part_I/06.Library[compute]'

python-flint is not installed automatically because it is a compiled
extension and is not needed to access the precomputed zero tables.

This package requires an editable install from a cloned copy of the
repository.  The zero data files are not bundled; they are loaded at
runtime from their canonical locations in the repository tree.  A
standalone wheel distribution is not supported.

Public API
----------
Dirichlet L-function zeros (quadratic characters):

    from persistent_heuristics_I import get_zeros, get_bound, get_seal
    from persistent_heuristics_I import available_characters, info

    zeros = get_zeros(5, n=20)           # first 20 zeros of L(s, chi_5)
    zero  = get_zero(5, 1)               # first zero as arb ball
    bound = get_bound(5, 1)              # certified |L(1/2+i*gamma)| bound
    seal  = get_seal(5)                  # seal height and zero count

Riemann zeta zeros:

    from persistent_heuristics_I import zeta
    zeros = zeta.get_zeros(n=60)
    zero  = zeta.get_zero(1)

Precision note
--------------
All zero ordinates are stored at 70 decimal places (digits after the
decimal point).  get_zeros() returns arb balls at that precision by
default; pass dp=N to clip to N digits, or as_strings=True for string
output.  The certified bound for each zero satisfies |L(1/2+i*gamma)| <
10^{-420} at the table-wide floor (some zeros exceed 10^{-500}).
"""

from .dirichlet import (
    available_characters,
    info,
    get_zero,
    get_zeros,
    get_bound,
    get_bounds,
    get_seal,
    get_bound_stats,
)
from . import zeta

__all__ = [
    "available_characters",
    "info",
    "get_zero",
    "get_zeros",
    "get_bound",
    "get_bounds",
    "get_seal",
    "get_bound_stats",
    "zeta",
]
