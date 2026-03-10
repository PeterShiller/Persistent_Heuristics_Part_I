"""
compute_dirichlet_zeros.py  —  ARB-certified zero computation pipeline for quadratic Dirichlet L-functions
==========================================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

persistent_heuristics_I.compute_dirichlet_zeros
-------------------------------------------------
Exposes the ARB-certified zero computation pipeline for quadratic Dirichlet
L-functions as an importable module and as the CLI entry point
`ph1-compute-zeros`.

This module uses runtime path resolution to locate compute_Lfunc_zeros.py
in the repository tree.  It requires an editable install (pip install -e)
from a cloned copy of the repository; a standalone wheel is not supported.
python-flint must also be installed (pip install 'persistent-heuristics-I[compute]').

Usage (CLI after pip install -e '[compute]'):
    ph1-compute-zeros --d 5 --nzeros 1000 --high-precision

Usage (Python):
    from persistent_heuristics_I.compute_dirichlet_zeros import compute_zeros
    compute_zeros(d=5, nzeros=20, high_precision=True)
"""

import importlib.util
import pathlib
import sys

# persistent_heuristics_I/ -> 06.Library/ -> repo root
_HERE        = pathlib.Path(__file__).resolve().parent
_REPO        = _HERE.parent.parent
_COMPUTE_DIR = _REPO / "00.Computing L(s, \u03c7) Zeros"
_COMPUTE     = _COMPUTE_DIR / "compute_Lfunc_zeros.py"

if not _COMPUTE.exists():
    raise FileNotFoundError(
        f"compute_Lfunc_zeros.py not found at expected path:\n  {_COMPUTE}\n"
        "This package must be installed in editable mode (pip install -e) "
        "from a cloned copy of the Persistent_Heuristics_Part_I repository."
    )

_mod = None

def _load():
    """Load compute_Lfunc_zeros.py on first use.  Deferred so that importing
    this module does not require python-flint to be installed.  The compute
    directory is added to sys.path only for the duration of exec_module and
    removed immediately after to avoid permanent namespace pollution."""
    global _mod
    if _mod is not None:
        return _mod
    try:
        import flint  # noqa: F401
    except ImportError:
        raise ImportError(
            "python-flint is required to use persistent_heuristics_I.compute_dirichlet_zeros.\n"
            "Install it with:\n"
            "    pip install 'persistent-heuristics-I[compute]'\n"
            "or independently with:\n"
            "    pip install python-flint"
        ) from None
    _path_inserted = str(_COMPUTE_DIR) not in sys.path
    if _path_inserted:
        sys.path.insert(0, str(_COMPUTE_DIR))
    try:
        spec = importlib.util.spec_from_file_location("_compute_Lfunc_zeros", _COMPUTE)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if _path_inserted:
            try:
                sys.path.remove(str(_COMPUTE_DIR))
            except ValueError:
                pass
        # Remove internal modules from sys.modules so they are not
        # importable as a side effect of loading the compute pipeline.
        for _key in ("Kronecker_character_data", "_compute_Lfunc_zeros"):
            sys.modules.pop(_key, None)
    _mod = mod
    return _mod


def compute_zeros(*args, **kwargs):
    """Compute zeros of a quadratic Dirichlet L-function.  See
    compute_Lfunc_zeros.py for full documentation."""
    return _load().compute_zeros(*args, **kwargs)


def main():
    """CLI entry point for ph1-compute-zeros."""
    return _load().main()


__all__ = ["compute_zeros", "main"]
