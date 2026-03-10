"""
persistent_heuristics_I.dirichlet
----------------------------------
Re-exports the public API of L_function_zeros.py.
"""
import importlib.util, pathlib

_DATA = (pathlib.Path(__file__).resolve().parent.parent.parent
         / "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros"
         / "L_function_zeros.py")

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
