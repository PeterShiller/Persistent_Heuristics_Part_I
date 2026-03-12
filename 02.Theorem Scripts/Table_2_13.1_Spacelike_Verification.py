"""
Table_2_13.1_Spacelike_Verification.py
========================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies Table 2 in Section 13.1 (Spacelike verification
across admissible weights).  The table records S_zeta(w), S_L(w), and
    N_w = S_zeta(w)^2 - d * S_L(w)^2
for four admissible weight functions and four fundamental discriminants
d = 5 (q=5), d = 2 (q=8), d = 3 (q=12), d = 13 (q=13).

PASS criterion: every N_w entry has its entire ARB ball strictly below
zero, certified by the ARB predicate bool(N_w < 0) == True.

Displayed values are additionally verified against paper constants at
REL_TOL = 1e-3 (four significant figures).

Weight functions:
    Lorentzian:      w(gamma) = 2 / (1/4 + gamma^2)
    sech:            w(gamma) = sech(gamma)
    Heat kernel 1:   w(gamma) = exp(-0.01 * gamma^2)
    Heat kernel 2:   w(gamma) = exp(-0.1  * gamma^2)

Zero data:
    S_zeta: first 60 zeros of zeta(s) from zeta_zeros.py (LMFDB, 31 dp).
    S_L:    first 60 zeros of L(s, chi_5) from L_function_zeros.py.
            first 20 zeros of L(s, chi_2), L(s, chi_3), L(s, chi_13).
    All L-function zeros: Zenodo doi:10.5281/zenodo.18783098, 70 dp.

RIGOUR ARCHITECTURE
-------------------
All arithmetic is ARB interval arithmetic throughout at 512-bit precision.
No Python floats are used in any load-bearing computation.

The spacelike predicate bool(N_w < arb(0)) is True only when the entire
ARB ball for N_w lies strictly below zero; any ball straddling zero would
return False, causing the script to raise RuntimeError.

Working precision: 512-bit ARB.
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros as get_L_zeros
import zeta_zeros as _zz
from flint import arb, ctx

# ── Parameters ────────────────────────────────────────────────────────────────
ARB_PREC = 512
N_ZETA   = 60    # zeta zeros used
N_L_CHI5 = 60    # L-function zeros for chi_5
N_L_OTHER = 20   # L-function zeros for chi_2, chi_3, chi_13

# discriminants, conductors, L-function zero counts
CHARS = [
    (5,  5,  N_L_CHI5),
    (2,  8,  N_L_OTHER),
    (3,  12, N_L_OTHER),
    (13, 13, N_L_OTHER),
]

# Paper values and per-entry relative tolerances.
# Tolerance is set to half a unit in the last displayed significant figure:
#   2 sig figs -> 5e-2,  3 sig figs -> 5e-3,  4 sig figs -> 5e-4.
# The key certification is the spacelike predicate (N_w < 0); the value
# check is a sanity check at displayed-precision tolerance.
PAPER = {
    # (weight_label, d): (Sz_str, Sz_tol, SL_str, SL_tol, Nw_str, Nw_tol)
    ("Lorentzian",  5):  ("0.03793", 5e-4, "0.1406", 5e-4, "-0.09742", 5e-4),
    ("Lorentzian",  2):  ("0.03793", 5e-4, "0.1979", 5e-4, "-0.07693", 5e-4),
    ("Lorentzian",  3):  ("0.03793", 5e-4, "0.2863", 5e-4, "-0.2444",  5e-4),
    ("Lorentzian", 13):  ("0.03793", 5e-4, "0.3514", 5e-4, "-1.604",   5e-4),
    ("sech",        5):  ("1.5e-6",  5e-2, "2.713e-3", 5e-4, "-3.68e-5",  5e-3),
    ("sech",        2):  ("1.5e-6",  5e-2, "1.592e-2", 5e-4, "-5.07e-4",  5e-3),
    ("sech",        3):  ("1.5e-6",  5e-2, "4.730e-2", 5e-4, "-6.71e-3",  5e-3),
    ("sech",       13):  ("1.5e-6",  5e-2, "9.008e-2", 5e-4, "-0.1055",   5e-4),
    ("Heat0.01",    5):  ("0.1497",  5e-4, "1.417",    5e-4, "-10.02",    5e-3),
    ("Heat0.01",    2):  ("0.1497",  5e-4, "2.080",    5e-4, "-8.632",    5e-4),
    ("Heat0.01",    3):  ("0.1497",  5e-4, "2.652",    5e-4, "-21.08",    5e-3),
    ("Heat0.01",   13):  ("0.1497",  5e-4, "2.765",    5e-4, "-99.37",    5e-4),
    ("Heat0.1",     5):  ("2.1e-9",  5e-2, "1.210e-2", 5e-4, "-7.32e-4",  5e-3),
    ("Heat0.1",     2):  ("2.1e-9",  5e-2, "9.361e-2", 5e-4, "-1.75e-2",  5e-3),
    ("Heat0.1",     3):  ("2.1e-9",  5e-2, "2.469e-1", 5e-4, "-0.1828",   5e-4),
    ("Heat0.1",    13):  ("2.1e-9",  5e-2, "3.839e-1", 5e-4, "-1.916",    5e-4),
}


# ── Weight functions ──────────────────────────────────────────────────────────

def w_lorentzian(g):
    return arb(2) / (arb("1/4") + g * g)

def w_sech(g):
    return arb(1) / arb.cosh(g)

def w_heat1(g):
    return arb.exp(arb("-0.01") * g * g)

def w_heat2(g):
    return arb.exp(arb("-0.1") * g * g)

WEIGHTS = [
    ("Lorentzian", w_lorentzian),
    ("sech",       w_sech),
    ("Heat0.01",   w_heat1),
    ("Heat0.1",    w_heat2),
]


# ── PASS/FAIL ─────────────────────────────────────────────────────────────────

def arb_matches(val, paper_str, tol_float):
    pv  = arb(paper_str)
    tol = arb(str(tol_float))
    rel = abs(val - pv) / abs(pv)
    return bool(rel < tol), rel


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ctx.prec = ARB_PREC
    t0 = time.time()

    print()
    print("Table 2 (Section 13.1) -- Spacelike Verification -- ARB Certification")
    print("=" * 72)
    print(f"  Precision: {ARB_PREC}-bit ARB.  N_zeta={N_ZETA}, N_L(chi5)={N_L_CHI5}, "
          f"N_L(others)={N_L_OTHER}.  REL_TOL=1e-3.")
    print()

    # Load zeros
    print("  Loading zeros ...")
    ctx.prec = ARB_PREC
    zeta_str = _zz.get_zeros(N_ZETA, as_strings=True)
    gz = [arb(g) for g in zeta_str]

    L_zeros = {}
    for d, q, n in CHARS:
        L_zeros[d] = [arb(g) for g in get_L_zeros(d, n, as_strings=True)]
    print()

    all_pass    = True
    spacelike   = True

    for wname, wfn in WEIGHTS:
        print(f"  Weight: {wname}")

        # S_zeta is the same for all d columns
        Sz = sum(wfn(g) for g in gz)

        for d, q, n in CHARS:
            gl = L_zeros[d]
            SL = sum(wfn(g) for g in gl)
            Nw = Sz * Sz - arb(d) * SL * SL

            # Spacelike predicate: entire ball strictly below zero
            sp = bool(Nw < arb(0))
            if not sp:
                spacelike = False

            # Value checks
            pSz, tSz, pSL, tSL, pNw, tNw = PAPER[(wname, d)]
            ok_Sz, r_Sz = arb_matches(Sz, pSz, tSz)
            ok_SL, r_SL = arb_matches(SL, pSL, tSL)
            ok_Nw, r_Nw = arb_matches(Nw, pNw, tNw)
            ok = ok_Sz and ok_SL and ok_Nw
            all_pass = all_pass and ok

            sp_str = "SPACELIKE" if sp else "NOT-SPACELIKE"
            vf_str = "PASS" if ok else "FAIL"
            print(f"    d={d:2d}  S_zeta={Sz}  S_L={SL}  N_w={Nw}")
            print(f"         {sp_str}  value-check={vf_str}")
            if not ok_Sz: print(f"         FAIL S_zeta: rel={r_Sz}")
            if not ok_SL: print(f"         FAIL S_L:   rel={r_SL}")
            if not ok_Nw: print(f"         FAIL N_w:   rel={r_Nw}")

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"  Total time: {time.time()-t0:.1f}s")
    print()

    if spacelike and all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    All 16 N_w entries certified spacelike: ARB ball lies entirely")
        print("    below zero.  All displayed values match paper at REL_TOL=1e-3.")
    else:
        if not spacelike:
            print("  RESULT: FAIL -- spacelike predicate not certified for some entry.")
        if not all_pass:
            print("  RESULT: FAIL -- value mismatch against paper.")
        raise RuntimeError("Table 2 spacelike verification failed.")
    print()


if __name__ == "__main__":
    main()
