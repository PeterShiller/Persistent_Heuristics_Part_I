"""
Table_1_Proposition_9.1(Verified_Asymptotic_Independence).py
=============================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table in Section 9 (Proposition 9.1,
Verified asymptotic independence).  The proposition requires that no
nontrivial integer relation exists among the combined zero set

    {gamma_1, ..., gamma_{M1}} union {gamma'_1, ..., gamma'_{M2}}

with M1 = M2 = 20.  The paper restricts to order-2 and order-3
cross-function relations and order-4 within-function relations; see
the paper's table for the search scope.

The specific nearest-miss values reported in the paper are:

    Search space                  Nearest miss
    -------------------------------------------------------
    Within L(s,chi_5): order 4    0.00037
    Cross-function: order 2       |gamma_5 - gamma'_12| = 0.065
    Cross-function: order 3       |gamma_6 + gamma'_16 - gamma_20| = 0.0019

RIGOUR ARCHITECTURE
-------------------
All arithmetic is ARB interval arithmetic throughout.  No Python floats
are used in any load-bearing computation.

The exhaustive search operates directly on ARB values.  For each
combination, the certified ARB predicate

    bool(abs(combination) > ARB_FLOOR)

is evaluated.  This predicate returns True only when the entire ARB
interval for abs(combination) lies strictly above ARB_FLOOR.  A True
result is therefore a rigorous certificate that the combination is
nonzero (in fact, bounded away from zero by ARB_FLOOR).

ARB_FLOOR = arb("1e-10") is chosen far below the smallest observed
nearest miss (~3.7e-4) and far above the ARB ball radii (~1e-74),
so every certified combination clears it by a factor of at least 3700.

The PASS/FAIL decision for each combination uses only the ARB predicate;
no float midpoint extraction or float comparison is used.

Zero data:
    First 20 zeros of L(s, chi_5): from L_function_zeros.py,
        Zenodo doi:10.5281/zenodo.18783098, 70 decimal places,
        certified |L(1/2 + i*gamma'_k, chi_5)| < 10^{-449}.
    First 20 zeros of zeta(s): from zeta_zeros.py, LMFDB data,
        31 decimal places.

Working precision: 256-bit ARB throughout.
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
ARB_PREC  = 256
N_ZEROS   = 20

# Certified floor: every combination must clear this bound.
# Chosen well below the nearest miss (~3.7e-4) and well above
# ARB ball radii (~1e-74).
ARB_FLOOR = arb("1e-10")

# ── Load zeros as ARB ─────────────────────────────────────────────────────────

def load_zeros_arb():
    """Return ARB lists of first N_ZEROS chi_5 and zeta zeros."""
    ctx.prec = ARB_PREC
    chi5_str = get_L_zeros(5, N_ZEROS, as_strings=True)
    zeta_str = _zz.get_zeros(N_ZEROS, as_strings=True)
    g  = [arb(z) for z in chi5_str]
    gz = [arb(z) for z in zeta_str]
    return g, gz


# ── ARB search routines ───────────────────────────────────────────────────────

def search_within_order4(g):
    """
    Certify that all combinations gamma'_a + gamma'_b - gamma'_c - gamma'_d
    (a<=b, c<=d, (a,b)!=(c,d)) are nonzero via ARB predicate.

    Returns (min_arb_val, (a,b,c,d), n_combinations, n_certified).
    """
    ctx.prec = ARB_PREC
    best_mid = None
    best_val = None
    best_idx = None
    n_total  = 0
    n_cert   = 0

    for a in range(N_ZEROS):
        for b in range(a, N_ZEROS):
            for c in range(N_ZEROS):
                for d in range(c, N_ZEROS):
                    if (a, b) == (c, d):
                        continue
                    n_total += 1
                    val = abs(g[a] + g[b] - g[c] - g[d])
                    if bool(val > ARB_FLOOR):
                        n_cert += 1
                    else:
                        raise RuntimeError(
                            f"ARB predicate FAILED: gamma'_{a+1} + gamma'_{b+1} "
                            f"- gamma'_{c+1} - gamma'_{d+1} not certified > {ARB_FLOOR}\n"
                            f"  ARB value: {val}"
                        )
                    mid = float(val.mid())
                    if best_mid is None or mid < best_mid:
                        best_mid = mid
                        best_val = val
                        best_idx = (a+1, b+1, c+1, d+1)

    return best_val, best_idx, n_total, n_cert


def search_cross_order2(g, gz):
    """
    Certify that all |gamma_i - gamma'_j| are nonzero via ARB predicate.
    (gamma_i + gamma'_j > 14 for all i,j; only differences can be near zero.)

    Returns (min_arb_val, (i,j), n_combinations, n_certified).
    """
    ctx.prec = ARB_PREC
    best_mid = None
    best_val = None
    best_idx = None
    n_total  = 0
    n_cert   = 0

    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            n_total += 1
            val = abs(gz[i] - g[j])
            if bool(val > ARB_FLOOR):
                n_cert += 1
            else:
                raise RuntimeError(
                    f"ARB predicate FAILED: |gamma_{i+1} - gamma'_{j+1}| "
                    f"not certified > {ARB_FLOOR}\n  ARB value: {val}"
                )
            mid = float(val.mid())
            if best_mid is None or mid < best_mid:
                best_mid = mid
                best_val = val
                best_idx = (i+1, j+1)

    return best_val, best_idx, n_total, n_cert


def search_cross_order3(g, gz):
    """
    Certify that all 3-term cross-function combinations are nonzero via ARB.

    Sign patterns searched (z = zeta, L = chi_5):
      (2z, 1L):  gamma_i + gamma_j - gamma'_k   (i <= j)
                 gamma_i - gamma_j - gamma'_k   (i != j)
                 gamma_i - gamma_j + gamma'_k   (i != j)
      (1z, 2L):  gamma_i + gamma'_j - gamma'_k  (j < k)
                 gamma_i - gamma'_j + gamma'_k  (j < k)
                -gamma_i + gamma'_j + gamma'_k  (j < k)

    Returns (min_arb_val, description, n_combinations, n_certified).
    """
    ctx.prec = ARB_PREC
    best_mid  = None
    best_val  = None
    best_desc = ""
    n_total   = 0
    n_cert    = 0

    def check(val, desc):
        nonlocal best_mid, best_val, best_desc, n_total, n_cert
        n_total += 1
        av = abs(val)
        if bool(av > ARB_FLOOR):
            n_cert += 1
        else:
            raise RuntimeError(
                f"ARB predicate FAILED: {desc} not certified > {ARB_FLOOR}\n"
                f"  ARB value: {av}"
            )
        mid = float(av.mid())
        if best_mid is None or mid < best_mid:
            best_mid  = mid
            best_val  = av
            best_desc = desc

    # (2z, 1L): gamma_i + gamma_j - gamma'_k  (i <= j)
    for i in range(N_ZEROS):
        for j in range(i, N_ZEROS):
            for k in range(N_ZEROS):
                check(gz[i] + gz[j] - g[k],
                      f"gamma_{i+1} + gamma_{j+1} - gamma'_{k+1}")

    # (2z, 1L): gamma_i - gamma_j - gamma'_k  (i != j)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            if i == j:
                continue
            for k in range(N_ZEROS):
                check(gz[i] - gz[j] - g[k],
                      f"gamma_{i+1} - gamma_{j+1} - gamma'_{k+1}")

    # (2z, 1L): gamma_i - gamma_j + gamma'_k  (i != j)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            if i == j:
                continue
            for k in range(N_ZEROS):
                check(gz[i] - gz[j] + g[k],
                      f"gamma_{i+1} - gamma_{j+1} + gamma'_{k+1}")

    # (1z, 2L): gamma_i + gamma'_j - gamma'_k  (j < k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(j + 1, N_ZEROS):
                check(gz[i] + g[j] - g[k],
                      f"gamma_{i+1} + gamma'_{j+1} - gamma'_{k+1}")

    # (1z, 2L): gamma_i - gamma'_j + gamma'_k  (j < k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(j + 1, N_ZEROS):
                check(gz[i] - g[j] + g[k],
                      f"gamma_{i+1} - gamma'_{j+1} + gamma'_{k+1}")

    # (1z, 2L): -gamma_i + gamma'_j + gamma'_k  (j < k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(j + 1, N_ZEROS):
                check(-gz[i] + g[j] + g[k],
                      f"-gamma_{i+1} + gamma'_{j+1} + gamma'_{k+1}")

    return best_val, best_desc, n_total, n_cert


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ctx.prec = ARB_PREC
    t0 = time.time()

    print()
    print("Proposition 9.1 -- Verified Asymptotic Independence -- ARB Certification")
    print("=" * 74)
    print(f"  Certifying absence of integer relations among first {N_ZEROS} zeros")
    print(f"  of zeta(s) and L(s, chi_5).")
    print(f"  All arithmetic: {ARB_PREC}-bit ARB throughout.")
    print(f"  Certified floor: ARB_FLOOR = 1e-10")
    print(f"  PASS criterion: bool(abs(combination) > ARB_FLOOR) == True for all.")
    print()

    g, gz = load_zeros_arb()

    # ── Search 1: within chi_5, order 4 ──────────────────────────────────────
    print("  [1/3] Within L(s,chi_5), order 4 ...")
    t1 = time.time()
    w4_val, w4_idx, w4_n, w4_c = search_within_order4(g)
    a, b, c, d = w4_idx
    print(f"        Done ({time.time()-t1:.1f}s)  --  {w4_n} combinations, {w4_c} certified")
    print(f"        Nearest miss (ARB): {w4_val}")
    print(f"        Combination:  gamma'_{a} + gamma'_{b} - gamma'_{c} - gamma'_{d}")
    paper_w4 = arb("3.7e-4")
    rel_w4   = abs(w4_val - paper_w4) / paper_w4
    match_w4 = bool(rel_w4 < arb("0.01"))
    print(f"        Paper value:  0.00037")
    print(f"        ARB rel err vs paper: {rel_w4}  -->  {'PASS' if match_w4 else 'FAIL'}")
    print()

    # ── Search 2: cross-function, order 2 ────────────────────────────────────
    print("  [2/3] Cross-function, order 2 ...")
    t1 = time.time()
    c2_val, c2_idx, c2_n, c2_c = search_cross_order2(g, gz)
    zi, lj = c2_idx
    print(f"        Done ({time.time()-t1:.1f}s)  --  {c2_n} combinations, {c2_c} certified")
    print(f"        Nearest miss (ARB): {c2_val}")
    print(f"        Combination:  |gamma_{zi} - gamma'_{lj}|")
    paper_c2 = arb("6.5e-2")
    rel_c2   = abs(c2_val - paper_c2) / paper_c2
    match_c2 = bool(rel_c2 < arb("0.02"))
    print(f"        Paper value:  0.065")
    print(f"        ARB rel err vs paper: {rel_c2}  -->  {'PASS' if match_c2 else 'FAIL'}")
    print()

    # ── Search 3: cross-function, order 3 ────────────────────────────────────
    print("  [3/3] Cross-function, order 3 ...")
    t1 = time.time()
    c3_val, c3_desc, c3_n, c3_c = search_cross_order3(g, gz)
    print(f"        Done ({time.time()-t1:.1f}s)  --  {c3_n} combinations, {c3_c} certified")
    print(f"        Nearest miss (ARB): {c3_val}")
    print(f"        Combination:  {c3_desc}")
    paper_c3 = arb("1.9e-3")
    rel_c3   = abs(c3_val - paper_c3) / paper_c3
    match_c3 = bool(rel_c3 < arb("0.01"))
    print(f"        Paper value:  0.0019")
    print(f"        ARB rel err vs paper: {rel_c3}  -->  {'PASS' if match_c3 else 'FAIL'}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_combos = w4_n + c2_n + c3_n
    total_cert   = w4_c + c2_c + c3_c
    all_pass = match_w4 and match_c2 and match_c3 and (total_combos == total_cert)

    print(f"  Summary:")
    print(f"    Total combinations searched: {total_combos}")
    print(f"    Certified nonzero (ARB):     {total_cert}")
    print(f"    Total time: {time.time()-t0:.1f}s")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    Every combination certified nonzero via ARB predicate")
        print("    bool(abs(combination) > 1e-10) == True.")
        print("    All paper nearest-miss values confirmed in ARB arithmetic.")
        print("    Conditions of Proposition 9.1 satisfied at M_1 = M_2 = 20")
        print("    for the search orders stated in the paper's table.")
    else:
        print("  RESULT: FAIL")
        raise RuntimeError("Proposition 9.1 ARB certification failed.")
    print()


if __name__ == "__main__":
    main()
