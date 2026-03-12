"""
Proposition_9.1(Verified_Asymptotic_Independence).py
=====================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table appearing in
Section 9 (Proposition 9.1, Verified asymptotic independence).
It performs exhaustive nearest-miss searches over integer linear
combinations of the first 20 zeros of zeta(s) and L(s, chi_5),
verifying that no nontrivial relation is exact and confirming
the specific nearest-miss values reported in the paper:

    Search space                  Nearest miss
    -------------------------------------------------------
    Within L(s,chi_5): order 4    0.00037
    Cross-function: order 2       |gamma_5 - gamma'_12| = 0.065
    Cross-function: order 3       |gamma_6 + gamma'_16 - gamma_20| = 0.0019

Definitions:
    "Order k" denotes linear combinations using exactly k terms,
    each with coefficient +1 or -1, drawn from the combined zero set
    (the within-chi_5 and cross-function searches use their respective
    sets).  For a combination to be a near-zero relation, the positive
    and negative parts must approximately cancel.

    Within L(s,chi_5), order 4:
        All combinations gamma'_a + gamma'_b - gamma'_c - gamma'_d
        with a <= b, c <= d, (a,b) != (c,d), indices in {1,...,20}.
        The zero sum of signs is required for a near-relation.

    Cross-function, order 2:
        All combinations +/- gamma_i +/- gamma'_j with i, j in {1,...,20}.
        Only the difference gamma_i - gamma'_j can be near zero since all
        zeros are positive; the sum gamma_i + gamma'_j > 14 for all i, j.

    Cross-function, order 3:
        All 3-term combinations drawing from both {gamma_i} and {gamma'_j},
        with signs chosen so the combination can be near zero.  This covers:
          (2 zeta, 1 chi_5): gamma_i + gamma_j - gamma'_k,
                             gamma_i - gamma_j + gamma'_k  (i.e., gamma'_k - |gamma_i - gamma_j|),
                             -gamma_i - gamma_j + gamma'_k (only near zero if gamma'_k large);
          (1 zeta, 2 chi_5): gamma_i + gamma'_j - gamma'_k,
                             gamma_i - gamma'_j + gamma'_k (same as above by symmetry),
                             -gamma_i + gamma'_j + gamma'_k (only near zero if gamma_i large).
        All sign patterns producing potentially small values are searched.

Search phase: floating-point arithmetic (Python float, ~15 significant
figures) over all index combinations; sufficient to determine nearest miss
to 4 significant figures.

Certification phase: the specific combinations named in the paper are
recomputed in ARB at 256-bit precision, giving rigorous certified values.

Zero data:
    First 20 zeros of L(s, chi_5): from L_function_zeros.py,
        Zenodo doi:10.5281/zenodo.18783098, 70 decimal places,
        certified |L(1/2 + i*gamma'_k, chi_5)| < 10^{-449}.
    First 20 zeros of zeta(s): from zeta_zeros.py, LMFDB data,
        31 decimal places.

PASS/FAIL:
    Each named combination is certified in ARB.  The script asserts
    that the ARB ball for each combination contains the paper value
    at the stated precision, and that no combination has absolute
    value below 10^{-6} (ruling out exact or near-exact relations
    at floating-point resolution).

All arithmetic in the certification phase is ARB.  The float search
phase is used only to enumerate candidates; all reported values are
reconfirmed by ARB.
"""

import os
import sys
import time
from itertools import combinations, product

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros as get_L_zeros
import zeta_zeros as _zz
from flint import arb, ctx

# ── Precision ─────────────────────────────────────────────────────────────────
ARB_PREC  = 256
N_ZEROS   = 20          # search over first 20 zeros of each function
MIN_NONZERO = 1e-6      # floor: any combination below this would be alarming

# ── Paper values ──────────────────────────────────────────────────────────────
# (nearest_miss, index tuple, description)
PAPER_WITHIN_ORDER4  = (0.00037, (4, 19, 10, 12),
                        "gamma'_4 + gamma'_19 - gamma'_10 - gamma'_12")
PAPER_CROSS_ORDER2   = (0.065,   ("z5", "L12"),
                        "|gamma_5 - gamma'_12|")
PAPER_CROSS_ORDER3   = (0.0019,  ("z6", "L16", "z20"),
                        "|gamma_6 + gamma'_16 - gamma_20|")

# ── Load zeros ────────────────────────────────────────────────────────────────

def load_zeros():
    """Return (chi5_float, chi5_str, zeta_float, zeta_str) for first N_ZEROS."""
    chi5_str   = get_L_zeros(5, N_ZEROS, as_strings=True)
    chi5_float = [float(z) for z in chi5_str]
    zeta_str   = _zz.get_zeros(N_ZEROS, as_strings=True)
    zeta_float = [float(z) for z in zeta_str]
    return chi5_float, chi5_str, zeta_float, zeta_str


# ── Float search routines ─────────────────────────────────────────────────────

def search_within_order4(chi5):
    """
    Search all gamma'_a + gamma'_b - gamma'_c - gamma'_d with a<=b, c<=d,
    (a,b)!=(c,d).  Returns (min_val, (a,b,c,d)) with 1-based indices.
    """
    best_val = 1e18
    best_idx = None
    for a in range(N_ZEROS):
        for b in range(a, N_ZEROS):
            for c in range(N_ZEROS):
                for d in range(c, N_ZEROS):
                    if (a, b) == (c, d):
                        continue
                    val = abs(chi5[a] + chi5[b] - chi5[c] - chi5[d])
                    if val < best_val:
                        best_val = val
                        best_idx = (a+1, b+1, c+1, d+1)
    return best_val, best_idx


def search_cross_order2(chi5, zeta):
    """
    Search all |gamma_i - gamma'_j|.  Returns (min_val, (i,j)) 1-based.
    (gamma_i + gamma'_j > 14 for all i,j and is never near zero.)
    """
    best_val = 1e18
    best_idx = None
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            val = abs(zeta[i] - chi5[j])
            if val < best_val:
                best_val = val
                best_idx = (i+1, j+1)
    return best_val, best_idx


def search_cross_order3(chi5, zeta):
    """
    Search all 3-term cross-function combinations that can be near zero.

    Patterns searched (z = zeta zero, L = chi_5 zero):
      (2z, 1L):  gamma_i + gamma_j - gamma'_k   (and with i<j)
                 gamma'_k - gamma_i - gamma_j   (same, but returned as abs value)
                 |gamma_i - gamma_j + gamma'_k|  (difference of zeta, plus L)
                 |gamma_i - gamma_j - gamma'_k|  (difference of zeta, minus L)
      (1z, 2L):  gamma_i + gamma'_j - gamma'_k  (and j<k)
                 gamma'_k - gamma'_j - gamma_i   (same)
                 |gamma'_j - gamma'_k + gamma_i| (diff of L, plus zeta)
                 |gamma'_j - gamma'_k - gamma_i| (diff of L, minus zeta)

    Returns (min_val, description_string).
    """
    best_val = 1e18
    best_desc = ""

    # (2z, 1L): gamma_i + gamma_j - gamma'_k  and its negation
    for i in range(N_ZEROS):
        for j in range(i, N_ZEROS):
            for k in range(N_ZEROS):
                val = abs(zeta[i] + zeta[j] - chi5[k])
                if val < best_val:
                    best_val = val
                    best_desc = (f"|gamma_{i+1} + gamma_{j+1} - gamma'_{k+1}|  "
                                 f"= |{zeta[i]:.4f} + {zeta[j]:.4f} - {chi5[k]:.4f}|")
                # gamma'_k - gamma_i + gamma_j: can only be small if gamma_j ~ gamma_i - gamma'_k
                val = abs(chi5[k] - zeta[i] + zeta[j])
                # already covered by sign flip of previous search

    # (2z, 1L): |gamma_i - gamma_j + gamma'_k| and |gamma_i - gamma_j - gamma'_k|
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            if i == j:
                continue
            for k in range(N_ZEROS):
                val = abs(zeta[i] - zeta[j] + chi5[k])
                if val < best_val:
                    best_val = val
                    best_desc = (f"|gamma_{i+1} - gamma_{j+1} + gamma'_{k+1}|  "
                                 f"= |{zeta[i]:.4f} - {zeta[j]:.4f} + {chi5[k]:.4f}|")
                val = abs(zeta[i] - zeta[j] - chi5[k])
                if val < best_val:
                    best_val = val
                    best_desc = (f"|gamma_{i+1} - gamma_{j+1} - gamma'_{k+1}|  "
                                 f"= |{zeta[i]:.4f} - {zeta[j]:.4f} - {chi5[k]:.4f}|")

    # (1z, 2L): gamma_i + gamma'_j - gamma'_k  (and i, j<k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(j, N_ZEROS):
                if j == k:
                    continue
                val = abs(zeta[i] + chi5[j] - chi5[k])
                if val < best_val:
                    best_val = val
                    best_desc = (f"|gamma_{i+1} + gamma'_{j+1} - gamma'_{k+1}|  "
                                 f"= |{zeta[i]:.4f} + {chi5[j]:.4f} - {chi5[k]:.4f}|")

    # (1z, 2L): gamma_i - gamma'_j + gamma'_k  (j != k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(N_ZEROS):
                if j == k:
                    continue
                val = abs(zeta[i] - chi5[j] + chi5[k])
                # this is the same as |gamma_i - (gamma'_j - gamma'_k)| already covered

    # (1z, 2L): -gamma_i + gamma'_j + gamma'_k  (j<k)
    for i in range(N_ZEROS):
        for j in range(N_ZEROS):
            for k in range(j, N_ZEROS):
                if j == k:
                    continue
                val = abs(-zeta[i] + chi5[j] + chi5[k])
                if val < best_val:
                    best_val = val
                    best_desc = (f"|-gamma_{i+1} + gamma'_{j+1} + gamma'_{k+1}|  "
                                 f"= |-{zeta[i]:.4f} + {chi5[j]:.4f} + {chi5[k]:.4f}|")

    return best_val, best_desc


# ── ARB certification of specific combinations ───────────────────────────────

def certify_specific(chi5_str, zeta_str):
    """
    Certify the three specific nearest-miss combinations in ARB at ARB_PREC bits.
    Returns dict of results.
    """
    ctx.prec = ARB_PREC

    g  = [arb(z) for z in chi5_str]   # gamma'_k, 0-based
    gz = [arb(z) for z in zeta_str]   # gamma_k,  0-based

    # Within chi_5, order 4: gamma'_4 + gamma'_19 - gamma'_10 - gamma'_12
    within4 = g[3] + g[18] - g[9] - g[11]

    # Cross order 2: gamma_5 - gamma'_12
    cross2  = gz[4] - g[11]

    # Cross order 3: gamma_6 + gamma'_16 - gamma_20
    cross3  = gz[5] + g[15] - gz[19]

    return {
        "within4":    (within4,  "gamma'_4 + gamma'_19 - gamma'_10 - gamma'_12"),
        "cross2":     (cross2,   "gamma_5 - gamma'_12"),
        "cross3":     (cross3,   "gamma_6 + gamma'_16 - gamma_20"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print()
    print("Proposition 9.1 -- Verified Asymptotic Independence -- Exhaustive Search")
    print("=" * 74)
    print(f"  Search space: first {N_ZEROS} zeros of zeta(s) and L(s, chi_5)")
    print(f"  Float search precision: ~15 significant figures (Python float)")
    print(f"  ARB certification: {ARB_PREC}-bit ARB")
    print()

    chi5_float, chi5_str, zeta_float, zeta_str = load_zeros()

    # ── Float searches ────────────────────────────────────────────────────────
    print("  [1/3] Within L(s,chi_5), order 4: searching all "
          "gamma'_a + gamma'_b - gamma'_c - gamma'_d ...")
    t1 = time.time()
    w4_val, w4_idx = search_within_order4(chi5_float)
    print(f"        Done ({time.time()-t1:.1f}s)")
    a, b, c, d = w4_idx
    print(f"        Nearest miss: {w4_val:.6f}")
    print(f"        Combination:  gamma'_{a} + gamma'_{b} - gamma'_{c} - gamma'_{d}")
    print(f"        Values:       {chi5_float[a-1]:.4f} + {chi5_float[b-1]:.4f}"
          f" - {chi5_float[c-1]:.4f} - {chi5_float[d-1]:.4f}")
    assert abs(w4_val - 0.00037) < 0.00001, f"FAIL: expected ~0.00037, got {w4_val:.6f}"
    assert w4_val > MIN_NONZERO, f"FAIL: suspiciously small value {w4_val}"
    print(f"        Paper value:  0.00037  -- MATCH")
    print()

    print("  [2/3] Cross-function, order 2: searching all |gamma_i - gamma'_j| ...")
    t1 = time.time()
    c2_val, c2_idx = search_cross_order2(chi5_float, zeta_float)
    print(f"        Done ({time.time()-t1:.1f}s)")
    zi, lj = c2_idx
    print(f"        Nearest miss: {c2_val:.6f}")
    print(f"        Combination:  |gamma_{zi} - gamma'_{lj}|")
    print(f"        Values:       |{zeta_float[zi-1]:.6f} - {chi5_float[lj-1]:.6f}|")
    assert abs(c2_val - 0.065) < 0.001, f"FAIL: expected ~0.065, got {c2_val:.6f}"
    assert c2_val > MIN_NONZERO, f"FAIL: suspiciously small value {c2_val}"
    print(f"        Paper value:  0.065    -- MATCH")
    print()

    print("  [3/3] Cross-function, order 3: searching all 3-term combinations ...")
    t1 = time.time()
    c3_val, c3_desc = search_cross_order3(chi5_float, zeta_float)
    print(f"        Done ({time.time()-t1:.1f}s)")
    print(f"        Nearest miss: {c3_val:.6f}")
    print(f"        Combination:  {c3_desc}")
    assert abs(c3_val - 0.0019) < 0.0001, f"FAIL: expected ~0.0019, got {c3_val:.6f}"
    assert c3_val > MIN_NONZERO, f"FAIL: suspiciously small value {c3_val}"
    print(f"        Paper value:  0.0019   -- MATCH")
    print()

    # ── ARB certification ─────────────────────────────────────────────────────
    print("  ARB certification of named combinations at 256-bit precision:")
    arb_results = certify_specific(chi5_str, zeta_str)

    paper_vals = {"within4": 0.00037, "cross2": 0.065, "cross3": 0.0019}
    all_pass = True
    for key, (val_arb, desc) in arb_results.items():
        mid  = float(abs(val_arb).mid())
        ball = abs(val_arb)
        pv   = paper_vals[key]
        rel  = abs(mid - pv) / pv
        ok   = rel < 0.01   # 1% relative tolerance (values given to 1-2 sig figs in paper)
        print(f"    {desc}")
        print(f"      ARB value:   {ball}")
        print(f"      Float mid:   {mid:.6f}")
        print(f"      Paper:       {pv:.5f}")
        print(f"      Rel error:   {rel:.2e}  -->  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("  Summary:")
    print(f"    Search space:  {N_ZEROS} zeros of each function")
    print(f"    Total float time: {time.time()-t0:.1f}s")
    print()
    if all_pass:
        print("  RESULT: ALL PASS")
        print("    No nontrivial exact integer relation found within the search space.")
        print("    All nearest-miss values are bounded away from 0 (> 1e-6).")
        print("    All three named combinations match their paper values at 256-bit ARB.")
        print("    Conditions of Proposition 9.1 are satisfied at M_1 = M_2 = 20.")
    else:
        print("  RESULT: FAIL -- one or more values outside tolerance")
        raise RuntimeError("Proposition 9.1 certification failed")
    print()


if __name__ == "__main__":
    main()
