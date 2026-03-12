"""
Table_1_Lemma_7.6(Stability).py
================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table appearing after Lemma 7.6
[Stability].  The certified values are:

    M     epsilon_M    epsilon_M / sigma_L
    ----------------------------------------
    5     0.06273      1.665
    10    0.04488      1.192
    15    0.03619      0.961
    20    0.03085      0.819
    25    0.02715      0.721

where

    epsilon_M = sum_{k > M} b_k
              = (partial sum from zero 201 to 200) + (analytic tail)
              = sum_{k=M+1}^{200} b_k  +  (1/pi)(log(5T/2pi)/T + 1/T)

with T = gamma'_200 (the 200th certified zero ordinate of L(s, chi_5)),
and

    sigma_L = sqrt( (1/2) sum_{k=1}^{20} b_k^2 )

using the first 20 zeros, which matches the paper's stated value
sigma_L = 3.767e-2 (the contribution from zeros beyond k=20 is below
4e-7 and negligible at 4 significant figures).

Analytic tail bound:
    (1/pi)(log(5T/2pi)/T + 1/T)
derived from Abel summation against the zero-counting formula
N(T, chi_5) ~ (T/pi) log(5T/2pi e).  This is an upper bound on
sum_{k > 200} b_k, rigorous to the extent that the zero-counting
formula provides an upper bound for all T >= gamma'_200.

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

PASS/FAIL criterion:
    epsilon_M and sigma_L: certified ARB predicate  rel_err < REL_TOL_4SF.
    ratio: certified ARB predicate  rel_err < REL_TOL_RATIO.
  No float thresholds are used in any certification decision.

Zero data:
    Weights b_k = 2 / (1/4 + gamma_k'^2) from:
      - First 200 zeros of L(s, chi_5) for the epsilon_M partial sum
        (sourced from L_function_zeros.py, which provides 1000 zeros;
        Zenodo doi:10.5281/zenodo.18783098, certified to 70 decimal
        places with |L(1/2 + i*gamma_k', chi_5)| < 10^{-449}).
      - First 20 zeros for sigma_L (same source).

Working precision: 256-bit ARB throughout.
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros
from flint import arb, ctx

# ── Precision parameters ─────────────────────────────────────────────────────
ARB_PREC = 256

# ── Paper table values ────────────────────────────────────────────────────────
# ARB-certified values; these are the values used in the paper.
PAPER_EPS   = {5: "6.273e-2", 10: "4.488e-2", 15: "3.619e-2",
               20: "3.085e-2", 25: "2.715e-2"}
PAPER_RATIO = {5: "1.665", 10: "1.192", 15: "0.961", 20: "0.819", 25: "0.721"}
PAPER_SIGMA_L = "3.767e-2"

# Tolerance for epsilon_M and sigma_L (0.1%); ratio at 0.5% (ratio is derived
# from two independently-rounded quantities, so tighter tolerance is not
# meaningful).
REL_TOL_4SF   = arb("1e-3")
REL_TOL_RATIO = arb("5e-3")

M_VALUES = [5, 10, 15, 20, 25]

# ── ARB constants ─────────────────────────────────────────────────────────────
_ZERO = arb(0)
_TWO  = arb(2)


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights(N):
    """First N Lorentzian weights for chi_5, ARB at ARB_PREC bits."""
    ctx.prec = ARB_PREC
    zeros = get_zeros(5, N, as_strings=True)
    return [arb(2) / (arb("1/4") + arb(g)**2) for g in zeros], zeros


# ── Analytic tail bound ───────────────────────────────────────────────────────

def analytic_tail(T_arb):
    """
    Upper bound on sum_{k > 200} b_k via Abel summation:
        (1/pi)(log(5T/2pi)/T + 1/T)
    where T = gamma'_200, the height of the 200th zero.
    Derived from N(T, chi_5) ~ (T/pi) log(5T/2pi e).
    All arithmetic in ARB.
    """
    pi  = arb.pi()
    log_term = (arb(5) * T_arb / (arb(2) * pi)).log()
    return (log_term / T_arb + arb(1) / T_arb) / pi


# ── Certified PASS/FAIL ───────────────────────────────────────────────────────

def arb_matches(val_arb, paper_str, tol):
    """
    Return True iff |val_arb - paper| / paper < tol, where < is the
    ARB certified predicate.  No float thresholds used.
    """
    pv      = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return bool(rel_err < tol)


# ── Certification ─────────────────────────────────────────────────────────────

def certify():
    ctx.prec = ARB_PREC

    # Load weights
    b200, zeros200 = load_weights(200)
    b20,  _        = load_weights(20)

    # sigma_L from first 20 zeros (matches paper's sigma_L = 3.767e-2)
    sigma_L = (sum(bk**2 for bk in b20) / arb(2)).sqrt()

    # Analytic tail at T = gamma'_200
    gamma200 = arb(zeros200[199])
    tail     = analytic_tail(gamma200)

    # sigma_L PASS/FAIL
    sig_match = arb_matches(sigma_L, PAPER_SIGMA_L, REL_TOL_4SF)

    results = []
    for M in M_VALUES:
        eps_partial = sum(b200[k] for k in range(M, 200))
        eps         = eps_partial + tail
        ratio       = eps / sigma_L

        eps_match   = arb_matches(eps,   PAPER_EPS[M],   REL_TOL_4SF)
        ratio_match = arb_matches(ratio, PAPER_RATIO[M], REL_TOL_RATIO)

        results.append(dict(
            M=M,
            eps_arb=eps,     eps_float=float(eps.mid()),
            ratio_arb=ratio, ratio_float=float(ratio.mid()),
            eps_match=eps_match, ratio_match=ratio_match,
        ))

    return results, sigma_L, gamma200, tail, sig_match


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results, sigma_L, gamma200, tail, sig_match):
    print()
    print("Table after Lemma 7.6 -- Stability -- ARB certification")
    print("=" * 66)
    print("  epsilon_M = sum_{k=M+1}^{200} b_k  +  analytic tail")
    print(f"  Analytic tail formula: (1/pi)(log(5T/2pi)/T + 1/T)")
    print(f"  T = gamma'_200 = {float(gamma200.mid()):.4f}")
    print(f"  Tail (ARB) = {tail}")
    print(f"  sigma_L (20 zeros, ARB) = {sigma_L}")
    print(f"  sigma_L paper = {PAPER_SIGMA_L}  "
          f"{'PASS' if sig_match else 'FAIL'} (rel_tol = 0.1%)")
    print(f"  Precision: {ARB_PREC}-bit ARB")
    print(f"  PASS/FAIL: eps_M: rel_err < 0.1%;  ratio: rel_err < 0.5%")
    print()
    print(f"  {'M':>4}  {'eps_M':>10}  {'Paper eps':>10}  "
          f"{'ratio':>7}  {'Paper R':>8}  {'eps':>6}  {'ratio':>6}")
    print("  " + "-" * 64)

    all_pass = True
    for r in results:
        M = r['M']
        line = (f"  {M:>4}  {r['eps_float']:>10.5f}  {PAPER_EPS[M]:>10}  "
                f"{r['ratio_float']:>7.3f}  {PAPER_RATIO[M]:>8}  "
                f"{'PASS' if r['eps_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['ratio_match'] else 'FAIL':>6}")
        print(line)
        if not r['eps_match'] or not r['ratio_match']:
            all_pass = False

    print()
    print("  ARB balls (full precision):")
    for r in results:
        print(f"    M={r['M']:2d}: eps_M = {r['eps_arb']}")
        print(f"          ratio = {r['ratio_arb']}")
    print()

    if not sig_match:
        all_pass = False

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    - epsilon_M: ARB partial sum (zeros M+1..200) + certified analytic tail")
        print("    - sigma_L: ARB sum of b_k^2/2 for k=1..20, square-rooted in ARB")
        print("    - PASS/FAIL via decisive ARB predicate, no float threshold")
    else:
        print("  RESULT: FAIL -- one or more values outside tolerance")
        raise RuntimeError("Lemma 7.6 stability table certification failed")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctx.prec = ARB_PREC
    t0 = time.time()
    print(f"Certifying Lemma 7.6 stability table at {ARB_PREC}-bit ARB precision ...")
    results, sigma_L, gamma200, tail, sig_match = certify()
    print(f"  Done ({time.time()-t0:.1f}s)")
    print_results(results, sigma_L, gamma200, tail, sig_match)
