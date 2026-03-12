"""
Table_1_Theorem_7.4(Telescoping_Convergence).py
================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table after Theorem 7.4
[Telescoping convergence], confirming the b_M^2 scaling of marginal
changes under resonance absence.  The certified values are:

    M     |f^(M)(0) - f^(M-1)(0)|    b_M        b_M^2       Δf/b_M^2
    -------------------------------------------------------------------
    10    6.70e-3                     0.00247    6.1e-6      1100
    15    1.83e-3                     0.00138    1.9e-6       966
    20    7.91e-4                     0.00093    8.6e-7       924

The signed density at the origin is:

    f^(M)(0) = (1/pi) int_0^T prod_{k=1}^M J_0(b_k t) dt

Marginal changes are |f^(M)(0) - f^(M-1)(0)| for M in {10, 15, 20}.
This requires computing f^(M)(0) for M in {9, 10, 14, 15, 19, 20}.

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Algorithm
---------
The integrand prod_{k=1}^M J_0(b_k t) has no absolute values; it is
entire as a function of t, so acb.integral (Petras algorithm) is
rigorous over any finite interval without strip decomposition.
Integration domain: [T_EPS, T_UPPER] = [1e-30, 2000].

PASS/FAIL criterion:
  - |Δf|, b_M, b_M^2: certified ARB predicate  rel_err < REL_TOL_3SF.
  - Δf/b_M^2: integer match (certified ARB ball within 0.5 of paper value).
  No float thresholds are used in any certification decision.

Zero data:
  Weights b_k = 2 / (1/4 + gamma_k'^2) computed in ARB from the first
  20 certified zero ordinates of L(s, chi_5) sourced from
  L_function_zeros.py (Zenodo doi:10.5281/zenodo.18783098).  Zeros
  certified to 70 decimal places with
  |L(1/2 + i*gamma_k', chi_5)| < 10^{-449}.

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
from flint import arb, acb, ctx

# ── Precision and integration parameters ─────────────────────────────────────
ARB_PREC    = 256
T_UPPER     = 2000
T_EPS       = arb("1e-30")
T_MAX_ARB   = arb(T_UPPER)
REL_TOL_INT = 2**(-200)  # acb.integral relative tolerance

# ── Paper table values ────────────────────────────────────────────────────────
# ARB-certified results; these are the values used in the paper.
PAPER_DF    = {10: "6.699e-3", 15: "1.827e-3", 20: "7.910e-4"}
PAPER_B     = {10: "2.468e-3", 15: "1.375e-3", 20: "9.251e-4"}
PAPER_B2    = {10: "6.092e-6", 15: "1.892e-6", 20: "8.559e-7"}
PAPER_RATIO = {10: 1100,       15: 966,         20: 924}

# 0.1% relative tolerance for certified-to-certified matching.
REL_TOL_3SF = arb("1e-3")

# M values for table rows; requires f^(M) and f^(M-1) for each.
M_VALUES    = [10, 15, 20]
M_ALL       = sorted({M for row in M_VALUES for M in (row - 1, row)})

# ── ARB constants ─────────────────────────────────────────────────────────────
_TWO = arb(2)
# No module-level pi constant: arb.pi() called fresh inside each function.

# ── Bessel function ───────────────────────────────────────────────────────────

def J0_acb(z_acb):
    """J_0(z) via recurrence (DLMF 10.6.1)."""
    return (acb(_TWO) * acb.bessel_j(z_acb, acb(1)) / z_acb
            - acb.bessel_j(z_acb, acb(_TWO)))


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights():
    """First 20 Lorentzian weights for chi_5, ARB at ARB_PREC bits."""
    ctx.prec = ARB_PREC
    zeros    = get_zeros(5, 20, as_strings=True)
    return [arb(2) / (arb("1/4") + arb(g)**2) for g in zeros]


# ── Signed integral ───────────────────────────────────────────────────────────

def compute_f0(b_arb, M_int):
    """
    f^(M)(0) = (1/pi) int_{T_EPS}^{T_UPPER} prod_{k=1}^M J_0(b_k t) dt.

    The integrand is entire; acb.integral is rigorous without strip
    decomposition.  Returns an ARB ball.
    """
    ctx.prec = ARB_PREC
    b_acb    = [acb(bk) for bk in b_arb[:M_int]]

    def integrand(t, _):
        r = J0_acb(t * b_acb[0])
        for k in range(1, M_int):
            r = r * J0_acb(t * b_acb[k])
        return r / acb(arb.pi())

    return acb.integral(integrand, acb(T_EPS), acb(T_MAX_ARB),
                        rel_tol=REL_TOL_INT, eval_limit=10**7).real


# ── Certified PASS/FAIL ───────────────────────────────────────────────────────

def arb_matches_3sf(val_arb, paper_str):
    """
    Return True iff |val_arb - paper_val| / paper_val < REL_TOL_3SF,
    where < is the ARB certified predicate.  No float thresholds used.
    """
    pv      = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return bool(rel_err < REL_TOL_3SF)


def arb_matches_ratio(ratio_arb, paper_int):
    """
    Return True iff the ARB ball for ratio_arb is within 0.5 of paper_int,
    certified by ARB <.
    """
    return bool(abs(ratio_arb - arb(str(paper_int))) < arb("0.5"))


# ── Certification loop ────────────────────────────────────────────────────────

def certify(b_arb):
    ctx.prec = ARB_PREC

    # Compute f^(M)(0) for all required M values.
    print(f"  Computing f^(M)(0) for M in {M_ALL} ...")
    f0_cache = {}
    for M in M_ALL:
        t0          = time.time()
        f0_cache[M] = compute_f0(b_arb, M)
        print(f"    M={M:2d}: f^(M)(0) = {float(f0_cache[M].mid()):.8f}  "
              f"({time.time()-t0:.1f}s)")

    print()
    results = []
    for M in M_VALUES:
        bM    = b_arb[M - 1]
        bM2   = bM ** arb(2)
        delta = abs(f0_cache[M] - f0_cache[M - 1])
        ratio = delta / bM2

        df_match    = arb_matches_3sf(delta, PAPER_DF[M])
        b_match     = arb_matches_3sf(bM,    PAPER_B[M])
        b2_match    = arb_matches_3sf(bM2,   PAPER_B2[M])
        ratio_match = arb_matches_ratio(ratio, PAPER_RATIO[M])

        results.append(dict(
            M=M,
            delta_arb=delta,  delta_float=float(delta.mid()),
            bM_arb=bM,        bM_float=float(bM.mid()),
            bM2_arb=bM2,      bM2_float=float(bM2.mid()),
            ratio_arb=ratio,  ratio_float=float(ratio.mid()),
            df_match=df_match, b_match=b_match,
            b2_match=b2_match, ratio_match=ratio_match,
        ))

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results):
    print()
    print("Table after Theorem 7.4 -- Telescoping convergence -- ARB certification")
    print("=" * 74)
    print("  Quantity : f^(M)(0) = (1/pi) int_0^T prod_{k=1}^M J_0(b_k t) dt")
    print(f"  Domain   : [1e-30, {T_UPPER}]")
    print("  Method   : Direct acb.integral (entire integrand; no strip decomp)")
    print(f"  Precision: {ARB_PREC}-bit ARB")
    print(f"  PASS/FAIL: |Δf|, b_M, b_M^2: rel_err < {float(REL_TOL_3SF.mid())}  "
          f"(display threshold; certification uses ARB <)")
    print(f"           : ratio: certified within 0.5 of paper integer")
    print()
    print(f"  {'M':>4}  {'|Δf|':>10}  {'Pap':>8}  {'b_M':>8}  {'Pap':>7}  "
          f"{'b_M^2':>9}  {'Pap':>7}  {'Ratio':>6}  {'Pap':>5}  "
          f"{'df':>6}  {'b':>6}  {'b2':>6}  {'R':>6}")
    print("  " + "-" * 100)

    all_pass = True
    for r in results:
        M = r['M']
        line = (f"  {M:>4}  {r['delta_float']:>10.3e}  {PAPER_DF[M]:>8}  "
                f"{r['bM_float']:>8.5f}  {PAPER_B[M]:>7}  "
                f"{r['bM2_float']:>9.2e}  {PAPER_B2[M]:>7}  "
                f"{r['ratio_float']:>6.0f}  {PAPER_RATIO[M]:>5d}  "
                f"{'PASS' if r['df_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['b_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['b2_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['ratio_match'] else 'FAIL':>6}")
        print(line)
        if not all([r['df_match'], r['b_match'], r['b2_match'], r['ratio_match']]):
            all_pass = False

    print()
    print("  ARB balls (full precision):")
    for r in results:
        print(f"    M={r['M']:2d}: |Δf|  = {r['delta_arb']}")
        print(f"          b_M   = {r['bM_arb']}")
        print(f"          b_M^2 = {r['bM2_arb']}")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    - signed integrals via direct acb.integral (entire integrand)")
        print("    - differences of successive f^(M)(0) values, ARB-certified")
        print("    - PASS/FAIL via decisive ARB predicate, no float threshold")
    else:
        print("  RESULT: FAIL -- one or more values outside tolerance")
        raise RuntimeError("Theorem 7.4 table certification failed")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctx.prec = ARB_PREC
    print(f"Loading chi_5 weights at {ARB_PREC}-bit ARB precision ...")
    b_arb = load_weights()
    print(f"  b_10 = {float(b_arb[9].mid()):.5f}")
    print(f"  b_15 = {float(b_arb[14].mid()):.5f}")
    print(f"  b_20 = {float(b_arb[19].mid()):.5f}")
    print(f"  ({len(b_arb)} weights loaded)")
    print()
    results = certify(b_arb)
    print_results(results)
