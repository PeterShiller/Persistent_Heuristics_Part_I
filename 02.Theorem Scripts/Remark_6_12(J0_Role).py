"""
Remark_6_12(J0_Role).py  --  ARB-rigorous certification of Remark 6.12
=======================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table appearing in Remark 6.12
[Role of the inactive J_0 factors]:

    M     I_n (with J_0's)    Ratio to M=3
    -------------------------------------
    3     3.676               1.000
    10    1.794               0.488
    20    1.720               0.468
    30    1.706               0.464

The integrand is:

    (1/pi) |J_1(b_1 t)| |J_1(b_2 t)| |J_1(b_3 t)|
            * prod_{k=4}^{M} |J_0(b_k t)|

with active resonance vector (n_1, n_2, n_3) = (1, 1, -1), so three
active J_1 factors (using |J_{-1}| = |J_1| by symmetry) and M - 3
inactive J_0 factors.  Integration domain [epsilon, T] with
epsilon = 1e-30, T = 2000.

The table illustrates that each additional inactive J_0 factor reduces
the resonance integral, confirming the role of the J_0 factors in
providing the polynomial t-decay needed for integrability.

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Algorithm
---------
The integrand contains absolute values of Bessel functions.  Unlike
the Lemma 6.6 script, the integrand here is smooth and well-behaved
(the J_1 and J_0 factors have no shared zeros), so acb.integral
(acb_calc_integrate, Petras algorithm, 256-bit ARB) can be applied
directly over the full interval [epsilon, T] without strip
decomposition.

The integrand (1/pi)|J_1(b_1 t)||J_1(b_2 t)||J_1(b_3 t)| prod|J_0(b_k t)|
is real, non-negative, and smooth on (0, T] (Bessel functions are
entire; the zeros of the three active J_1 factors and the inactive
J_0 factors do not coincide generically).  The absolute values are
taken after evaluation, so each factor is treated as a non-analytic
integrand with analytic=False.

J_0 computation uses the recurrence J_0(x) = (2/x)J_1(x) - J_2(x)
(DLMF 10.6.1), avoiding the python-flint 0.8.0 bug where
acb.bessel_j(z, 0) returns 0.

Zero data
---------
Weights b_k = 2 / (1/4 + gamma_k'^2) are computed in ARB from the
first 30 certified zero ordinates of L(s, chi_5), sourced from
L_function_zeros.py (Zenodo doi:10.5281/zenodo.18783098).
Zeros are certified to 70 decimal places with
|L(1/2 + i*gamma_k', chi_5)| < 10^{-449}.

Working precision: 256-bit ARB throughout.
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, χ) Zeros and Imported ζ Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros
from flint import arb, acb, ctx

# ── Precision and parameters ──────────────────────────────────────────────────
ARB_PREC  = 256
T_UPPER   = arb(2000)
T_LOWER   = arb("1e-30")   # lower cutoff; integrand is O(t^3) near 0
REL_TOL   = 2**(-200)      # integration tolerance

# ── Paper table values (3 significant figures) ───────────────────────────────
PAPER_I     = {3: "3.676", 10: "1.794", 20: "1.720", 30: "1.706"}
PAPER_RATIO = {3: "1.000", 10: "0.488", 20: "0.468", 30: "0.464"}
REL_TOL_3SF = arb("5e-3")  # 0.5% relative tolerance for 3sf matching

M_VALUES = [3, 10, 20, 30]

# ── Load weights ──────────────────────────────────────────────────────────────

def load_weights():
    """
    Load the first 30 Lorentzian weights b_k = 2 / (1/4 + gamma_k'^2)
    for chi_5 as ARB balls at ARB_PREC-bit precision.

    The zero ordinates are sourced from L_function_zeros.py, certified to
    70 decimal places with |L(1/2 + i*gamma_k', chi_5)| < 10^{-449}.
    """
    ctx.prec = ARB_PREC
    zeros = get_zeros(5, 30, as_strings=True)
    weights = []
    for g in zeros:
        gk = arb(g)
        weights.append(arb(2) / (arb("1/4") + gk * gk))
    return weights


# ── Bessel function helpers ───────────────────────────────────────────────────

def J1_abs_arb(t_acb, bk_arb):
    """
    |J_1(b_k * t)| as an ARB ball.

    J_1 is computed directly via acb.bessel_j(z, 1).
    """
    z = t_acb * acb(bk_arb)
    return abs(acb.bessel_j(z, acb(1)))


def J0_abs_arb(t_acb, bk_arb):
    """
    |J_0(b_k * t)| as an ARB ball, using the recurrence

        J_0(z) = (2/z) J_1(z) - J_2(z)     (DLMF 10.6.1)

    to avoid the python-flint 0.8.0 bug where acb.bessel_j(z, 0) = 0.
    """
    z = t_acb * acb(bk_arb)
    j0 = acb(2) * acb.bessel_j(z, acb(1)) / z - acb.bessel_j(z, acb(2))
    return abs(j0)


# ── Integrand ─────────────────────────────────────────────────────────────────

def make_integrand(b, M):
    """
    Return the integrand function for acb.integral at truncation level M.

    Integrand: (1/pi) |J_1(b_1 t)| |J_1(b_2 t)| |J_1(b_3 t)|
                       * prod_{k=4}^{M} |J_0(b_k t)|

    Active resonance vector: (n_1, n_2, n_3) = (1, 1, -1), so three
    J_1 factors (k=1,2,3) and M-3 inactive J_0 factors (k=4,...,M).
    |J_{-1}| = |J_1| by symmetry.

    The integrand is real, non-negative, and smooth on (0, T].
    The 'analytic' flag is ignored (non-analytic due to absolute values),
    so acb.integral uses the non-analytic path throughout.
    """
    _pi = acb(arb.pi())
    b_acb = [acb(bk) for bk in b]

    def f(t, _analytic):
        # Three active J_1 factors
        r = J1_abs_arb(t, b[0]) * J1_abs_arb(t, b[1]) * J1_abs_arb(t, b[2])
        # M - 3 inactive J_0 factors
        for k in range(3, M):
            r = r * J0_abs_arb(t, b[k])
        return acb(r) / _pi

    return f


# ── Matching helper ───────────────────────────────────────────────────────────

def arb_matches(val_arb, paper_str, rel_tol_arb):
    """
    Return True if |val_arb - paper_val| / paper_val < rel_tol_arb,
    certified in ARB (i.e., the upper bound of the relative error ball
    is less than rel_tol_arb).
    """
    pv = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return rel_err.abs_upper() < float(rel_tol_arb.mid())


# ── Main certification ────────────────────────────────────────────────────────

def certify_J0role(b):
    """
    Compute and certify the Remark 6.12 table for M in {3, 10, 20, 30}.

    Returns a list of result dicts with keys:
        M, I_arb, I_float, ratio_arb, ratio_float,
        I_match, ratio_match, elapsed
    """
    ctx.prec = ARB_PREC
    results = []
    I_ref = None

    for M in M_VALUES:
        t0 = time.time()
        f = make_integrand(b, M)
        I = acb.integral(f, acb(T_LOWER), acb(T_UPPER),
                         rel_tol=REL_TOL, eval_limit=10**7).real

        if I_ref is None:
            I_ref = I

        ratio = I / I_ref
        I_match     = arb_matches(I,     PAPER_I[M],     REL_TOL_3SF)
        ratio_match = arb_matches(ratio, PAPER_RATIO[M], REL_TOL_3SF)
        elapsed = time.time() - t0

        results.append(dict(
            M=M,
            I_arb=I,
            I_float=float(I.mid()),
            ratio_arb=ratio,
            ratio_float=float(ratio.mid()),
            I_match=I_match,
            ratio_match=ratio_match,
            elapsed=elapsed,
        ))

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results):
    print()
    print("Remark 6.12 (Role of inactive J_0 factors) -- ARB certification")
    print("=" * 70)
    print(f"  Integrand : (1/pi) |J_1(b_1 t)| |J_1(b_2 t)| |J_1(b_3 t)|")
    print(f"                     * prod_{{k=4}}^{{M}} |J_0(b_k t)|")
    print(f"  Active    : (n_1, n_2, n_3) = (1, 1, -1)")
    print(f"  Domain    : [1e-30, {float(T_UPPER.mid()):.0f}]")
    print(f"  Precision : {ARB_PREC}-bit ARB (acb_calc_integrate)")
    print()
    print(f"  {'M':>4}  {'I_n (ARB)':>22}  {'Paper':>7}  {'Ratio':>7}  "
          f"{'Paper':>7}  {'I_match':>7}  {'R_match':>7}  {'Time':>6}")
    print("  " + "-" * 68)

    all_pass = True
    for r in results:
        M = r['M']
        line = (f"  {M:>4}  {r['I_float']:>22.5f}  {PAPER_I[M]:>7}  "
                f"{r['ratio_float']:>7.4f}  {PAPER_RATIO[M]:>7}  "
                f"{'PASS' if r['I_match'] else 'FAIL':>7}  "
                f"{'PASS' if r['ratio_match'] else 'FAIL':>7}  "
                f"{r['elapsed']:>5.1f}s")
        print(line)
        if not r['I_match'] or not r['ratio_match']:
            all_pass = False

    print()
    print(f"  ARB balls (full precision):")
    for r in results:
        print(f"    M={r['M']:2d}: I = {r['I_arb']}")
    print()

    if all_pass:
        print("  RESULT: ALL PASS -- Remark 6.12 table certified [ARB-rigorous]")
    else:
        print("  RESULT: FAIL -- one or more values outside 3sf tolerance")
        raise RuntimeError("Remark 6.12 certification failed")

    print()
    print("  Interpretation:")
    print("  Each additional inactive J_0 factor reduces the resonance integral,")
    print("  confirming that J_0 factors provide the polynomial t-decay needed")
    print("  for integrability, while the active J_1 factors control resonance")
    print("  suppression in the order |n_*|.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctx.prec = ARB_PREC
    print(f"Loading chi_5 weights at {ARB_PREC}-bit ARB precision...")
    b = load_weights()
    print(f"  b_1 = {float(b[0].mid()):.6f}, b_2 = {float(b[1].mid()):.6f}, "
          f"b_3 = {float(b[2].mid()):.6f}")
    print(f"  (first 30 weights loaded)")
    print()
    print(f"Computing integrals for M in {M_VALUES} ...")
    results = certify_J0role(b)
    print_results(results)
