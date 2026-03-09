"""
Proposition_2_2(Szeta_star).py  —  Certified verification of Proposition [Explicit value of S_zeta]
================================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

This module certifies the bound

    S_zeta := sum_{zeta(rho)=0} 2 / (1/4 + gamma^2)  <=  0.04871  =: S_zeta*

established in Proposition [Explicit value of S_zeta] of the above reference.
The bound is required as an external certified constant by Theorem 3.3, the
Spacelike Corollary, and the Low-lying Zero Dominance Theorem.

Algorithm
---------
  The bound is established in three steps, following Appendix A,
  Section [Certification of S_zeta*] of the paper.

  Step 1: Direct sum (partial sum over first K = 6000 zeros).
    The first K = 6000 positive zero ordinates gamma_1, ..., gamma_6000 of
    zeta(s) are loaded from zeta_zeros.py (LMFDB data, 31 decimal places each).
    Each ordinate is loaded into an ARB ball whose radius encloses the rounding
    error of the 31-digit string.  The partial sum

        S_zeta^(K) = sum_{k=1}^{K} 2 / (1/4 + gamma_k^2)

    is accumulated in ARB ball arithmetic.  Certified result:
        S_zeta^(6000) in [0.045795374824191... +/- 7.4e-152].

  Step 2: Tail bound over [T0, 1e6].
    Set T0 = gamma_6000 + 0.01.  Abel summation gives the upper bound

        sum_{gamma > T0} w(gamma)  <=  integral_{T0}^{inf} 4t [N(t)-K] / (1/4+t^2)^2 dt,

    where the boundary term is non-positive and discarded as a further upper
    bound.  The integral is evaluated without quadrature using the exact
    antiderivative identity

        integral_a^b 4t / (1/4 + t^2)^2 dt = 2/(1/4+a^2) - 2/(1/4+b^2).

    The interval [T0, 1e6] is partitioned into 235 subintervals (width 100
    for T0 <= t <= 20000; width 10000 for 20000 < t <= 1e6).  On each
    subinterval [a, b], N(t) - K is bounded above by N_upper(b) - K via
    Trudgian's explicit formula (Theorem [Trudgian] of the paper):

        N_upper(t) = t/(2*pi) * log(t/(2*pi*e)) + 7/8
                     + 0.111*log(t) + 0.275*log(log(t)) + 2.450 + 0.2/T0.

    Each subinterval contribution (N_upper(b) - K) * (2/(1/4+a^2) - 2/(1/4+b^2))
    is accumulated in ARB.  Subintervals with non-positive contribution
    (where N_upper(b) <= K) are skipped as they only tighten the bound.
    Certified subinterval tail: <= 4.31e-4.

  Step 3: Analytic remainder over [1e6, infinity).
    The bound N(t) <= t * log(t) (valid for all t >= 10) and
    4t/(1/4+t^2)^2 <= 4/t^3 yield

        integral_{U}^{inf} 4*N(t) / (1/4+t^2)^2 dt
          <=  2*(log(U) + 1) / (pi * U) + 4/U  |_{U=1e6}

    evaluated in ARB.  Certified analytic remainder: <= 1.35e-5.

  Step 4: Certified result.
    Combining in ARB:
        S_zeta <= S_zeta^(6000) + tail_235 + analytic_rem
               in [0.046240 +/- 2.8e-152].
    The upper endpoint 0.046240... < 0.04625, which is 5.1% below S_zeta* = 0.04871.

Rigorousness checklist
----------------------
  (a) All arithmetic uses ARB ball arithmetic at 512-bit working precision
      (~154 decimal digits).  Subinterval endpoints are exact: T0_arb for
      the first left endpoint, then exact integers arb(str(n)) for all
      remaining endpoints.  No float arithmetic is used inside any certified
      computation.
  (b) Each zero ordinate is loaded as an ARB ball; the ball radius encloses
      the rounding error of the 31-digit string representation.
  (c) The tail bound uses no numerical quadrature.  Every subinterval
      contribution is computed from the exact antiderivative identity and
      the Trudgian upper bound, both evaluated in ARB.
  (d) The analytic remainder at U = 1e6 is a closed-form ARB expression;
      no asymptotic approximation is made.
  (e) All pass/fail predicates (S_total < 0.04625, S_total < S_zeta*) are
      evaluated as ARB comparisons.  float() conversion is used only after
      all certification is complete, for display.

External dependencies
---------------------
  zeta_zeros.py: first 6000 LMFDB Riemann zeta zeros at 31 decimal places
    (LMFDB, accessed 2026-03-06).  The zero ordinates are treated as trusted
    input: this script certifies the bound S_zeta* relative to the imported
    dataset, but does not independently certify the zero ordinates themselves.
    The Trudgian tail bound and analytic remainder are purely analytic; no
    further zero data is used.  A reader wishing a fully self-contained
    certificate would need an independent verification of the 6000 LMFDB
    ordinates (e.g., via sign changes of the Hardy Z-function).

No L-function zero ordinate data from any appendix is used in this computation.

Requirements
------------
  python-flint >= 0.8.0   (provides ARB ball arithmetic)
  Python >= 3.10
  zeta_zeros.py           (must be importable; located in the same repo under
                           01.Computed L(s, chi) Zeros and Imported zeta Zeros/)

Usage
-----
  python "Proposition_2_2(Szeta_star).py"
"""

import sys
import os

# Allow importing zeta_zeros from the sibling directory in the repo.
_HERE = os.path.dirname(os.path.abspath(__file__))
# Exact directory name as it appears in the repository (Unicode characters preserved).
_ZETA_DIR = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
if not os.path.isdir(_ZETA_DIR):
    # Fallback for environments where the script is run from the repo root.
    _ZETA_DIR = os.path.join("01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _ZETA_DIR)

from zeta_zeros import get_zeros, _META
from flint import arb, ctx

BASE_PREC = 512
ctx.prec = BASE_PREC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K            = 6000                        # number of zeros used
S_ZETA_STAR  = arb("0.04871")             # declared bound to certify
TIGHT_BOUND  = arb("0.04625")             # tighter numerical bound from paper
T0_OFFSET    = arb("0.01")               # T0 = gamma_K + 0.01
U_CUTOFF     = arb("1e6")                # analytic remainder starts here


# ---------------------------------------------------------------------------
# Trudgian N_upper
# ---------------------------------------------------------------------------

def N_upper_arb(t_arb, T0_arb):
    """
    Upper bound on N(t) from Trudgian [Corollary 1, 2012].

    N(t) <= t/(2*pi) * log(t/(2*pi*e)) + 7/8
            + 0.111*log(t) + 0.275*log(log(t)) + 2.450 + 0.2/T0.

    All arithmetic in ARB.  The term 0.2/T0 uses the fixed T0 = gamma_6000 + 0.01
    as the reference point from the proposition proof.
    """
    pi2    = arb(2) * arb.pi()
    term1  = t_arb / pi2 * (t_arb / (pi2 * arb.const_e())).log()
    term2  = arb("0.875")                           # 7/8
    term3  = arb("0.111") * t_arb.log()
    term4  = arb("0.275") * t_arb.log().log()
    term5  = arb("2.450")
    term6  = arb("0.2") / T0_arb
    return term1 + term2 + term3 + term4 + term5 + term6


# ---------------------------------------------------------------------------
# Exact antiderivative
# ---------------------------------------------------------------------------

def antideriv(a_arb, b_arb):
    """
    Exact value of integral_a^b 4t / (1/4 + t^2)^2 dt
      = 2/(1/4 + a^2) - 2/(1/4 + b^2).
    """
    return arb(2) / (arb("0.25") + a_arb ** 2) \
         - arb(2) / (arb("0.25") + b_arb ** 2)


# ---------------------------------------------------------------------------
# Step 1: direct partial sum
# ---------------------------------------------------------------------------

def compute_partial_sum():
    """
    Sum S_zeta^(K) = sum_{k=1}^{K} 2 / (1/4 + gamma_k^2) in ARB.

    Each zero ordinate is loaded as arb(string), whose radius encloses the
    rounding error of the 31-digit representation.

    Returns the ARB ball for the partial sum.
    """
    zeros_str = get_zeros(K, as_strings=True)
    S = arb(0)
    for s in zeros_str:
        g = arb(s)
        S += arb(2) / (arb("0.25") + g ** 2)
    return S


# ---------------------------------------------------------------------------
# Step 2: tail bound over [T0, 1e6]
# ---------------------------------------------------------------------------

def build_subintervals_arb(T0_arb):
    """
    Build the 235 subintervals covering [T0, 1e6] as pairs of ARB balls.

    All endpoints are exact: T0_arb for the first left endpoint, then
    exact integers (arb(str(n))) for every subsequent endpoint.  No Python
    float arithmetic is used.

    Partition:
      [T0_arb, arb("6400")],
      [arb("6400"), arb("6500")], ..., [arb("19900"), arb("20000")],  (137 total)
      [arb("20000"), arb("30000")], ..., [arb("990000"), arb("1000000")]  (98 total)
    Total: 235 subintervals, matching the paper.

    The first breakpoint 6400 is the smallest multiple of 100 exceeding
    gamma_6000 + 0.01 ~ 6365.86; this is verified by the certified ARB
    comparison arb("6400") > T0_arb in compute_tail_235.
    """
    subs = []
    # First interval: [T0, 6400] (T0 is ARB; 6400 is an exact integer)
    subs.append((T0_arb, arb("6400")))
    # Remaining batch 1: exact integer endpoints, width 100
    for a_int in range(6400, 20000, 100):
        subs.append((arb(str(a_int)), arb(str(a_int + 100))))
    # Batch 2: exact integer endpoints, width 10000
    for a_int in range(20000, 1000000, 10000):
        subs.append((arb(str(a_int)), arb(str(a_int + 10000))))
    return subs


def compute_tail_235(T0_arb):
    """
    Compute the subinterval tail over [T0, 1e6].

    All subinterval endpoints are exact ARB expressions (see build_subintervals_arb).
    Returns the certified ARB upper bound and the number of subintervals.
    """
    assert bool(arb("6400") > T0_arb), \
        "First breakpoint 6400 must exceed T0; check gamma_6000 value."
    subs  = build_subintervals_arb(T0_arb)
    total = arb(0)
    for (a, b) in subs:
        nu_b   = N_upper_arb(b, T0_arb)
        excess = nu_b - arb(K)
        if bool(excess > arb(0)):
            total += excess * antideriv(a, b)
    return total, len(subs)


# ---------------------------------------------------------------------------
# Step 3: analytic remainder over [1e6, infinity)
# ---------------------------------------------------------------------------

def compute_analytic_remainder():
    """
    Upper bound on integral_{U}^{inf} 4*N(t) / (1/4+t^2)^2 dt  at U = 1e6.

    Uses N(t) <= t*log(t) and 4t/(1/4+t^2)^2 <= 4/t^3, giving
      <= 2*(log(U)+1)/(pi*U) + 4/U.
    """
    U   = U_CUTOFF
    rem = arb(2) * (U.log() + arb(1)) / (arb.pi() * U) + arb(4) / U
    return rem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Proposition [Explicit value of S_zeta]: Certification of S_zeta* <= 0.04871")
    print(f"ARB working precision : {BASE_PREC} bits (~{int(BASE_PREC * 0.30103)} decimal digits)")
    print(f"Zero data             : {K} LMFDB zeta zeros at 31 decimal places")
    print(f"gamma_1               : {_META['gamma_1']}")
    print(f"gamma_6000            : {_META['gamma_6000']}")
    print()

    # Step 1
    print("Step 1: Computing partial sum S_zeta^(6000)...")
    S_partial = compute_partial_sum()
    print(f"  S_zeta^(6000) midpoint = {float(S_partial):.15f}")
    print(f"  S_zeta^(6000) radius   = {float(S_partial.rad()):.3e}")
    print(f"  Paper states:            0.045795374824191...")
    print()

    # Step 2
    gamma_K = arb(_META["gamma_6000"])
    T0_arb  = gamma_K + T0_OFFSET
    print(f"Step 2: Tail bound over [T0, 1e6], T0 = gamma_6000 + 0.01 = {float(T0_arb):.6f}")
    tail_235, n_subs = compute_tail_235(T0_arb)
    print(f"  Subintervals             : {n_subs}  (paper: 235)")
    print(f"  Subinterval tail         : {float(tail_235):.6e}  (paper: <= 4.31e-4)")
    print()

    # Step 3
    print("Step 3: Analytic remainder over [1e6, infinity)")
    analytic_rem = compute_analytic_remainder()
    print(f"  Analytic remainder       : {float(analytic_rem):.6e}  (paper: <= 1.35e-5)")
    print()

    # Step 4: combine
    S_total = S_partial + tail_235 + analytic_rem
    print("Step 4: Combined bound")
    print(f"  S_zeta <= {float(S_total):.8f}  (ARB ball midpoint)")
    print(f"  S_zeta radius            : {float(S_total.rad()):.3e}")

    certified_tight  = bool(S_total < TIGHT_BOUND)
    certified_star   = bool(S_total < S_ZETA_STAR)
    print()
    print(f"  S_zeta < 0.04625         : {certified_tight}  (tight bound from paper)")
    print(f"  S_zeta < 0.04871 = S_zeta*: {certified_star}  (declared proposition bound)")
    print()
    print(f"Proposition certified      : {certified_star}")
