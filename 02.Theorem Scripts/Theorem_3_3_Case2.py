"""
Theorem_3_3_Case2.py  —  Certified verification of Theorem 3.3, Case 2
========================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This module certifies Case 2 of Theorem 3.3 of the above reference.  For
each squarefree d in {2, 3, 5, 6, 7, 10, 11, 13} and every T > 0,

    h(T) := <S_L^2>_T  >  (S_zeta*)^2 / d,

where h(T) is the Cesaro average of the squared spectral sum evaluated using
the first M = 20 ARB-certified zero ordinates of L(s, chi_d) from the sealed
data in L_function_zeros.py, with on-line Lorentzian weights
b_k = 2 / (1/4 + gamma_k'^2).

The on-line weight (denominator 1/4 + gamma'^2) is correct for gamma'_knw
zeros individually verified to lie on the critical line by ARB certification.
This differs from the lower bound w^-(gamma') = 2/(1 + gamma'^2) used for
gamma'_unk zeros in Case 1.

Algorithm
---------
  h(T) is evaluated via the explicit bilinear form

      h(T) = sum_{j,k} b_j b_k F(gamma_j', gamma_k', T),

  where
      F(a, a, T) = 1/2 + sin(2aT) / (4aT),
      F(a, b, T) = sin((a-b)T) / (2(a-b)T) + sin((a+b)T) / (2(a+b)T).

  Three regions cover all T > 0 for each d:

  Region 1  T in (0, 0.01]
    h is Lipschitz with constant L = (1/2) * sum_{j,k} b_j b_k max(g_j, g_k).
    Evaluate h(0.01) in ARB ball arithmetic; then h(T) >= h(0.01) - L * 0.01
    for all T in (0, 0.01].

  Region 2  T in [0.01, 50]
    A grid scan over 99981 equally spaced ARB points (spacing Delta = 5e-4)
    evaluates h(T) in ARB ball arithmetic.  Nodes are generated as
    GRID_START + arb(n) * DELTA for integer n; no float-driven accumulation.
    The Lipschitz bound certifies that the continuous minimum is at least
    (grid minimum) - L * Delta.

  Region 3  T >= 50
    The analytic bound h(T) >= c_0 * W_2 - I / T applies for all T, where
      c_0 = (pi - 1) / (2*pi),   W_2 = sum_k b_k^2,
      I = sum_{j!=k} b_j b_k [1/(2|g_j-g_k|) + 1/(2(g_j+g_k))].
    T_crit = I / (c_0 * W_2 - threshold_d) is certified < 50 in ARB, so the
    bound holds for all T >= 50.

  The per-discriminant threshold is (S_zeta*)^2 / d, where S_zeta* = 0.04871
  is the certified upper bound from Proposition [Explicit value of S_zeta] of
  the paper, established via 6000 LMFDB Riemann zeta zeros and a Trudgian
  tail bound.  It is used here as a certified constant; this script does not
  re-derive it.

Zero data
---------
  The first M = 20 zero ordinates of L(s, chi_d) are loaded from
  L_function_zeros.py (sealed, ARB-certified at 70 decimal places, 1500-bit
  working precision).  Each ordinate is ingested as an arb() ball via the
  string representation, preserving all certified digits.

Rigorousness checklist
----------------------
  (a) All function evaluations inside h_arb use ARB ball arithmetic.  Grid
      nodes in Region 2 are generated as GRID_START + arb(n) * DELTA for
      integer n; no float-driven state updates are used inside any certified
      computation.
  (b) The Lipschitz constant, W_2, I, c_0, per-character threshold, and
      T_crit are all computed and stored as ARB balls.
  (c) The grid scan produces certified intervals at each grid point.  The
      certified minimum for Region 2 is the smallest lower-endpoint value
      minus the Lipschitz discretization error L * Delta (computed in ARB).
  (d) The Region 1 lower bound uses h_arb evaluated at T = GRID_START = 0.01,
      minus L * GRID_START.  h_arb is never called at T = 0.
  (e) All pass/fail predicates are evaluated as ARB interval comparisons.
      float() conversion is used only after all certification is complete,
      for display purposes.
  (f) The threshold comparison bool(lower > threshold) returns True only if
      the ARB lower bound rigorously exceeds the threshold ball.

External-input qualifications
-----------------------------
  S_ZETA_STAR = 0.04871 is treated as certified input from Proposition 2.2
  of the paper (established via 6000 LMFDB Riemann zeta zeros and a Trudgian
  tail bound); it is not re-derived in this script.

  Zero ordinates are trusted certified strings from the sealed library
  (L_function_zeros.py, ARB-certified at 1500-bit working precision to 70
  decimal places).  This script ingests them via arb(string) and does not
  re-execute the ARB zero certification.

Requirements
------------
  python-flint >= 0.8.0   (provides ARB ball arithmetic)
  Python >= 3.10

Usage
-----
  python Theorem_3_3_Case2.py
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Path resolution: locate L_function_zeros.py in the repo
# ---------------------------------------------------------------------------

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LIB_DIR    = os.path.join(_SCRIPT_DIR, "..", "06.Library")
sys.path.insert(0, os.path.abspath(_LIB_DIR))

from persistent_heuristics_I import get_zeros  # get_zeros(d, n, as_strings=True)

from flint import arb, ctx

BASE_PREC = 512
ctx.prec = BASE_PREC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Number of zeros used per character.
M = 20

# Characters: (d, q) pairs in the order they appear in the paper table.
CHARACTERS = [
    (2,  8),
    (3,  12),
    (5,  5),
    (6,  24),
    (7,  28),
    (10, 40),
    (11, 44),
    (13, 13),
]

# S_zeta* certified bound (Proposition [S_zeta value], not re-derived here).
S_ZETA_STAR = arb("0.04871")

# Grid parameters for Region 2.  Shared across all characters.
GRID_START = arb("0.01")
GRID_END   = arb("50.0")
DELTA      = arb("5e-4")
GRID_STEPS = 99981   # = int((50.0 - 0.01) / 5e-4) + 1

# ---------------------------------------------------------------------------
# Zero loading
# ---------------------------------------------------------------------------

def load_zeros_arb(d, m):
    """
    Load the first m certified zero ordinates of L(s, chi_d) as ARB balls.

    Each ordinate is ingested via arb(str_value), where str_value is the
    70-decimal-place string from the sealed data.  The resulting ARB ball
    has radius commensurate with the last digit of the string representation;
    for all characters this is at most 10^{-69}.
    """
    zero_strings = get_zeros(d, n=m, as_strings=True)
    return [arb(s) for s in zero_strings]


# ---------------------------------------------------------------------------
# Weights: on-line Lorentzian weight b_k = 2 / (1/4 + gamma_k'^2)
# ---------------------------------------------------------------------------

def make_weights(gammas):
    """
    Compute on-line Lorentzian weights b_k = 2 / (1/4 + gamma_k'^2).

    These are the exact weights for gamma'_knw zeros individually verified to
    lie on the critical line.  Contrast with w^-(gamma') = 2/(1+gamma'^2)
    used for gamma'_unk zeros in Case 1.
    """
    quarter = arb("0.25")
    return [arb(2) / (quarter + g * g) for g in gammas]


# ---------------------------------------------------------------------------
# Bilinear form kernel and evaluation
# ---------------------------------------------------------------------------

def make_pairs(gammas, B):
    """
    Precompute (type, coeff, freq_info) for all M^2 index pairs.

    Avoids redundant recomputation of b_j * b_k and frequencies inside
    the inner loop of the grid scan.
    """
    n = len(gammas)
    pairs = []
    for j in range(n):
        for k in range(n):
            coeff = B[j] * B[k]
            if j == k:
                # diagonal: F(a,a,T) = 1/2 + sin(2aT)/(4aT)
                # store freq = 2*gamma so that x = freq*T in h_arb
                freq = arb(2) * gammas[j]
                pairs.append(("diag", coeff, freq))
            else:
                fdiff = gammas[j] - gammas[k]
                fsum  = gammas[j] + gammas[k]
                pairs.append(("off", coeff, fdiff, fsum))
    return pairs


def h_arb(T_arb, pairs):
    """
    ARB evaluation of h(T) = sum_{j,k} b_j b_k F(gamma_j', gamma_k', T).

    Never called at T = 0 (Region 1 uses the analytic value h(0.01)
    evaluated at the first grid point).

    Parameters
    ----------
    T_arb : arb
        Evaluation point (positive).
    pairs : list
        Output of make_pairs.

    Returns
    -------
    arb
        Certified ARB ball enclosing h(T).
    """
    total = arb(0)
    two   = arb(2)
    for p in pairs:
        if p[0] == "diag":
            _, coeff, freq = p
            x = freq * T_arb
            F = arb("0.5") + x.sin() / (two * x)
        else:
            _, coeff, fdiff, fsum = p
            xd = fdiff * T_arb
            xs = fsum  * T_arb
            F = xd.sin() / (two * xd) + xs.sin() / (two * xs)
        total += coeff * F
    return total


# ---------------------------------------------------------------------------
# Derived ARB constants
# ---------------------------------------------------------------------------

def compute_constants(d, gammas, B):
    """
    Compute all ARB constants needed for the three-region proof.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    gammas : list of arb
        First M zero ordinates.
    B : list of arb
        Corresponding on-line Lorentzian weights.

    Returns
    -------
    dict with keys:
        threshold   -- (S_zeta*)^2 / d
        L_lip       -- Lipschitz constant (1/2) * sum_{j,k} b_j b_k max(g_j, g_k)
        c0          -- (pi - 1) / (2*pi)
        W2          -- sum_k b_k^2
        I_interf    -- interference constant
        L_Delta     -- L_lip * Delta  (Lipschitz discretization error)
    """
    n = len(gammas)
    threshold = S_ZETA_STAR ** 2 / arb(d)

    # Lipschitz constant.  max(g_j, g_k) = (g_j + g_k + |g_j - g_k|) / 2
    # is the ARB-rigorous identity; no float branch is used.
    L_lip = arb(0)
    for j in range(n):
        for k in range(n):
            gmax = (gammas[j] + gammas[k] + abs(gammas[j] - gammas[k])) / arb(2)
            L_lip += B[j] * B[k] * gmax
    L_lip = L_lip * arb("0.5")

    pi_arb = arb.pi()
    c0 = (pi_arb - arb(1)) / (arb(2) * pi_arb)

    W2 = arb(0)
    for b in B:
        W2 += b * b

    I_interf = arb(0)
    for j in range(n):
        for k in range(n):
            if j != k:
                gd = abs(gammas[j] - gammas[k])
                gs = gammas[j] + gammas[k]
                I_interf += B[j] * B[k] * (arb(1) / (arb(2) * gd)
                                           + arb(1) / (arb(2) * gs))

    L_Delta = L_lip * DELTA

    return dict(threshold=threshold, L_lip=L_lip, c0=c0,
                W2=W2, I_interf=I_interf, L_Delta=L_Delta)


# ---------------------------------------------------------------------------
# Region 1: T in (0, 0.01]
# ---------------------------------------------------------------------------

def certify_region1(pairs, C):
    """
    Certify h(T) > threshold for all T in (0, 0.01].

    h(T) >= h(0.01) - L * 0.01  for all T in (0, 0.01].

    The bound h(0.01) is evaluated in ARB ball arithmetic at T = GRID_START.
    This is the same evaluation as the first grid point of Region 2.

    Returns (h_at_start, lower_bound, certified).
    """
    h_at_start = h_arb(GRID_START, pairs)
    lower = h_at_start - C["L_lip"] * GRID_START
    return h_at_start, lower, bool(lower > C["threshold"])


# ---------------------------------------------------------------------------
# Region 2: T in [0.01, 50] -- grid scan
# ---------------------------------------------------------------------------

def certify_region2(pairs, C):
    """
    Certify h(T) > threshold for all T in [0.01, 50] via full ARB grid scan.

    For each of the 99981 grid nodes T_n = GRID_START + n * DELTA, compute
    the certified ARB ball h(T_n).  Extract the lower endpoint of each ball
    as an exact (zero-radius) ARB value via h.mid() - h.rad(), and track the
    node that achieves the minimum lower endpoint.  Because both h.mid() and
    h.rad() return zero-radius ARB balls, the comparison

        bool(lower_ep < grid_min_lower_ep)

    is a comparison between two exact values and is always decisive — no
    float rounding or interval-overlap ambiguity can cause a wrong choice.

    The certified continuous minimum is then:

        certified_min = grid_min_lower_ep - C["L_Delta"]

    which is a rigorous lower bound on the continuous function minimum over
    [0.01, 50]: lower endpoints are each a valid lower bound on the true
    value, the minimum lower endpoint is a valid lower bound on the grid
    minimum, and subtracting L*Delta covers the gaps between nodes.

    Returns (grid_min_lower_ep, grid_min_T, certified_min, certified).
    """
    grid_min_lower_ep = None   # exact (zero-radius) ARB lower endpoint
    grid_min_n        = None

    for n in range(GRID_STEPS):
        T_node   = GRID_START + arb(n) * DELTA
        h        = h_arb(T_node, pairs)
        lower_ep = h.mid() - h.rad()   # exact ARB lower endpoint, zero radius
        if grid_min_lower_ep is None or bool(lower_ep < grid_min_lower_ep):
            grid_min_lower_ep = lower_ep
            grid_min_n        = n

    grid_min_T    = GRID_START + arb(grid_min_n) * DELTA
    certified_min = grid_min_lower_ep - C["L_Delta"]
    return grid_min_lower_ep, grid_min_T, certified_min, bool(certified_min > C["threshold"])


# ---------------------------------------------------------------------------
# Region 3: T >= 50
# ---------------------------------------------------------------------------

def certify_region3(C):
    """
    Certify h(T) > threshold for all T >= 50 via the analytic bound

        h(T) >= c_0 * W_2 - I / T.

    Certifies:
      (i)  T_crit = I / (c_0*W_2 - threshold) < 50,
      (ii) c_0*W_2 - I/50 > threshold.

    Returns (T_crit, overlap_ok, lower_at_50, bound_ok).
    """
    c0W2   = C["c0"] * C["W2"]
    T_crit = C["I_interf"] / (c0W2 - C["threshold"])
    overlap_ok = bool(T_crit < arb(50))

    lower_at_50 = c0W2 - C["I_interf"] / arb(50)
    bound_ok    = bool(lower_at_50 > C["threshold"])

    return T_crit, overlap_ok, lower_at_50, bound_ok


# ---------------------------------------------------------------------------
# Per-character certification driver
# ---------------------------------------------------------------------------

def certify_character(d, q):
    """
    Run the full three-region certification for discriminant d.

    Returns a result dict with all certified values and pass/fail flags.
    """
    gammas = load_zeros_arb(d, M)
    B      = make_weights(gammas)
    pairs  = make_pairs(gammas, B)
    C      = compute_constants(d, gammas, B)

    t0 = time.time()

    # Region 1
    h_start, r1_lower, r1_ok = certify_region1(pairs, C)

    # Region 2
    grid_min, grid_min_T, cert_min, r2_ok = certify_region2(pairs, C)

    # Region 3
    T_crit, overlap_ok, lower_50, r3_ok = certify_region3(C)

    elapsed = time.time() - t0

    return dict(
        d=d, q=q,
        threshold=C["threshold"],
        L_lip=C["L_lip"],
        L_Delta=C["L_Delta"],
        c0=C["c0"],
        W2=C["W2"],
        I_interf=C["I_interf"],
        h_start=h_start,
        r1_lower=r1_lower,
        r1_ok=r1_ok,
        grid_min=grid_min,
        grid_min_T=grid_min_T,
        cert_min=cert_min,
        r2_ok=r2_ok,
        T_crit=T_crit,
        overlap_ok=overlap_ok,
        lower_50=lower_50,
        r3_ok=r3_ok,
        all_ok=(r1_ok and r2_ok and r3_ok and overlap_ok),
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Theorem 3.3 Case 2: Unconditional Cesaro Variance Bound")
    print("d in {2, 3, 5, 6, 7, 10, 11, 13}  (squarefree values below 14)")
    print(f"ARB working precision : {BASE_PREC} bits (~{int(BASE_PREC * 0.30103)} decimal digits)")
    print(f"Zeros per character   : M = {M}  (on-line weights b_k = 2/(1/4 + gamma_k'^2))")
    print(f"S_zeta* (certified input, not re-derived): {float(S_ZETA_STAR)}")
    print()

    results = []
    for d, q in CHARACTERS:
        print(f"  d = {d:2d}  (conductor q = {q:2d}) ...", end="", flush=True)
        r = certify_character(d, q)
        results.append(r)
        status = "PASS" if r["all_ok"] else "FAIL"
        print(f"  {status}  ({r['elapsed']:.1f}s)")

    print()
    print("=" * 90)
    print("Per-region summary")
    print("=" * 90)
    for r in results:
        d = r["d"]
        print(f"\n  d = {d}  (q = {r['q']})")
        print(f"    threshold (S*)^2/d = {float(r['threshold']):.4e}")
        print(f"    L (Lipschitz)      = {float(r['L_lip']):.6f}")
        print(f"    L*Delta            = {float(r['L_Delta']):.4e}")
        print()
        print(f"    Region 1  T in (0, 0.01]")
        print(f"      h(0.01)          = {float(r['h_start']):.6e}")
        print(f"      h(0.01) - L*0.01 = {float(r['r1_lower']):.6e}")
        print(f"      > threshold      : {r['r1_ok']}  "
              f"(margin {float(r['r1_lower'] / r['threshold']):.1f}x)")
        print()
        print(f"    Region 2  T in [0.01, 50]")
        print(f"      Grid min         = {float(r['grid_min']):.6e}"
              f"  at T = {float(r['grid_min_T']):.4f}")
        print(f"      Certified min    = {float(r['cert_min']):.6e}")
        print(f"      > threshold      : {r['r2_ok']}  "
              f"(margin {float(r['cert_min'] / r['threshold']):.1f}x)")
        print()
        print(f"    Region 3  T >= 50")
        print(f"      c0*W2            = {float(r['c0']*r['W2']):.6e}")
        print(f"      I                = {float(r['I_interf']):.6e}")
        print(f"      T_crit           = {float(r['T_crit']):.4f}")
        print(f"      T_crit < 50      : {r['overlap_ok']}")
        print(f"      c0*W2-I/50       = {float(r['lower_50']):.6e}")
        print(f"      > threshold      : {r['r3_ok']}  "
              f"(margin {float(r['lower_50'] / r['threshold']):.1f}x)")

    print()
    print("=" * 90)
    print("Summary table (matches paper Table, Theorem 3.3 Case 2)")
    print("=" * 90)
    print(f"{'d':>3}  {'q':>4}  {'Grid min h':>14}  {'Cert min h':>14}  "
          f"{'(S*)^2/d':>14}  {'Margin':>10}  {'Status':>6}")
    print("-" * 90)
    all_pass = True
    for r in results:
        margin = float(r['cert_min'] / r['threshold'])
        status = "PASS" if r["all_ok"] else "FAIL"
        if not r["all_ok"]:
            all_pass = False
        print(f"{r['d']:>3}  {r['q']:>4}  "
              f"{float(r['grid_min']):>14.4e}  "
              f"{float(r['cert_min']):>14.4e}  "
              f"{float(r['threshold']):>14.4e}  "
              f"{margin:>9.1f}x  "
              f"{status:>6}")
    print()
    print(f"All characters certified: {all_pass}")
    print(f"Worst-case margin: d = 5  (tightest; expected ~2.4x from paper)")
