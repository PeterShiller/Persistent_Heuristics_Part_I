"""
Theorem_4_10_Case1.py  —  Certified verification of Theorem 4.10, Case 1 (d >= 14)
================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This module certifies Case 1 of Theorem 4.10 of the above reference.  For
every squarefree d >= 14 and every T > 0,

    h(T) := <S_L^2>_T  >  (S_zeta*)^2 / d,

where h(T) is the Cesaro average of the squared spectral sum evaluated at the
worst-case five-zero configuration gamma' in {6.0, 7.25, 8.5, 9.75, 11.0}
with weights b_k = 2 / (1 + gamma_k'^2).

Algorithm
---------
  h(T) is evaluated via the explicit bilinear form

      h(T) = sum_{j,k} b_j b_k F(gamma_j', gamma_k', T),

  where
      F(a, a, T) = 1/2 + sin(2aT) / (4aT),
      F(a, b, T) = sin((a-b)T) / (2(a-b)T) + sin((a+b)T) / (2(a+b)T).

  Three regions cover all T > 0:

  Region 1  T in (0, 0.01]
    h is Lipschitz with constant L = (1/2) * sum_{j,k} b_j b_k max(g_j, g_k).
    The bound h(T) >= h(0) - L * T, with h(0) = (sum b_k)^2, certifies the
    inequality for all T in (0, 0.01].  Certified lower bound 0.02325,
    exceeding the threshold by 137x.

  Region 2  T in [0.01, 50]
    A grid scan over 99981 equally spaced ARB points (spacing Delta = 5e-4)
    evaluates h(T) in ARB ball arithmetic.  Nodes are generated as
    GRID_START + arb(n) * DELTA for integer n; no float-driven accumulation.
    The Lipschitz bound certifies that the continuous minimum is at least
    (grid minimum) - L * Delta.  Certified minimum 0.001953, exceeding the
    threshold by 11.5x.

  Region 3  T >= 50
    The analytic bound h(T) >= c_0 * W_2 - I / T applies for all T, where
      c_0 = (pi - 1) / (2*pi),   W_2 = sum_k b_k^2,
      I = sum_{j!=k} b_j b_k [1/(2|g_j-g_k|) + 1/(2(g_j+g_k))].
    T_crit = I / (c_0 * W_2 - threshold) is certified < 50 in ARB, so the
    bound holds for all T >= 50.

  The threshold is (S_zeta*)^2 / 14, where S_zeta* = 0.04871 is the certified
  upper bound from Proposition [Explicit value of S_zeta] of the paper,
  established via 6000 LMFDB Riemann zeta zeros and a Trudgian tail bound.
  It is used here as a certified constant; this script does not re-derive it.
  The five-zero configuration is the worst-case hypothetical extremal; every
  actual squarefree d >= 14 has more zeros and a strictly better margin.

Rigorousness checklist
----------------------
  (a) All function evaluations inside h_arb use ARB ball arithmetic
      (arb.sin, arb arithmetic).  Grid nodes in Region 2 are generated as
      GRID_START + arb(n) * DELTA for integer n; no float-driven state
      updates are used inside any certified computation.
  (b) The Lipschitz constant, h(0), W_2, I, c_0, threshold, and T_crit
      are all computed and stored as ARB balls.
  (c) The grid scan produces a certified interval with radius below 1e-150
      at each grid point.  The Lipschitz discretization error L*Delta is
      subtracted as an ARB quantity before the threshold comparison.
  (d) All pass/fail predicates are evaluated as ARB comparisons.  float()
      conversion is used only after all certification is complete, for display.
  (e) h_arb is never called at T = 0; Region 1 uses h(0) = (sum b_k)^2
      directly.  The grid scan starts at T = 0.01, so all sinc arguments
      are nonzero and every division is well-defined.

No zero ordinate data from any appendix is used in this computation.

Requirements
------------
  python-flint >= 0.8.0   (provides ARB ball arithmetic)
  Python >= 3.10

Usage
-----
  python Theorem_4_10_Case1.py
"""

from flint import arb, ctx
import time

BASE_PREC = 512
ctx.prec = BASE_PREC

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Worst-case five-zero configuration for d >= 14 (hypothetical extremal).
GAMMAS_FLOAT = [6.0, 7.25, 8.5, 9.75, 11.0]

# S_zeta* certified bound (Proposition [S_zeta value], established via 6000
# LMFDB zeta zeros; used here as a certified constant, not re-derived).
S_ZETA_STAR = arb("0.04871")

# Grid parameters for Region 2.
GRID_START = arb("0.01")
GRID_END   = arb("50.0")
DELTA      = arb("5e-4")
GRID_STEPS = 99981   # = int((50.0 - 0.01) / 5e-4) + 1

# ---------------------------------------------------------------------------
# Precomputed ARB constants
# ---------------------------------------------------------------------------

GAMMAS = [arb(str(g)) for g in GAMMAS_FLOAT]
B      = [arb(2) / (arb(1) + g ** 2) for g in GAMMAS]
N      = len(GAMMAS)


def _make_pairs():
    """
    Precompute (type, coeff, freq_info) for all 25 index pairs.
    Avoids repeated recomputation of b_j * b_k and frequencies inside the scan.
    """
    pairs = []
    for j in range(N):
        for k in range(N):
            coeff = B[j] * B[k]
            if j == k:
                freq = arb(2) * GAMMAS[j]          # argument of sin in F(a,a,T)
                pairs.append(("diag", coeff, freq))
            else:
                fdiff = GAMMAS[j] - GAMMAS[k]      # (a - b)
                fsum  = GAMMAS[j] + GAMMAS[k]      # (a + b)
                pairs.append(("off", coeff, fdiff, fsum))
    return pairs

PAIRS = _make_pairs()


def h_arb(T_arb):
    """
    ARB evaluation of h(T) = sum_{j,k} b_j b_k F(gamma_j', gamma_k', T).

    Returns a certified ARB ball.  This function is never called at T = 0;
    Region 1 uses the exact value h(0) = (sum b_k)^2 directly.  The grid scan
    starts at T = GRID_START = 0.01 > 0, so the sinc arguments are always
    nonzero and the division is well-defined.
    """
    total = arb(0)
    for p in PAIRS:
        if p[0] == "diag":
            _, coeff, freq = p
            x = freq * T_arb
            F = arb("0.5") + x.sin() / (arb(2) * x)
        else:
            _, coeff, fdiff, fsum = p
            xd = fdiff * T_arb
            xs = fsum  * T_arb
            F = xd.sin() / (arb(2) * xd) + xs.sin() / (arb(2) * xs)
        total += coeff * F
    return total


# ---------------------------------------------------------------------------
# Derived ARB constants
# ---------------------------------------------------------------------------

def compute_constants():
    """
    Compute and return all ARB constants needed for the three-region proof.

    Returns a dict with keys:
      threshold    -- (S_zeta*)^2 / 14, upper bound on (S_zeta*)^2 / d for d >= 14
      h0           -- h(0) = (sum b_k)^2 = W_1(L)^2
      L_lip        -- Lipschitz constant for h
      c0           -- (pi - 1) / (2*pi), lower bound on F(a,a,T)
      W2           -- sum_k b_k^2
      I_interf     -- interference constant
      L_Delta      -- Lipschitz discretization error for Region 2
    """
    threshold = S_ZETA_STAR ** 2 / arb(14)

    h0 = sum(B[j] * B[k] for j in range(N) for k in range(N))

    L_lip = arb("0.5") * sum(
        B[j] * B[k] * arb(str(max(GAMMAS_FLOAT[j], GAMMAS_FLOAT[k])))
        for j in range(N) for k in range(N)
    )

    pi_arb = arb.pi()
    c0 = (pi_arb - arb(1)) / (arb(2) * pi_arb)

    W2 = sum(B[k] ** 2 for k in range(N))

    I_interf = arb(0)
    for j in range(N):
        for k in range(N):
            if j != k:
                gd = abs(GAMMAS[j] - GAMMAS[k])
                gs = GAMMAS[j] + GAMMAS[k]
                I_interf += B[j] * B[k] * (arb(1) / (arb(2) * gd)
                                           + arb(1) / (arb(2) * gs))

    L_Delta = L_lip * DELTA

    return dict(threshold=threshold, h0=h0, L_lip=L_lip, c0=c0,
                W2=W2, I_interf=I_interf, L_Delta=L_Delta)


# ---------------------------------------------------------------------------
# Region 1: T in (0, 0.01]
# ---------------------------------------------------------------------------

def certify_region1(C):
    """
    Certify h(T) > threshold for all T in (0, 0.01] via Lipschitz bound.

    h(T) >= h(0) - L * T >= h(0) - L * 0.01  for all T in (0, 0.01].

    Returns (lower_bound, certified).
    """
    lower = C["h0"] - C["L_lip"] * GRID_START
    return lower, bool(lower > C["threshold"])


# ---------------------------------------------------------------------------
# Region 2: T in [0.01, 50] -- grid scan
# ---------------------------------------------------------------------------

def certify_region2(C):
    """
    Certify h(T) > threshold for all T in [0.01, 50] via grid scan.

    Evaluates h at 99981 equally spaced ARB points, finds the grid minimum,
    subtracts the Lipschitz discretization error L*Delta, and certifies that
    the resulting lower bound exceeds the threshold.

    Returns (grid_min_val, grid_min_T, certified_min, certified).
    """
    grid_min = None
    grid_min_n = None

    for n in range(GRID_STEPS):
        # Node generated by exact integer arithmetic: no float accumulation.
        T_node = GRID_START + arb(n) * DELTA
        h = h_arb(T_node)
        if grid_min is None or bool(h < grid_min):
            grid_min = h
            grid_min_n = n

    certified_min = grid_min - C["L_Delta"]
    grid_min_T = GRID_START + arb(grid_min_n) * DELTA
    return grid_min, grid_min_T, certified_min, bool(certified_min > C["threshold"])


# ---------------------------------------------------------------------------
# Region 3: T >= 50
# ---------------------------------------------------------------------------

def certify_region3(C):
    """
    Certify h(T) > threshold for all T >= 50 via the analytic bound

        h(T) >= c_0 * W_2 - I / T.

    Certifies:
      (i)  T_crit = I / (c_0*W_2 - threshold) < 50  (overlap with Region 2),
      (ii) c_0*W_2 - I/50 > threshold                (bound holds at T=50).

    Returns (T_crit, overlap_ok, lower_at_50, bound_ok).
    """
    c0W2 = C["c0"] * C["W2"]
    T_crit = C["I_interf"] / (c0W2 - C["threshold"])
    overlap_ok = bool(T_crit < arb(50))

    lower_at_50 = c0W2 - C["I_interf"] / arb(50)
    bound_ok = bool(lower_at_50 > C["threshold"])

    return T_crit, overlap_ok, lower_at_50, bound_ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Theorem 4.10 Case 1: Unconditional Cesaro Variance Bound (d >= 14)")
    print(f"ARB working precision : {BASE_PREC} bits (~{int(BASE_PREC * 0.30103)} decimal digits)")
    print(f"Worst-case configuration: gamma' = {GAMMAS_FLOAT}")
    print(f"S_zeta* (certified input, Prop. S_zeta value): {float(S_ZETA_STAR)}")
    print()

    C = compute_constants()

    print(f"Derived constants")
    print(f"  threshold = (S_zeta*)^2/14 = {float(C['threshold']):.6e}")
    print(f"  h(0)      = (sum b_k)^2   = {float(C['h0']):.8f}")
    print(f"  L (Lipschitz)             = {float(C['L_lip']):.8f}")
    print(f"  c_0 = (pi-1)/(2*pi)       = {float(C['c0']):.8f}")
    print(f"  W_2 = sum b_k^2           = {float(C['W2']):.8f}")
    print(f"  I   (interference)        = {float(C['I_interf']):.6e}")
    print(f"  L*Delta                   = {float(C['L_Delta']):.6e}")
    print()

    # Region 1
    r1_lower, r1_ok = certify_region1(C)
    print("Region 1  T in (0, 0.01]")
    print(f"  h(0) - L*0.01 = {float(r1_lower):.8f}")
    print(f"  > threshold   : {r1_ok}  (margin {float(r1_lower / C['threshold']):.1f}x)")
    print()

    # Region 2
    print("Region 2  T in [0.01, 50]  (grid scan, ~15s)")
    t0 = time.time()
    grid_min, grid_min_T, cert_min, r2_ok = certify_region2(C)
    elapsed = time.time() - t0
    print(f"  Grid scan: {GRID_STEPS} points in {elapsed:.1f}s")
    print(f"  Grid minimum h = {float(grid_min):.8f}  at T = {grid_min_T}")
    print(f"  L*Delta        = {float(C['L_Delta']):.6e}")
    print(f"  Certified min  = {float(cert_min):.8f}")
    print(f"  > threshold    : {r2_ok}  (margin {float(cert_min / C['threshold']):.2f}x)")
    print()

    # Region 3
    T_crit, overlap_ok, lower_50, r3_ok = certify_region3(C)
    print("Region 3  T >= 50")
    print(f"  T_crit = I / (c_0*W_2 - threshold) = {float(T_crit):.4f}")
    print(f"  T_crit < 50 (overlap with Region 2) : {overlap_ok}")
    print(f"  c_0*W_2 - I/50 = {float(lower_50):.8f}")
    print(f"  > threshold    : {r3_ok}  (margin {float(lower_50 / C['threshold']):.1f}x)")
    print()

    all_certified = r1_ok and r2_ok and r3_ok
    print(f"All regions certified: {all_certified}")
