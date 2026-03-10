"""
Theorem_12_2.py  —  Certified verification of Theorem 12.2 (Integer crossing obstruction)
=======================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

Proof structure and certification strategy per case
-----------------------------------------------------
  Case 1  s = 2, d = 3
    The identity zeta(2)^2 = 3 * L(2, chi_3)^2 is certified by evaluating both
    sides via ARB Hurwitz decomposition at s = 2 and certifying the absolute
    difference is < 1e-100.

  Case 2a  s = 2, d >= 22 (threshold argument)
    The bound c_d >= (2 - pi^2/6) * sqrt(d) / pi^2 implies c_d > 1/6 for all
    squarefree d >= 22.  The threshold (2 - pi^2/6) * sqrt(22) / pi^2 > 1/6 is
    certified as an ARB inequality (pure pi arithmetic, no L-function evaluation).

  Case 2b  s = 2, d in {2, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21}
    For each d, compute c_d = sqrt(d) * L(2, chi_d) / pi^2 via ARB Hurwitz
    decomposition and certify directly that the resulting ARB ball is disjoint
    from 1/6, i.e., certify either c_d > 1/6 or c_d < 1/6 as an ARB predicate.
    No exact rational identification of c_d is claimed or required; the theorem
    needs only c_d != 1/6, which is exactly what ARB disjointness delivers.
    The table in the paper lists the numerically computed rational approximations
    for the reader's convenience; their exactness follows independently from the
    Klingen-Siegel Bernoulli formula but is not part of the certified proof.

  Case 3  s >= 4, all d
    At s = 4, the all-inert bound gives G(4) <= zeta(4)^2 / zeta(8).  The ARB
    evaluation certifies this ratio equals exactly 7/6 (|ratio - 7/6| < 1e-100).
    The final step 49 < 72 (i.e., (7/6)^2 < 2 <= d) is a Python integer
    comparison, requiring no floating-point arithmetic.

  Cases 4+5  s = 3, d >= 3 and d = 2
    Both cases are closed by explicit partial-product bounds (computed in the
    paper by hand).  The final inequalities 1507653^2 < 3 * 1007500^2 (d >= 3)
    and 178326^2 < 2 * 138229^2 (d = 2) are Python integer comparisons.

Rigorousness checklist
-----------------------
  (a) All L-function evaluations use ARB ball arithmetic via acb.zeta (Hurwitz
      decomposition) at 512-bit working precision.  No mpmath or float arithmetic
      is used inside any certified computation.
  (b) Case 2b certifies c_d != 1/6 directly: the ARB ball for c_d is certified
      disjoint from 1/6 via a single ARB predicate (c_d > 1/6 or c_d < 1/6).
      No exact rational identification is claimed.
  (c) Case 3 uses ARB to confirm the Bernoulli-number identity and pure integer
      comparison for the final bound.
  (d) Cases 4 and 5 both use ARB interval arithmetic.  Case 4 certifies the
      universal all-inert bound at s = 3 via acb.zeta(3) and acb.zeta(6).
      Case 5 certifies G(3, 2)^2 < 2 directly via acb.zeta(3) and L_hurwitz.
      No partial-product bounds from the paper are used in either case.
  (e) All pass/fail predicates are evaluated as ARB comparisons or exact integer
      comparisons.  float() conversion is used only after all certification is
      complete, for display.

No zero ordinate data from any appendix is used in this computation.
"""

from flint import arb, acb, ctx
import sympy

BASE_PREC = 512
ctx.prec = BASE_PREC

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def fundamental_discriminant(d):
    """Return the fundamental discriminant of Q(sqrt(d))."""
    return d if d % 4 == 1 else 4 * d


def chi_kronecker(D, n):
    """
    Kronecker symbol (D/n) for the primitive character of conductor D.

    For odd n, delegates to sympy.jacobi_symbol (which is exact for odd
    second argument).  For even n, factors out the power of 2 and applies
    the supplementary law for (D/2) before multiplying by the Jacobi symbol
    on the odd part.
    """
    n = int(n) % D
    if n == 0:
        return 0
    if n % 2 == 0:
        v, m = 0, n
        while m % 2 == 0:
            v += 1
            m //= 2
        r = D % 8
        k2 = 1 if r in (1, 7) else (-1 if r in (3, 5) else 0)
        if k2 == 0:
            return 0
        km = int(sympy.jacobi_symbol(D, m)) if m > 1 else 1
        return (k2 ** v) * km
    return int(sympy.jacobi_symbol(D, n))


def L_hurwitz(s_arb, d):
    """
    L(s, chi_d) via Hurwitz decomposition (ARB interval arithmetic).

      L(s, chi_d) = q^{-s} * sum_{a=1}^{q} chi_d(a) * zeta(s, a/q)

    where q = fundamental_discriminant(d).  All arithmetic is performed
    inside ARB; the return value is a certified real ARB ball.
    """
    D = fundamental_discriminant(d)
    q_arb = arb(D)
    s_c = acb(s_arb)
    total = arb(0)
    for a in range(1, D + 1):
        chi = chi_kronecker(D, a)
        if chi == 0:
            continue
        hz = acb.zeta(s_c, acb(arb(a)) / acb(q_arb)).real
        total += arb(chi) * hz
    return total / q_arb ** s_arb


def zeta_arb(s_int):
    """ARB enclosure of zeta(s) at a positive integer s."""
    return acb.zeta(acb(arb(s_int))).real


# ---------------------------------------------------------------------------
# Case 1: s = 2, d = 3
# ---------------------------------------------------------------------------

def certify_case1():
    """
    Certify the identity  zeta(2)^2 = 3 * L(2, chi_3)^2.

    Returns (diff_ball, certified) where diff_ball is the ARB enclosure of
    |zeta(2)^2 - 3 * L(2, chi_3)^2| and certified is True iff diff_ball < 1e-100.
    """
    z2 = zeta_arb(2)
    L2_3 = L_hurwitz(arb(2), 3)
    diff = abs(z2 ** 2 - arb(3) * L2_3 ** 2)
    return diff, bool(diff < arb("1e-100"))


# ---------------------------------------------------------------------------
# Case 2a: threshold for d >= 22
# ---------------------------------------------------------------------------

def certify_case2a():
    """
    Certify (2 - pi^2/6) * sqrt(22) / pi^2 > 1/6.

    This implies c_d > 1/6 for all squarefree d >= 22, ruling out s_*(d) = 2
    for those d.  The computation uses only pi; no L-function evaluation.

    Returns (lhs, rhs, margin, certified).
    """
    pi2 = arb.pi() ** 2
    lhs = (arb(2) - pi2 / arb(6)) * arb(22).sqrt() / pi2
    rhs = arb(1) / arb(6)
    certified = bool(lhs > rhs)
    return lhs, rhs, lhs - rhs, certified


# ---------------------------------------------------------------------------
# Case 2b: exact c_d values for squarefree d <= 21, d != 3
# ---------------------------------------------------------------------------

# Rational approximations to c_d, listed for display purposes only.
# Their exactness follows from the Klingen-Siegel Bernoulli formula but is
# not used in the certified proof; certification uses only ARB disjointness
# from 1/6.
C_D_DISPLAY = {
    2:  (1,  8),
    5:  (4,  25),
    6:  (1,  4),
    7:  (2,  7),
    10: (7,  20),
    11: (7,  22),
    13: (4,  13),
    14: (5,  14),
    15: (2,  5),
    17: (8,  17),
    19: (1,  2),
    21: (8,  21),
}


def certify_case2b():
    """
    For each d in C_D_DISPLAY, certify c_d != 1/6 by direct ARB disjointness.

    The ARB ball for c_d = sqrt(d) * L(2, chi_d) / pi^2 is evaluated via
    Hurwitz decomposition.  The predicate  c_d > 1/6  or  c_d < 1/6  is then
    certified as an ARB comparison.  This is all the theorem requires; no exact
    rational identification of c_d is claimed.

    The display rationals p/q from C_D_DISPLAY are passed through for printing
    only; they play no role in the certified inequality.

    Returns a dict  d -> (p, q, c_d_ball, diff_ball, certified).
    """
    sixth = arb(1) / arb(6)
    results = {}
    for d, (p, q) in C_D_DISPLAY.items():
        L2 = L_hurwitz(arb(2), d)
        c_d = arb(d).sqrt() * L2 / arb.pi() ** 2
        diff = c_d - sixth
        certified = bool(diff > arb(0)) or bool(diff < arb(0))
        results[d] = (p, q, c_d, diff, certified)
    return results


# ---------------------------------------------------------------------------
# Case 3: s >= 4, all d
# ---------------------------------------------------------------------------

def certify_case3():
    """
    Certify zeta(4)^2 / zeta(8) = 7/6  (ARB),  then  49 < 72  (integer).

    The first step uses the Bernoulli values zeta(4) = pi^4/90,
    zeta(8) = pi^8/9450, whose ratio squared gives 9450/8100 = 7/6.
    ARB certifies |ratio - 7/6| < 1e-100.

    The second step is the Python integer comparison 49 < 72, establishing
    (7/6)^2 = 49/36 < 2 <= d for every squarefree d >= 2, so G(s)^2 < d for
    all s >= 4.

    Returns (ratio_ball, ratio_err, ratio_ok, int_ok).
    """
    z4 = zeta_arb(4)
    z8 = zeta_arb(8)
    ratio = z4 ** 2 / z8
    ratio_err = abs(ratio - arb(7) / arb(6))
    ratio_ok = bool(ratio_err < arb("1e-100"))
    int_ok = (49 < 72)                 # (7/6)^2 = 49/36; 49/36 < 2 iff 49 < 72
    return ratio, ratio_err, ratio_ok, int_ok


# ---------------------------------------------------------------------------
# Cases 4+5: s = 3
# ---------------------------------------------------------------------------

def certify_case4():
    """
    Case 4 (s = 3, d >= 3): certify G(3, d)^2 < d for all squarefree d >= 3.

    The all-inert Euler product bound gives G(s, d) <= zeta(s)^2 / zeta(2s) for
    all squarefree d >= 2 and all s > 1.  At s = 3 this yields

      G(3, d) <= zeta(3)^2 / zeta(6)  for all d.

    We certify via ARB that (zeta(3)^2 / zeta(6))^2 < 3.  Since the bound is
    universal in d and sqrt(3) <= sqrt(d) for all d >= 3, this closes the case.

    The all-inert bound is NOT tight enough for d = 2: the ARB computation
    confirms (zeta(3)^2/zeta(6))^2 > 2, which is why d = 2 requires Case 5.

    Returns (all_inert_sq, all_inert_sq_lt_3, all_inert_sq_lt_2).
    """
    z3 = zeta_arb(3)
    z6 = zeta_arb(6)
    all_inert = z3 ** 2 / z6
    sq = all_inert ** 2
    return sq, bool(sq < arb(3)), bool(sq < arb(2))


def certify_case5():
    """
    Case 5 (s = 3, d = 2): certify G(3, 2)^2 < 2 directly via ARB.

    The all-inert bound is insufficient here (Case 4 shows its square exceeds 2).
    Instead we compute G(3, 2) = zeta(3) / L(3, chi_8) directly: zeta(3) via
    acb.zeta and L(3, chi_8) via Hurwitz decomposition, both in ARB at 512-bit
    precision.  The predicate G(3, 2)^2 < 2 is then certified as an ARB
    comparison.

    Returns (G3_sq, certified).
    """
    z3 = zeta_arb(3)
    L3_2 = L_hurwitz(arb(3), 2)
    G3_2 = z3 / L3_2
    G3_sq = G3_2 ** 2
    return G3_sq, bool(G3_sq < arb(2))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Theorem 12.2: Integer Crossing Obstruction")
    print(f"ARB working precision : {BASE_PREC} bits (~{int(BASE_PREC * 0.30103)} decimal digits)")
    print()

    all_certified = True

    # --- Case 1 ---
    diff, c1_ok = certify_case1()
    all_certified &= c1_ok
    print("Case 1  (s=2, d=3)")
    print(f"  |zeta(2)^2 - 3*L(2,chi_3)^2| = {diff}")
    print(f"  Certified < 1e-100           : {c1_ok}")
    print()

    # --- Case 2a ---
    lhs, rhs, margin, c2a_ok = certify_case2a()
    all_certified &= c2a_ok
    print("Case 2a  (s=2, d >= 22, threshold)")
    print(f"  LHS = (2 - pi^2/6)*sqrt(22)/pi^2 = {lhs}")
    print(f"  RHS = 1/6                         = {rhs}")
    print(f"  Margin (LHS - RHS)                = {margin}")
    print(f"  LHS > RHS certified               : {c2a_ok}")
    print()

    # --- Case 2b ---
    c2b = certify_case2b()
    c2b_ok = all(v[4] for v in c2b.values())
    all_certified &= c2b_ok
    print("Case 2b  (s=2, d in {2,5,6,7,10,11,13,14,15,17,19,21})")
    print(f"  {'d':>3}  {'display c_d':>8}  {'c_d(ARB) ball':>45}  {'c_d - 1/6':>25}  certified")
    print(f"  {'-'*100}")
    for d in sorted(c2b):
        p, q, c_d_ball, diff, ok = c2b[d]
        print(f"  {d:>3}  {p}/{q:<6}  {str(c_d_ball):>45}  {str(diff):>25}  {ok}")
    print(f"  All certified != 1/6 (ARB disjointness): {c2b_ok}")
    print()

    # --- Case 3 ---
    ratio, ratio_err, ratio_ok, int_ok = certify_case3()
    c3_ok = ratio_ok and int_ok
    all_certified &= c3_ok
    print("Case 3  (s >= 4, all d)")
    print(f"  zeta(4)^2 / zeta(8)          = {ratio}")
    print(f"  |ratio - 7/6| < 1e-100       : {ratio_ok}")
    print(f"  49 < 72 (=> (7/6)^2 < 2 <= d): {int_ok}")
    print()

    # --- Case 4 ---
    all_inert_sq, c4_ok, c4_lt2 = certify_case4()
    all_certified &= c4_ok
    print("Case 4  (s=3, d >= 3)")
    print(f"  (zeta(3)^2/zeta(6))^2 = {all_inert_sq}")
    print(f"  < 3 certified (closes all d >= 3) : {c4_ok}")
    print(f"  < 2 certified (all-inert for d=2) : {c4_lt2}  [expected False]")
    print()

    # --- Case 5 ---
    G3_sq, c5_ok = certify_case5()
    all_certified &= c5_ok
    print("Case 5  (s=3, d=2)")
    print(f"  G(3,2)^2 = (zeta(3)/L(3,chi_8))^2 = {G3_sq}")
    print(f"  < 2 certified                      : {c5_ok}")
    print()

    print(f"All cases certified: {all_certified}")
