"""
Theorem_9_1.py  —  Certified verification of Theorem 9.1 (Unique null crossing)
===============================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This module certifies the null crossing table in Theorem 9.1 of the above
reference.  For each squarefree d in {2, 3, 5, 7, 13}, it computes:

    s_*(d) : the unique s > 1 satisfying G(s) := zeta(s) / L(s, chi_d) = sqrt(d),

together with L(1, chi_d) and a certified residual |G(s_*(d)) - sqrt(d)|,
all kept in ARB ball arithmetic throughout.

Algorithm
---------
  G(s) = zeta(s) / L(s, chi_d), where L(s, chi_d) is evaluated via the
  Hurwitz zeta decomposition

      L(s, chi_d) = q^{-s} * sum_{a=1}^{q} chi_d(a) * zeta(s, a/q),

  with q = |Delta_K| the conductor of Q(sqrt(d)).  Each Hurwitz value
  zeta(s, a/q) is computed by ARB's built-in acb.zeta.

  Two certification strategies are used, chosen by the nature of s_*(d):

  (A) Generic bisection (d in {2, 5, 7, 13}): s_*(d) is irrational.
      At each step the algorithm evaluates G(s_mid) as an ARB ball and
      performs a decisive trichotomy:
        - G(s_mid) certainly > sqrt(d)  =>  advance lower endpoint
        - G(s_mid) certainly < sqrt(d)  =>  advance upper endpoint
        - intervals overlap              =>  raise ctx.prec by PREC_STEP
                                             and retry (up to 5 times)
      The bracket invariant G(s_lo) > target > G(s_hi) is maintained by
      certified comparisons.  The initial bracket is itself certified
      before bisection begins.  After bisection, L_hurwitz is confirmed
      > 0 on the final bracket (denominator safety), and the residual
      is certified < 1e-60 as an ARB predicate.

  (B) Integer crossing (d = 3): s_*(3) = 2 exactly (Theorem 9.2).
      The ARB bisection approach is inapplicable here because G(2) =
      sqrt(3) holds as an identity; eventually the midpoint coincides
      with the answer and no decisive comparison is possible.  Instead,
      the module certifies the crossing directly:
        (i)   G(1.99, 3) > sqrt(3)   [certified ARB predicate]
        (ii)  G(2.01, 3) < sqrt(3)   [certified ARB predicate]
        (iii) |G(2, 3) - sqrt(3)| < 1e-60  [certified ARB predicate]
      Together with the strict monotonicity of G (Lemma 9.1 of the paper),
      (i)-(ii) certify existence and uniqueness of s_*(3) in (1.99, 2.01),
      and (iii) certifies the evaluation at the integer point.

  L(1, chi_d) is computed via the digamma formula
      L(1, chi) = -(1/q) * sum_{a=1}^{q} chi(a) * psi(a/q).
  The Kronecker symbol chi_Delta(a) for even a is computed by factoring
  out powers of 2, since sympy.jacobi_symbol requires an odd denominator.

Rigorousness checklist
----------------------
  (a) Every function evaluation uses ARB ball arithmetic (acb.zeta,
      acb.digamma, arb.sqrt, arb.log, acb.lgamma); no mpmath or float
      arithmetic is used inside any certified computation.
  (b) Bisection uses a decisive trichotomy; the undecidable case raises
      precision rather than silently branching.
  (c) Initial brackets are numerically certified before bisection begins.
  (d) L_hurwitz > 0 on the entire final bracket is certified by evaluating
      L_hurwitz on a single ARB ball s_lo.union(s_hi) that encloses the whole
      interval.  Positivity of this enclosure certifies L > 0 for all s in
      [s_lo, s_hi], not merely at the two endpoints.
  (e) All pass/fail predicates (residual < 1e-60, bracket endpoints,
      L > 0) are evaluated as ARB comparisons.  float() conversion is
      used only after all certification is complete, for display.
  (f) d = 3 is handled by a direct integer-crossing strategy that avoids
      the structurally undecidable bisection comparison at the exact root.

No zero ordinate data from any appendix is used in this computation.

Requirements
------------
  python-flint >= 0.8.0   (provides ARB ball arithmetic)
  sympy >= 1.12
  Python >= 3.10

Usage
-----
  python Theorem_9_1.py
"""

from flint import arb, acb, ctx
import sympy


# ---------------------------------------------------------------------------
# Precision schedule
# ---------------------------------------------------------------------------

BASE_PREC = 512    # ~154 decimal digits; matches the paper
PREC_STEP = 128    # precision increment on undecidable bisection comparison


# ---------------------------------------------------------------------------
# Kronecker symbol chi_Delta(a) for all integer a >= 1.
# sympy.jacobi_symbol requires an odd denominator; even a is handled by
# factoring out powers of 2 and applying Kronecker(Delta, 2) separately.
# ---------------------------------------------------------------------------

def _kronecker_at_2(Delta):
    """Kronecker symbol (Delta/2): +1 if Delta=+/-1 mod 8, -1 if +/-3 mod 8, 0 if even."""
    r = Delta % 8
    if r in (1, 7):   return  1
    elif r in (3, 5): return -1
    else:             return  0


def kronecker_symbol(Delta, a):
    """
    Return chi_Delta(a) = Kronecker symbol (Delta/a) as an integer in {-1, 0, 1}.
    Valid for all integers a >= 1.
    """
    if a == 1:
        return 1
    if a % 2 == 0:
        # (Delta / 2^v*m) = (Delta/2)^v * (Delta/m),  m odd
        v, m = 0, a
        while m % 2 == 0:
            v += 1
            m //= 2
        k2 = _kronecker_at_2(Delta)
        if k2 == 0:
            return 0
        km = int(sympy.jacobi_symbol(Delta, m)) if m > 1 else 1
        return (k2 ** v) * km
    else:
        return int(sympy.jacobi_symbol(Delta, a))


# ---------------------------------------------------------------------------
# Fundamental discriminant |Delta_K| = conductor q for Q(sqrt(d))
# ---------------------------------------------------------------------------

def fundamental_discriminant(d):
    """Return conductor q = d if d = 1 mod 4, else 4d."""
    return d if d % 4 == 1 else 4 * d


# ---------------------------------------------------------------------------
# L(s, chi_d) via the Hurwitz zeta decomposition (ARB ball arithmetic)
# ---------------------------------------------------------------------------

def L_hurwitz(s_arb, d):
    """
    Compute L(s, chi_d) = q^{-s} * sum_{a=1}^{q} chi_d(a) * zeta(s, a/q).
    Returns a certified arb ball.  Terms with chi_d(a) = 0 are skipped.
    """
    Delta = fundamental_discriminant(d)
    q     = Delta
    q_arb = arb(q)
    s_c   = acb(s_arb)

    total = arb(0)
    for a in range(1, q + 1):
        chi = kronecker_symbol(Delta, a)
        if chi == 0:
            continue
        shift = acb(arb(a)) / acb(q_arb)
        hz    = acb.zeta(s_c, shift).real   # real-valued for real s > 1
        total = total + arb(chi) * hz

    return total / q_arb ** s_arb


# ---------------------------------------------------------------------------
# L(1, chi_d) via the digamma formula (ARB ball arithmetic)
# L(1, chi) = -(1/q) * sum_{a=1}^{q} chi(a) * psi(a/q)
# ---------------------------------------------------------------------------

def L1_digamma(d):
    """
    Compute L(1, chi_d) via the digamma formula.  Returns a certified arb ball.
    """
    Delta = fundamental_discriminant(d)
    q     = Delta
    q_arb = arb(q)

    total = arb(0)
    for a in range(1, q + 1):
        chi = kronecker_symbol(Delta, a)
        if chi == 0:
            continue
        shift   = acb(arb(a)) / acb(q_arb)
        psi_val = acb.digamma(shift).real
        total   = total - arb(chi) * psi_val

    return total / q_arb


# ---------------------------------------------------------------------------
# G(s) = zeta(s) / L(s, chi_d)
# ---------------------------------------------------------------------------

def zeta_arb(s_arb):
    """Riemann zeta at real s > 1, as a certified arb ball."""
    return acb.zeta(acb(s_arb)).real


def G(s_arb, d):
    """G(s) = zeta(s) / L(s, chi_d), as a certified arb ball."""
    return zeta_arb(s_arb) / L_hurwitz(s_arb, d)


# ---------------------------------------------------------------------------
# Certified bracket check
# ---------------------------------------------------------------------------

def certify_bracket(s_lo, s_hi, target, d):
    """
    Certify G(s_lo) > target and G(s_hi) < target as ARB predicates.
    Raises ValueError if either comparison is undecidable.
    """
    g_lo = G(s_lo, d)
    g_hi = G(s_hi, d)

    if not (g_lo > target):
        raise ValueError(
            f"d={d}: lower bracket not certified: "
            f"G(s_lo)={g_lo}, target={target}."
        )
    if not (g_hi < target):
        raise ValueError(
            f"d={d}: upper bracket not certified: "
            f"G(s_hi)={g_hi}, target={target}."
        )


# ---------------------------------------------------------------------------
# Strategy A: decisive trichotomy bisection for irrational s_*(d)
# ---------------------------------------------------------------------------

# Brackets [s_lo, s_hi] of width ~1 surrounding s_*(d) for d != 3.
_BRACKETS = {
    2:  ("2.0", "3.5"),
    5:  ("1.5", "2.5"),
    7:  ("1.2", "1.7"),
    13: ("1.2", "1.7"),
}


def find_null_crossing_bisection(d, n_steps=220, max_prec_escalations=5):
    """
    Certify s_*(d) for d in {2, 5, 7, 13} by decisive ARB bisection.

    Returns (s_star, residual, n_escalations, prec_used).
    Raises RuntimeError if any certification step fails.
    """
    ctx.prec  = BASE_PREC
    target    = arb(d).sqrt()
    s_lo      = arb(_BRACKETS[d][0])
    s_hi      = arb(_BRACKETS[d][1])

    # Step 1: certify initial bracket
    certify_bracket(s_lo, s_hi, target, d)

    # Step 2: decisive trichotomy bisection
    n_escals = 0
    step     = 0

    while step < n_steps:
        s_mid = (s_lo + s_hi) * arb("0.5")
        g_mid = G(s_mid, d)

        if g_mid > target:
            s_lo  = s_mid
            step += 1
        elif g_mid < target:
            s_hi  = s_mid
            step += 1
        else:
            # Intervals overlap: undecidable at current precision.
            if n_escals >= max_prec_escalations:
                raise RuntimeError(
                    f"d={d}: bisection comparison undecidable after "
                    f"{max_prec_escalations} precision escalations "
                    f"(final prec={ctx.prec} bits)."
                )
            ctx.prec += PREC_STEP
            n_escals += 1
            # Recompute target at the new precision.  s_lo and s_hi remain
            # valid ARB enclosures at any precision: a ball certified at
            # lower precision is a valid (wider) enclosure at higher
            # precision, so no conversion or re-initialisation is needed.
            target = arb(d).sqrt()

    prec_used = ctx.prec

    # Step 3: certify L > 0 on the entire final bracket (denominator safety).
    # s_lo.union(s_hi) returns the smallest ARB ball enclosing both endpoints.
    # Evaluating L_hurwitz on this ball gives a certified enclosure of
    # L(s, chi_d) for ALL s in [s_lo, s_hi] — not merely at the two endpoints.
    # Positivity of the resulting ball certifies L > 0 on the whole interval.
    s_bracket   = s_lo.union(s_hi)
    L_on_bracket = L_hurwitz(s_bracket, d)
    if not (L_on_bracket > arb(0)):
        raise RuntimeError(
            f"d={d}: L_hurwitz not certified > 0 on final bracket. "
            f"L(union bracket)={L_on_bracket}."
        )

    # Step 4: certify residual < 1e-60 as ARB predicate
    s_star   = (s_lo + s_hi) * arb("0.5")
    residual = abs(G(s_star, d) - target)

    if not (residual < arb("1e-60")):
        raise RuntimeError(
            f"d={d}: residual not certified < 1e-60. residual={residual}."
        )

    ctx.prec = BASE_PREC
    return s_star, residual, n_escals, prec_used


# ---------------------------------------------------------------------------
# Strategy B: direct integer crossing for d = 3
#
# s_*(3) = 2 exactly (Theorem 9.2).  The bisection approach is structurally
# inapplicable: when the midpoint reaches 2, G(2) = sqrt(3) holds as an
# identity, so neither G > sqrt(3) nor G < sqrt(3) can be certified.
# Instead, three ARB predicates are checked directly:
#   (i)   G(1.99, 3) > sqrt(3)
#   (ii)  G(2.01, 3) < sqrt(3)
#   (iii) |G(2, 3) - sqrt(3)| < 1e-60
# With G strictly decreasing (Lemma 9.1), (i)+(ii) certify that s_*(3)
# lies in (1.99, 2.01), and (iii) certifies the evaluation at the point.
# ---------------------------------------------------------------------------

def certify_integer_crossing(d, s_int, epsilon="0.01"):
    """
    Certify s_*(d) = s_int (an integer) by direct ARB evaluation.

    Returns (s_star, residual, 0, ctx.prec).
    Raises RuntimeError if any of the three predicates fails.
    """
    ctx.prec = BASE_PREC
    target   = arb(d).sqrt()
    eps      = arb(epsilon)
    s_star   = arb(s_int)

    # (i) G strictly above target just below s_int
    if not (G(s_star - eps, d) > target):
        raise RuntimeError(
            f"d={d}: G(s_int - {epsilon}) not certified > target."
        )

    # (ii) G strictly below target just above s_int
    if not (G(s_star + eps, d) < target):
        raise RuntimeError(
            f"d={d}: G(s_int + {epsilon}) not certified < target."
        )

    # (iii) residual at the integer point
    residual = abs(G(s_star, d) - target)
    if not (residual < arb("1e-60")):
        raise RuntimeError(
            f"d={d}: residual at integer not certified < 1e-60. "
            f"residual={residual}."
        )

    return s_star, residual, 0, ctx.prec


# ---------------------------------------------------------------------------
# Unified entry point: dispatch to strategy A or B
# ---------------------------------------------------------------------------

def find_null_crossing(d, n_steps=220):
    """
    Certify s_*(d) using Strategy A (bisection) for d in {2, 5, 7, 13},
    or Strategy B (direct integer crossing) for d = 3.

    Returns (s_star, residual, n_escalations, prec_used).
    All returned values are ARB balls.  Raises on any certification failure.
    """
    if d == 3:
        return certify_integer_crossing(d, s_int=2)
    else:
        return find_null_crossing_bisection(d, n_steps=n_steps)


# ---------------------------------------------------------------------------
# ARB-native display string (for output only; not used in any predicate)
# ---------------------------------------------------------------------------

def arb_str(x, digits=10):
    old = ctx.prec
    ctx.prec = max(old, int(digits * 3.35) + 64)
    s = str(x)
    ctx.prec = old
    return s


# ---------------------------------------------------------------------------
# Main: reproduce Table 1 from Theorem 9.1, all predicates on ARB objects
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Paper table values, for display comparison only.
    paper_s_star = {2: 2.5635, 3: 2.0000, 5: 2.0492, 7: 1.4445, 13: 1.4608}
    paper_L1     = {2: 0.6232, 3: 0.7603, 5: 0.4304, 7: 1.0465, 13: 0.6627}

    print("Theorem 9.1: Null Crossing Table")
    print(f"ARB base precision : {BASE_PREC} bits (~{int(BASE_PREC*0.301)} decimal digits)")
    print("Certification      : decisive trichotomy bisection (d != 3);")
    print("                     direct integer-crossing check (d = 3)")
    print("Residual target    : < arb('1e-60'), evaluated as ARB predicate")
    print()
    print(f"{'d':>4}  {'Delta_K':>7}  {'L(1,chi_d)':>22}  {'s_*(d)':>22}  "
          f"{'residual':>14}  {'certified':>10}  {'strategy':>10}")
    print("-" * 102)

    ctx.prec    = BASE_PREC
    all_certified = True

    for d in [2, 3, 5, 7, 13]:
        Delta = fundamental_discriminant(d)

        # L(1, chi_d): certified ARB ball
        L1_ball = L1_digamma(d)

        # s_*(d): certified by the appropriate strategy
        s_star, residual, n_escals, prec_used = find_null_crossing(d)

        # --- All predicates evaluated on ARB objects ---
        res_ok = bool(residual < arb("1e-60"))
        L1_ok  = bool(abs(L1_ball - arb(paper_L1[d]))  < arb("5e-4"))
        s_ok   = bool(abs(s_star  - arb(paper_s_star[d])) < arb("5e-4"))
        ok     = res_ok and L1_ok and s_ok

        if not ok:
            all_certified = False

        strategy = "B (direct)" if d == 3 else "A (bisect)"
        note = (f"  [{n_escals} escalation(s), {prec_used}-bit]"
                if n_escals else "")

        print(f"{d:>4}  {Delta:>7}  "
              f"{float(L1_ball):>10.4f} (arb)  "
              f"{float(s_star):>10.4f} (arb)  "
              f"{float(residual):>14.2e}  "
              f"{'YES' if ok else 'FAIL':>10}  "
              f"{strategy:>10}"
              f"{note}")

    print()
    print(f"All predicates certified (ARB): {all_certified}")
    print()

    # -----------------------------------------------------------------------
    # Special case: d=3 integer crossing, extended display
    # -----------------------------------------------------------------------
    print("Special case: d=3, s_*(3) = 2 (unique integer null crossing)")
    ctx.prec = BASE_PREC
    G2_3  = G(arb(2), 3)
    sqrt3 = arb(3).sqrt()
    diff  = abs(G2_3 - sqrt3)
    print(f"  G(2, 3) (ARB ball)  = {arb_str(G2_3,  14)}")
    print(f"  sqrt(3) (ARB ball)  = {arb_str(sqrt3, 14)}")
    print(f"  |G(2,3)-sqrt(3)|    = {arb_str(diff,   6)}")
    print(f"  Certified < 1e-60   : {bool(diff < arb('1e-60'))}")
    print(f"  G(1.99,3) > sqrt(3) : {bool(G(arb('1.99'),3) > sqrt3)}")
    print(f"  G(2.01,3) < sqrt(3) : {bool(G(arb('2.01'),3) < sqrt3)}")
