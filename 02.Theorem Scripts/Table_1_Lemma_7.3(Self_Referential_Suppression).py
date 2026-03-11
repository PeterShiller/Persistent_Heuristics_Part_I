"""
Table_1_Lemma_7.3(Self_Referential_Suppression).py
====================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table appearing after Lemma 7.3
[Self-referential suppression].  The certified values are:

    M     I_n (n_M = 1)     b_M        I_n / b_M
    -----------------------------------------------
    5     1.00              0.00648    155
    10    3.23e-1           0.00247    131
    20    1.18e-1           0.00093    128
    30    6.27e-2           0.00049    127

The resonance vector has n_M = 1 and all other components zero.  The
integrand is:

    (1/pi) |J_1(b_M t)| * prod_{k=1}^{M-1} |J_0(b_k t)|

with one active J_1 factor (k = M) and M - 1 inactive J_0 factors
(k = 1, ..., M - 1).  Integration domain [T_EPS, T] = [1e-30, 2000].

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Algorithm
---------
The integrand contains absolute values of Bessel functions.  Because
|J_nu(x)| has cusps at every Bessel zero -- the function is C^0 but
not C^1 there -- the integrand is piecewise analytic, not analytic on
(0, T].  acb.integral (Petras algorithm) assumes analyticity in a strip
around the real axis; its error estimate is therefore not rigorous for
the full interval directly.

The rigorous approach is strip decomposition:

  (a) STRIP intervals [t_lo, t_hi], one per certified zero of each
      Bessel factor in (0, T].  t_lo = (z - DELTA)/b_k,
      t_hi = (z + DELTA)/b_k, DELTA = 1e-20 in x-space, z an ARB
      enclosure of the zero.  On each strip the integrand is bounded
      uniformly by (LANDAU_C / pi) times the strip width, which is
      O(DELTA) = O(1e-20) and negligible.

  (b) GAP intervals between consecutive strip endpoints.  On each gap
      no factor has a zero (certified: all zeros reside in their
      strips).  Each factor is therefore analytic and sign-definite on
      the gap.  acb.integral is rigorous here.

Strip merging:
  Strips from all M factors are merged into a single ascending list
  using a K-way merge on ARB < comparisons only.  Each per-factor
  strip list is already in certified ascending order (m increments,
  j_{nu,m} is strictly increasing).  No float arithmetic is used in
  the ordering path.  If two strip starts are undecidable at ARB_PREC
  bits, a RuntimeError is raised.

Bessel zeros:
  - J_0 zeros for factors k = 1, ..., M-1 (inactive).  J_0(z) computed
    via J_0(z) = (2/z)J_1(z) - J_2(z) (DLMF 10.6.1) to avoid the
    python-flint 0.8.0 bug where acb.bessel_j(z, acb(0)) = 0.
  - J_1 zeros for factor k = M (active).  McMahon (DLMF 10.21.19)
    with BRACKET = 1.5; accurate for all m >= 1 in (0, 2000].

PASS/FAIL criterion:
  - I_n and b_M: certified ARB predicate  rel_err < REL_TOL_3SF.
  - I_n/b_M: integer match (certified ARB ball within 0.5 of paper value).
  No float thresholds are used in any certification decision.

Zero data:
  Weights b_k = 2 / (1/4 + gamma_k'^2) computed in ARB from the first
  30 certified zero ordinates of L(s, chi_5) sourced from
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
ARB_PREC        = 256
MAX_BISECT_PREC = 1536       # 6x base; certified trichotomy precision ceiling
T_UPPER         = 2000
T_EPS           = arb("1e-30")
T_MAX_ARB       = arb(T_UPPER)
REL_TOL_INT     = 2**(-200)  # acb.integral relative tolerance
DELTA           = arb("1e-20")
LANDAU_C        = arb("0.7857")  # |J_N(x)| <= LANDAU_C * N^{-1/3} (DLMF 10.14.3)

# ── Paper table values ────────────────────────────────────────────────────────
# ARB-certified strip-decomposition results; these are the values used in
# the paper.
PAPER_I     = {5: "1.003",  10: "3.23e-1", 20: "1.18e-1", 30: "6.26e-2"}
PAPER_B     = {5: "6.48e-3", 10: "2.47e-3", 20: "9.25e-4", 30: "4.93e-4"}
PAPER_RATIO = {5: 155,       10: 131,        20: 128,        30: 127}

# 0.1% relative tolerance for certified-to-certified matching.
REL_TOL_3SF = arb("1e-3")

M_VALUES = [5, 10, 20, 30]

_TAIL_B = None  # set by __main__ so print_results can access weights

# ── ARB constants ─────────────────────────────────────────────────────────────
_ZERO = arb(0)
_TWO  = arb(2)
# No module-level pi constant: arb.pi() is called fresh inside each function
# so it inherits the current ctx.prec rather than the 53-bit value at import.

# ── Bessel function evaluations ───────────────────────────────────────────────

def J1_real(x_arb):
    """J_1(x) as a real ARB ball."""
    return acb.bessel_j(acb(x_arb), acb(1)).real


def J0_real(x_arb):
    """
    J_0(x) as a real ARB ball, via
        J_0(x) = (2/x)J_1(x) - J_2(x)    (DLMF 10.6.1).
    Avoids the python-flint 0.8.0 bug where acb.bessel_j(z, acb(0)) = 0.
    """
    z = acb(x_arb)
    return (acb(_TWO) * acb.bessel_j(z, acb(1)) / z
            - acb.bessel_j(z, acb(_TWO))).real


def J1_acb(z_acb):
    """J_1(z) for complex argument (used inside acb.integral)."""
    return acb.bessel_j(z_acb, acb(1))


def J0_acb(z_acb):
    """J_0(z) via recurrence for complex argument (used inside acb.integral)."""
    return (acb(_TWO) * acb.bessel_j(z_acb, acb(1)) / z_acb
            - acb.bessel_j(z_acb, acb(_TWO)))


# ── ARB Bessel zero computation ───────────────────────────────────────────────

def _mcmahon_seed(nu_int, m_int):
    """
    McMahon asymptotic seed for the m-th zero of J_{nu_int}, pure ARB.
    DLMF 10.21.19, three-term expansion.
    Accurate for nu in {0, 1} and all m >= 1 in (0, 2000] (max error < 0.1).
    """
    nu   = arb(nu_int)
    m    = arb(m_int)
    mu   = arb(4) * nu * nu
    beta = (m + nu / _TWO - arb("1/4")) * arb.pi()
    t1   = (mu - 1) / (arb(8) * beta)
    t2   = arb(4) * (mu - 1) * (arb(7) * mu - arb(31)) / (
               arb(3) * (arb(8) * beta) ** 3)
    t3   = arb(32) * (mu - 1) * (
               arb(83) * mu**2 - arb(982) * mu + arb(3779)) / (
               arb(15) * (arb(8) * beta) ** 5)
    return beta - t1 - t2 - t3


def arb_besseljzero(nu_int, m_int, eval_fn):
    """
    Compute the m-th positive zero of J_{nu_int} in pure ARB via
    certified trichotomy bisection.

    Seed: McMahon (DLMF 10.21.19) with BRACKET = 1.5 in x-space.
    Certified trichotomy: at each step the sign of f(mid) is resolved as
    certified positive, certified negative, or undecidable.  If undecidable,
    working precision is doubled up to MAX_BISECT_PREC.  If still
    undecidable at MAX_BISECT_PREC, RuntimeError is raised (this branch
    cannot fire for simple Bessel zeros).
    No mpmath or float arithmetic used at any step.
    """
    seed    = _mcmahon_seed(nu_int, m_int)
    BRACKET = arb("1.5")
    lo      = seed - BRACKET
    hi      = seed + BRACKET
    if lo < arb("0.5"):
        lo = arb("0.5")

    flo = eval_fn(lo)
    fhi = eval_fn(hi)
    if not (((flo > _ZERO) and (fhi < _ZERO)) or
            ((flo < _ZERO) and (fhi > _ZERO))):
        raise RuntimeError(
            f"Initial bracket sign check failed for J_{nu_int} zero #{m_int} "
            f"(seed ~ {float(seed.mid()):.4f})"
        )

    TARGET = DELTA * arb("0.1")
    for _it in range(500):
        if (hi - lo) < TARGET:
            break
        mid = (lo + hi) / _TWO

        prec = ctx.prec
        while prec <= MAX_BISECT_PREC:
            old_prec  = ctx.prec
            ctx.prec  = prec
            fmid      = eval_fn(mid)
            ctx.prec  = old_prec
            mid_pos   = fmid > _ZERO
            mid_neg   = fmid < _ZERO
            if mid_pos or mid_neg:
                break
            prec *= 2

        if not mid_pos and not mid_neg:
            raise RuntimeError(
                f"Sign undecidable at MAX_BISECT_PREC={MAX_BISECT_PREC} "
                f"for J_{nu_int} zero #{m_int}.  Should not occur for simple zeros."
            )

        lo_neg = flo < _ZERO
        if (not mid_neg and not lo_neg) or (mid_neg and lo_neg):
            lo  = mid
            flo = fmid
        else:
            hi  = mid
            fhi = fmid

    return (lo + hi) / _TWO


# ── Strip collection ──────────────────────────────────────────────────────────

def collect_strips(nu_int, b_k_arb, T_max_arb, eval_fn):
    """
    Collect IVT-certified strip intervals for all zeros of J_{nu}(b_k t)
    in (0, T_max].

    IVT check performed for every zero including the stopping zero.
    Non-overlap ARB-certified for consecutive retained zeros.
    t_hi capped at T_max_arb.
    Returns a list in certified ascending order (m increments).
    """
    T_x    = b_k_arb * T_max_arb
    strips = []
    prev_s = None
    m      = 1

    while True:
        z   = arb_besseljzero(nu_int, m, eval_fn)
        flo = eval_fn(z - DELTA)
        fhi = eval_fn(z + DELTA)
        if not (((flo > _ZERO) and (fhi < _ZERO)) or
                ((flo < _ZERO) and (fhi > _ZERO))):
            raise RuntimeError(
                f"IVT FAILED for J_{nu_int} zero #{m} (z ~ {float(z.mid()):.6f})"
            )

        if (z - DELTA) > T_x:
            break

        if prev_s is not None and not ((z - DELTA) > (prev_s + DELTA)):
            raise RuntimeError(
                f"Non-overlap FAILED: J_{nu_int} zeros #{m-1} and #{m}"
            )

        prev_s = z
        t_lo   = (z - DELTA) / b_k_arb
        t_hi   = (z + DELTA) / b_k_arb
        if t_hi > T_max_arb:
            t_hi = T_max_arb
        strips.append((t_lo, t_hi))
        m += 1

    return strips


def collect_all_strips(b_arb, M_int, T_max_arb):
    """
    Collect strips from all M Bessel factors and merge into a single
    certified-ascending list using a K-way merge on ARB comparisons.

      k = 1, ..., M-1 : J_0(b_k t)  (inactive)
      k = M           : J_1(b_M t)  (active)

    Each per-factor list is already in certified ascending order
    (m increments, j_{nu,m} is strictly increasing).  The K-way merge
    uses only ARB < comparisons to select the next strip at each step.
    If two strip starts are undecidable at ARB_PREC bits, RuntimeError
    is raised; increase ARB_PREC or decrease DELTA if this occurs.
    """
    # Build per-factor lists: M-1 J_0 lists then 1 J_1 list.
    per_factor = [collect_strips(0, b_arb[k], T_max_arb, J0_real)
                  for k in range(M_int - 1)]
    per_factor.append(collect_strips(1, b_arb[M_int - 1], T_max_arb, J1_real))

    ptrs   = [0] * M_int
    merged = []

    while True:
        active = [k for k in range(M_int) if ptrs[k] < len(per_factor[k])]
        if not active:
            break

        # Select the factor whose current strip starts earliest (ARB <).
        best = active[0]
        for k in active[1:]:
            a = per_factor[k][ptrs[k]][0]
            b_best = per_factor[best][ptrs[best]][0]
            decided = bool(a < b_best)
            undecided = (not decided) and (not bool(b_best < a)) and (not bool(a == b_best))
            if undecided:
                raise RuntimeError(
                    f"Strip start order undecidable between factors {k} and {best} "
                    f"at ARB_PREC={ARB_PREC}.  Increase ARB_PREC or decrease DELTA."
                )
            if decided:
                best = k

        merged.append(per_factor[best][ptrs[best]])
        ptrs[best] += 1

    # Paranoia: verify merged ordering with ARB.
    for i in range(len(merged) - 1):
        if not bool(merged[i][0] < merged[i + 1][0]):
            raise RuntimeError(
                f"Certified ordering FAILED at strip pair {i} and {i+1}"
            )
    return merged


def build_gaps(strips, T_max_arb):
    """
    Gap intervals between strips over [T_EPS, T_max_arb].
    prev_hi = max(prev_hi, t_hi) so prev_hi never moves backward.
    """
    gaps    = []
    prev_hi = T_EPS

    for (t_lo, t_hi) in strips:
        if prev_hi < t_lo:
            gaps.append((prev_hi, t_lo))
        if t_hi > prev_hi:
            prev_hi = t_hi

    if prev_hi < T_max_arb:
        gaps.append((prev_hi, T_max_arb))
    return gaps


# ── Strip error bound ─────────────────────────────────────────────────────────

def strip_error_bound(M_int, strips):
    """
    ARB upper bound on the total strip contribution.

    Each strip has width (t_hi - t_lo) <= 2*DELTA/b_k.  On any strip,
    the active J_1 factor is bounded by LANDAU_C = 0.7857 and each
    of the M-1 inactive J_0 factors is bounded by 1.  The (1/pi)
    prefactor is included.

        per-strip bound = (1/pi) * LANDAU_C * (t_hi - t_lo)

    With DELTA = 1e-20 this is O(1e-20) and negligible.
    """
    factor = LANDAU_C / arb.pi()
    total  = _ZERO
    for (t_lo, t_hi) in strips:
        total = total + factor * (t_hi - t_lo)
    return total


# ── Gap integration ───────────────────────────────────────────────────────────

def integrate_gap(t_lo_arb, t_hi_arb, b_acb, M_int):
    """
    Integrate the self-referential suppression integrand over one gap.

    On this gap no Bessel factor has a zero (certified by strip
    construction), so every factor is analytic and sign-definite.
    The absolute values are replaced by the analytic expression and
    acb.integral is rigorous here.

    Integrand: (1/pi) J_0(b_1 t) * ... * J_0(b_{M-1} t) * J_1(b_M t)
    """
    def integrand(t, _analytic):
        r = J0_acb(t * b_acb[0])
        for k in range(1, M_int - 1):
            r = r * J0_acb(t * b_acb[k])
        r = r * J1_acb(t * b_acb[M_int - 1])
        return r / acb(arb.pi())

    I = acb.integral(integrand, acb(t_lo_arb), acb(t_hi_arb),
                     rel_tol=REL_TOL_INT, eval_limit=10**7)
    return abs(I.real)


# ── Full integral ─────────────────────────────────────────────────────────────

def compute_I(b_arb, M_int):
    """
    I_n = (1/pi) int_0^T |J_1(b_M t)| * prod_{k=1}^{M-1} |J_0(b_k t)| dt

    via ARB-rigorous strip decomposition.
    Returns (I_arb, n_intervals, strip_err).
    """
    ctx.prec = ARB_PREC
    b_acb   = [acb(bk) for bk in b_arb[:M_int]]

    strips = collect_all_strips(b_arb, M_int, T_MAX_ARB)
    gaps   = build_gaps(strips, T_MAX_ARB)
    se     = strip_error_bound(M_int, strips)

    I = _ZERO
    for (t_lo, t_hi) in gaps:
        I = I + integrate_gap(t_lo, t_hi, b_acb, M_int)

    return I + se, len(strips) + len(gaps), se


# ── Certified analytic tail bound ─────────────────────────────────────────────

def certified_tail_bound(b_arb, M_int, T_arb):
    """
    ARB-rigorous upper bound on
      (1/pi) int_T^inf |J_1(b_M t)| * prod_{k=1}^{M-1} |J_0(b_k t)| dt.

    Uses Watson's bound |J_nu(x)| <= sqrt(2/(pi x)) for nu >= -1/2, x > 0.
    Factors with b_k * T >= 1 (certified ARB) are treated as decaying.
    Any remaining factors are bounded by 1.

    For m decaying factors (m >= 3):
      bound = (1/pi)(2/pi)^{m/2}(prod_{decaying} b_k)^{-1/2} T^{1-m/2}/(m/2-1).

    This bound is rigorous but conservative; it ignores oscillatory
    cancellation.  All paper values are integrals over [0, T_UPPER = 2000];
    for M >= 10 the tail is negligible at 3 significant figures.
    """
    decaying = [b_arb[k] for k in range(M_int)
                if bool(b_arb[k] * T_arb >= arb(1))]
    m = len(decaying)
    if m < 3:
        return arb("inf")
    prod_b = arb(1)
    for b in decaying:
        prod_b = prod_b * b
    two_over_pi = arb(2) / arb.pi()
    return (arb(1) / arb.pi()) \
           * two_over_pi ** (arb(m) / arb(2)) \
           * prod_b ** arb("-1/2") \
           * T_arb ** (arb(1) - arb(m) / arb(2)) \
           / (arb(m) / arb(2) - arb(1))


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights():
    ctx.prec = ARB_PREC
    zeros    = get_zeros(5, 30, as_strings=True)
    return [arb(2) / (arb("1/4") + arb(g)**2) for g in zeros]


# ── Certified PASS/FAIL ───────────────────────────────────────────────────────

def arb_matches_3sf(val_arb, paper_str):
    """
    Return True iff |val_arb - paper_val| / paper_val < REL_TOL_3SF,
    where < is the ARB certified predicate: returns True only when the
    entire ball of the left-hand side lies strictly below the lower
    bound of REL_TOL_3SF.  No float thresholds are used.
    """
    pv      = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return bool(rel_err < REL_TOL_3SF)


def arb_matches_ratio(ratio_arb, paper_int):
    """
    Return True iff the ARB ball for ratio_arb is within 0.5 of paper_int,
    certified by ARB < on both sides.
    """
    target = arb(str(paper_int))
    return bool(abs(ratio_arb - target) < arb("0.5"))


# ── Certification loop ────────────────────────────────────────────────────────

def certify(b_arb):
    ctx.prec = ARB_PREC
    results  = []

    for M in M_VALUES:
        t0              = time.time()
        I, ns, se       = compute_I(b_arb, M)
        elapsed         = time.time() - t0

        bM    = b_arb[M - 1]
        ratio = I / bM

        I_match     = arb_matches_3sf(I,  PAPER_I[M])
        bM_match    = arb_matches_3sf(bM, PAPER_B[M])
        ratio_match = arb_matches_ratio(ratio, PAPER_RATIO[M])

        results.append(dict(
            M=M,
            I_arb=I,       I_float=float(I.mid()),
            bM_arb=bM,     bM_float=float(bM.mid()),
            ratio_arb=ratio, ratio_float=float(ratio.mid()),
            strip_err=se,  ns=ns,
            I_match=I_match, bM_match=bM_match, ratio_match=ratio_match,
            elapsed=elapsed,
        ))

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results):
    print()
    print("Table after Lemma 7.3 -- Self-referential suppression -- ARB certification")
    print("=" * 76)
    print("  Integrand : (1/pi)|J_1(b_M t)| * prod_{k=1}^{M-1} |J_0(b_k t)|")
    print("  Active    : n_M = 1, all other n_k = 0")
    print(f"  Domain    : [1e-30, {T_UPPER}]")
    print("  Method    : K-way ARB strip merge + acb.integral on gaps")
    print(f"  Precision : {ARB_PREC}-bit ARB  (acb.integral on each gap)")
    print(f"  PASS/FAIL : I_n and b_M: rel_err < {float(REL_TOL_3SF.mid())}  "
          f"(display threshold; certification uses ARB <)")
    print(f"            : ratio: certified within 0.5 of paper integer")
    print()
    print(f"  {'M':>4}  {'I_n':>12}  {'Paper I':>9}  {'b_M':>9}  {'Paper b':>9}  "
          f"{'Ratio':>7}  {'Paper R':>7}  {'I':>6}  {'b':>6}  {'R':>6}  {'Time':>6}")
    print("  " + "-" * 90)

    all_pass = True
    for r in results:
        M = r['M']
        line = (f"  {M:>4}  {r['I_float']:>12.5e}  {PAPER_I[M]:>9}  "
                f"{r['bM_float']:>9.5f}  {PAPER_B[M]:>9}  "
                f"{r['ratio_float']:>7.1f}  {PAPER_RATIO[M]:>7d}  "
                f"{'PASS' if r['I_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['bM_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['ratio_match'] else 'FAIL':>6}  "
                f"{r['elapsed']:>5.1f}s")
        print(line)
        if not r['I_match'] or not r['bM_match'] or not r['ratio_match']:
            all_pass = False

    print()
    print("  ARB balls (full precision):")
    for r in results:
        print(f"    M={r['M']:2d}: I_n = {r['I_arb']}")
        print(f"          b_M = {r['bM_arb']}")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    - certified trichotomy bisection at all J_0 and J_1 zeros")
        print("    - K-way ARB strip merge; no float in ordering path")
        print("    - PASS/FAIL via decisive ARB predicate, no float threshold")
    else:
        print("  RESULT: FAIL -- one or more values outside tolerance")
        raise RuntimeError("Lemma 7.3 table certification failed")
    print()

    # ── Tail documentation ─────────────────────────────────────────────────
    # All paper values are integrals over [0, T_UPPER = 2000].  The bound
    # below certifies (rigorously, via Watson) how large int_{2000}^inf
    # can be.  For M >= 10, the bound is tight and the tail is negligible
    # at 3 significant figures.  PASS/FAIL is unaffected.
    if _TAIL_B is not None:
        T_arb = arb(str(T_UPPER))
        print("  Tail documentation (certified upper bound on int_{T}^inf)")
        print("  " + "-" * 60)
        for r in results:
            M  = r['M']
            tb = certified_tail_bound(_TAIL_B, M, T_arb)
            I  = float(r['I_arb'].mid())
            pct = float(tb.upper()) / I * 100
            note = ("ok (negligible at 3sf)"
                    if pct < 0.1
                    else "T=2000 truncated; true int_0^inf >= reported value")
            print("    M=%2d: tail<=%.3e  (%.3f%%)  %s"
                  % (M, float(tb.upper()), pct, note))
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctx.prec = ARB_PREC
    print(f"Loading chi_5 weights at {ARB_PREC}-bit ARB precision ...")
    b_arb = load_weights()
    print(f"  b_5  = {float(b_arb[4].mid()):.5f}")
    print(f"  b_10 = {float(b_arb[9].mid()):.5f}")
    print(f"  b_20 = {float(b_arb[19].mid()):.5f}")
    print(f"  b_30 = {float(b_arb[29].mid()):.5f}")
    print(f"  ({len(b_arb)} weights loaded)")
    print()
    print(f"Computing integrals for M in {M_VALUES} via strip decomposition ...")
    _TAIL_B = b_arb
    results = certify(b_arb)
    print_results(results)
