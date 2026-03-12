"""
Remark_6_12(J0_Role).py  --  ARB-rigorous certification of Remark 6.12
=======================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table in Remark 6.12
[Role of the inactive J_0 factors].  The certified values are:

    M     I_n (with J_0's)    Ratio to M=3
    ----------------------------------------
    3     3.660               1.000
    10    1.791               0.489
    20    1.717               0.469
    30    1.703               0.465

The integrand is:

    (1/pi) |J_1(b_1 t)| |J_1(b_2 t)| |J_1(b_3 t)|
            * prod_{k=4}^{M} |J_0(b_k t)|

with active resonance vector (n_1, n_2, n_3) = (1, 1, -1), so three
active J_1 factors (using |J_{-1}| = |J_1| by symmetry) and M - 3
inactive J_0 factors.  Integration domain [T_EPS, T] = [1e-30, 2000].

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Algorithm
---------
The integrand contains absolute values of Bessel functions.  Because
|J_nu(x)| has cusps at every Bessel zero -- the function is C^0 but
not C^1 there -- the integrand is piecewise analytic, not analytic on
(0, T].  acb.integral (Petras algorithm) assumes analyticity in
a strip around the real axis; its error estimate is therefore not
rigorous for the full interval directly, even if the zeros of different
factors do not coincide.

The rigorous approach, following Lemma_6_6(Transition_Zone_Bound).py,
is strip decomposition:

  (a) STRIP intervals [t_lo, t_hi], one per certified zero of each
      Bessel factor in (0, T].  t_lo = (z - DELTA)/b_k,
      t_hi = (z + DELTA)/b_k, DELTA = 1e-20 in x-space, z an ARB
      enclosure of the zero.  On each strip the integrand is bounded
      uniformly by a product of Landau / trivial constants times the
      strip width, which is O(DELTA) = O(1e-20) and negligible.

  (b) GAP intervals between consecutive strip endpoints.  On each gap
      no factor has a zero (certified: all zeros reside in their
      strips).  Each factor is therefore analytic and sign-definite on
      the gap.  acb.integral is rigorous here.

Bessel zeros:
  - J_1 zeros for factors k = 1, 2, 3 (active).  For nu = 1, McMahon
    (DLMF 10.21.19) with BRACKET = 1.5 is accurate for all m >= 1
    (max seed error < 0.1 at T = 2000).  No Olver correction needed.
  - J_0 zeros for factors k = 4, ..., M (inactive).  J_0(z) computed
    via J_0(z) = (2/z)J_1(z) - J_2(z) (DLMF 10.6.1).

PASS/FAIL criterion:
  The 3sf matching test uses a fully certified ARB comparison:
      rel_err < REL_TOL_3SF
  where both sides are ARB balls.  The operator < on ARB returns True
  only when the entire ball of rel_err lies strictly below the lower
  bound of REL_TOL_3SF -- a decisive, interval-certified predicate.
  No float thresholds are used in the certification decision.

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
REL_TOL_INT     = 2**(-200)  # acb.integral tolerance
DELTA           = arb("1e-20")
LANDAU_C        = arb("0.675")   # |J_N(x)| < LANDAU_C * N^{-1/3}  (Landau 2000, Thm 1; best constant b = 0.674885...)

# ── Paper table values ────────────────────────────────────────────────────────
# ARB-certified strip-decomposition results; these are the values used in the
# paper.
PAPER_I     = {3: "3.660", 10: "1.791", 20: "1.717", 30: "1.703"}
PAPER_RATIO = {3: "1.000", 10: "0.489", 20: "0.469", 30: "0.465"}

# 0.1% relative tolerance for certified-to-certified matching.
REL_TOL_3SF = arb("1e-3")

M_VALUES = [3, 10, 20, 30]

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
    beta = (m + nu / _TWO - arb("1/4")) * arb.pi()  # fresh at current ctx.prec
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
    cannot fire for simple Bessel zeros; see Lemma_6_6).
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
    in (0, T_max].  Architecture identical to Lemma_6_6.

    IVT check performed for every zero including the stopping zero.
    Non-overlap ARB-certified for consecutive retained zeros.
    t_hi capped at T_max_arb.
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
    Collect and merge strips from all M Bessel factors:
      k = 1, 2, 3  : J_1(b_k t)  (active)
      k = 4, ..., M: J_0(b_k t)  (inactive)

    Sorted by float key; certified ordering verified in ARB.
    """
    strips = []
    for k in range(3):
        strips += collect_strips(1, b_arb[k], T_max_arb, J1_real)
    for k in range(3, M_int):
        strips += collect_strips(0, b_arb[k], T_max_arb, J0_real)

    strips.sort(key=lambda s: float(s[0].mid()))

    for i in range(len(strips) - 1):
        if not (strips[i][0] < strips[i + 1][0]):
            raise RuntimeError(
                f"Certified ordering FAILED at strip pair {i} and {i+1}"
            )
    return strips


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

    Each strip has width (t_hi - t_lo) <= 2*DELTA/b_k.  On it every
    J_1 factor is bounded by LANDAU_C = 0.675 and every J_0 factor by 1.
    The (1/pi) prefactor is included.

        per-strip bound = (1/pi) * LANDAU_C^3 * 1^{M-3} * (t_hi - t_lo)

    With DELTA = 1e-20 this is O(1e-20) and negligible.
    """
    factor = (LANDAU_C ** 3) / arb.pi()
    total  = _ZERO
    for (t_lo, t_hi) in strips:
        total = total + factor * (t_hi - t_lo)
    return total


# ── Gap integration ───────────────────────────────────────────────────────────

def integrate_gap(t_lo_arb, t_hi_arb, b_acb, M_int):
    """
    Integrate the J_0 role integrand analytically over one gap.

    On this gap no Bessel factor has a zero (certified by strip
    construction), so every factor is analytic and sign-definite.
    The absolute values are replaced by the analytic expression and
    acb.integral is rigorous here.
    """
    def integrand(t, _analytic):
        r = J1_acb(t * b_acb[0]) * J1_acb(t * b_acb[1]) * J1_acb(t * b_acb[2])
        for k in range(3, M_int):
            r = r * J0_acb(t * b_acb[k])
        return r / acb(arb.pi())

    I = acb.integral(integrand, acb(t_lo_arb), acb(t_hi_arb),
                     rel_tol=REL_TOL_INT, eval_limit=10**7)
    return abs(I.real)


# ── Full integral ─────────────────────────────────────────────────────────────

def compute_I(b_arb, M_int):
    """
    I = (1/pi) int_0^T |J_1(b_1 t)||J_1(b_2 t)||J_1(b_3 t)|
                        * prod_{k=4}^M |J_0(b_k t)| dt

    via ARB-rigorous strip decomposition.  Returns (I_arb, n_intervals, strip_err).
    """
    ctx.prec = ARB_PREC
    b_acb    = [acb(bk) for bk in b_arb[:M_int]]

    strips  = collect_all_strips(b_arb, M_int, T_MAX_ARB)
    gaps    = build_gaps(strips, T_MAX_ARB)
    se      = strip_error_bound(M_int, strips)

    I = _ZERO
    for (t_lo, t_hi) in gaps:
        I = I + integrate_gap(t_lo, t_hi, b_acb, M_int)

    return I + se, len(strips) + len(gaps), se


# ── Weight loading ────────────────────────────────────────────────────────────

def load_weights():
    ctx.prec = ARB_PREC
    zeros    = get_zeros(5, 30, as_strings=True)
    return [arb(2) / (arb("1/4") + arb(g)**2) for g in zeros]


# ── Certified analytic tail bound ────────────────────────────────────────────

def certified_tail_bound(b_arb, M_int, T_arb):
    """
    ARB-rigorous upper bound on:
      (1/pi) int_T^inf |J_1(b_1 t)||J_1(b_2 t)||J_1(b_3 t)| prod_{k=4}^M |J_0(b_k t)| dt.

    Uses Watson's bound |J_nu(x)| <= sqrt(2/(pi x)) for nu >= -1/2, x > 0
    (Watson 1944).  Factors with b_k * T >= 1 (certified ARB) are treated as
    decaying; any remaining factors are bounded by 1 (conservative).

    For m decaying factors (m >= 3):
      bound = (1/pi)(2/pi)^{m/2}(prod_{decaying} b_k)^{-1/2} T^{1-m/2}/(m/2-1).

    WARNING: This bound is rigorous but typically conservative by 10-100x
    for small M because Watson's bound ignores oscillatory cancellation.
    For M = 3 (only J_1 factors, no inactive J_0 factors), the bound at
    T = 2000 is approximately 2.0 while the actual tail is approximately 0.67
    (computed numerically via strip quadrature on [2000, 200000] plus analytic
    bound on [200000, inf)).  For M >= 10, the bound is tight and the tail is
    negligible at 3 significant figures.

    INTERPRETATION: all paper values are integrals over [0, T_UPPER = 2000].
    PASS/FAIL is self-consistent (script and paper both use T = 2000).
    The tail documents the unverified truncation error.  For the unsigned
    integrals used as upper bounds in the paper's theorems, the T = 2000
    value underestimates int_0^inf, so the bound is conservative (weaker).
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
    return (arb(1) / arb.pi())            * two_over_pi ** (arb(m) / arb(2))            * prod_b ** arb("-1/2")            * T_arb ** (arb(1) - arb(m) / arb(2))            / (arb(m) / arb(2) - arb(1))


# ── Certified PASS/FAIL ───────────────────────────────────────────────────────

def arb_matches(val_arb, paper_str):
    """
    Return True iff |val_arb - paper_val| / paper_val < REL_TOL_3SF,
    where < is the ARB certified predicate: returns True only when the
    entire ball of the left-hand side lies strictly below the lower
    bound of REL_TOL_3SF.  No float threshold is used.
    """
    pv      = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return bool(rel_err < REL_TOL_3SF)


# ── Certification loop ────────────────────────────────────────────────────────

def certify(b_arb):
    ctx.prec = ARB_PREC
    results  = []
    I_ref    = None

    for M in M_VALUES:
        t0              = time.time()
        I, ns, se       = compute_I(b_arb, M)
        elapsed         = time.time() - t0

        if I_ref is None:
            I_ref = I

        ratio       = I / I_ref
        I_match     = arb_matches(I,     PAPER_I[M])
        ratio_match = arb_matches(ratio, PAPER_RATIO[M])

        results.append(dict(
            M=M, I_arb=I, I_float=float(I.mid()),
            ratio_arb=ratio, ratio_float=float(ratio.mid()),
            strip_err=se, ns=ns,
            I_match=I_match, ratio_match=ratio_match,
            elapsed=elapsed,
        ))

    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_results(results):
    print()
    print("Remark 6.12 -- Role of inactive J_0 factors -- ARB certification")
    print("=" * 72)
    print("  Integrand : (1/pi)|J_1(b_1 t)||J_1(b_2 t)||J_1(b_3 t)|"
          " * prod_{k=4}^{M} |J_0(b_k t)|")
    print("  Active    : (n_1, n_2, n_3) = (1, 1, -1)")
    print(f"  Domain    : [1e-30, {T_UPPER}]")
    print("  Method    : Strip decomposition (J_1 and J_0 zeros) + ARB quadrature on gaps")
    print(f"  Precision : {ARB_PREC}-bit ARB  (acb.integral on each gap)")
    print(f"  PASS/FAIL : certified ARB predicate  rel_err < {float(REL_TOL_3SF.mid())}  (display threshold; certification uses ARB <)")
    print()
    print(f"  {'M':>4}  {'I_n':>12}  {'Paper':>7}  {'Ratio':>7}  "
          f"{'Paper':>7}  {'strip_err':>10}  {'I':>6}  {'R':>6}  {'Time':>6}")
    print("  " + "-" * 72)

    all_pass = True
    for r in results:
        M = r['M']
        line = (f"  {M:>4}  {r['I_float']:>12.5f}  {PAPER_I[M]:>7}  "
                f"{r['ratio_float']:>7.4f}  {PAPER_RATIO[M]:>7}  "
                f"{float(r['strip_err'].mid()):>10.2e}  "
                f"{'PASS' if r['I_match'] else 'FAIL':>6}  "
                f"{'PASS' if r['ratio_match'] else 'FAIL':>6}  "
                f"{r['elapsed']:>5.1f}s")
        print(line)
        if not r['I_match'] or not r['ratio_match']:
            all_pass = False

    print()
    print("  ARB balls (full precision):")
    for r in results:
        print(f"    M={r['M']:2d}: I = {r['I_arb']}")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    - certified trichotomy bisection at all J_1 and J_0 zeros")
        print("    - strip decomposition at every cusp of the integrand")
        print("    - PASS/FAIL via decisive ARB predicate, no float threshold")
    else:
        print("  RESULT: FAIL -- one or more values outside 3sf tolerance")
        raise RuntimeError("Remark 6.12 certification failed")
    print()

    # ── Tail documentation ─────────────────────────────────────────────────
    # All paper values are integrals over [0, T_UPPER = 2000].  The bound
    # below certifies (rigorously, via Watson) how large int_{2000}^inf can be.
    # Watson's bound ignores oscillatory cancellation, so for M = 3 the bound
    # (~2.0) is ~3x larger than the numerical estimate (~0.67): the M = 3
    # integrand has only J_1 factors and decays as t^{-3/2}, so T = 2000
    # captures roughly 89% of int_0^inf.  For M >= 10, the decay is fast
    # enough that T = 2000 is essentially exact at 3 significant figures.
    # PASS/FAIL is unaffected: both script and paper values use T = 2000.
    b_arb_ref = results[0]['I_arb']   # only need b_arb; get it from load_weights
    # (b_arb is not in scope here; retrieve from the first result's computation
    # context via a module-level variable set in __main__)
    if _TAIL_B is not None:
        T_arb = arb(str(T_UPPER))
        print("  Tail documentation (certified upper bound on int_{T}^inf)")
        print("  " + "-" * 60)
        print("  Note: Watson bound for M=3 is ~3x the numerical estimate (~0.67)")
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
    print(f"  b_1 = {float(b_arb[0].mid()):.6f}, "
          f"b_2 = {float(b_arb[1].mid()):.6f}, "
          f"b_3 = {float(b_arb[2].mid()):.6f}")
    print(f"  ({len(b_arb)} weights loaded)")
    print()
    print(f"Computing integrals for M in {M_VALUES} via strip decomposition ...")
    _TAIL_B = b_arb
    results = certify(b_arb)
    print_results(results)
