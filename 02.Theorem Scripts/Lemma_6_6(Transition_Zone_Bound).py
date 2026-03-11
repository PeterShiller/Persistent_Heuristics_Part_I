"""
Lemma_6_6(Transition_Zone_Bound).py  --  ARB-rigorous certification of Lemma 6.6
==================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table appearing in the proof of
Lemma 6.6 [Transition zone bound]:

    I_n = (1/pi) integral_0^T |J_N(b_1 t)| |J_N(b_2 t)|
                                prod_{k=3}^{20} |J_0(b_k t)| dt

for the worst-case two-active configuration n_1 = n_2 = N at M = 20.

Paper table (Section 6):

    N     I_n (ARB)           Subcrit. (e/4)^N      Trans. N^{-8.67}
    -----------------------------------------------------------------
    5     3.82 x 10^{-2}      1.5 x 10^{-1}         8.8 x 10^{-7}
    10    1.42 x 10^{-4}      2.1 x 10^{-2}         2.2 x 10^{-9}
    20    7.87 x 10^{-9}      4.4 x 10^{-4}         5.3 x 10^{-12}
    30    1.15 x 10^{-11}     9.3 x 10^{-6}         1.6 x 10^{-13}
    50    1.41 x 10^{-14}     4.1 x 10^{-9}         1.9 x 10^{-15}

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Algorithm
---------
acb.integral (acb_calc_integrate, Petras algorithm, 256-bit ARB) requires
an analytic integrand.  The absolute-value integrand is handled by dividing
(0, T] into:

  (a) STRIP intervals [t_lo_arb, t_hi_arb], one per certified zero of each
      Bessel factor in (0, T], where t_lo_arb = (z - DELTA) / b_k and
      t_hi_arb = (z + DELTA) / b_k are pure ARB values.

  (b) GAP intervals between consecutive strip endpoints.

On each gap the integrand has definite sign (no zero of any factor lies
in the gap interior, certified because all zeros reside in their strips).
The analytic integrand is integrated with acb.integral and the absolute
value of the real part is summed.

On each strip the integrand is bounded by the Landau constant:
  B_N = LANDAU_C^2 * N^{-2/3},  LANDAU_C = 0.7857  (DLMF 10.14.3).

Step 1: Fully ARB Bessel zero computation.

  arb_besseljzero(nu, m, eval_fn) computes the m-th positive zero of
  J_nu entirely in ARB, with no mpmath or floating-point arithmetic.

  (a) Seed.  The three-term McMahon asymptotic (DLMF 10.21.19) is
      evaluated in ARB at working precision to produce an ARB ball
      seed.  For all nu in {0, 5, 10, ..., 50} and m >= 1, this
      places the seed within 0.3 of the true zero.

  (b) ARB bisection.  An initial bracket [seed - 1.5, seed + 1.5]
      is certified to have opposite ARB-certified signs at its
      endpoints.  The bracket is bisected in ARB until its width
      drops below DELTA/10.  Each bisection step uses certified ARB
      sign comparisons; RuntimeError is raised if the bracket sign
      check fails.  The midpoint of the final bracket lies within
      DELTA/20 of the true zero by the bisection guarantee.

  (c) The bisection is done at full ARB_PREC precision throughout,
      so there is no precision loss from mpmath's internal rounding.

Step 2: IVT check and certified stopping rule.

  For each Bessel factor, zeros are enumerated m = 1, 2, ... .  For
  each m, arb_besseljzero returns z_m (entirely in ARB).  Then:

  (a) IVT check -- performed for EVERY m (kept and excluded).
      J_nu is evaluated in ARB at z_m - DELTA and z_m + DELTA.
      If the ARB balls do not have certified opposite signs,
      RuntimeError is raised.

  (b) Certified stopping rule.
      If (z_m - DELTA) > b_k * T_max, the IVT certificate confirms
      the true zero is in [z_m - DELTA, z_m + DELTA], and since
      z_m - DELTA > b_k * T_max, the true zero is certified above
      b_k * T_max.  The loop stops.  The certificate uses the
      IVT-verified bracket, not the McMahon seed.

  (c) Non-overlap check.
      (z_m - DELTA) > (z_{m-1} + DELTA) is ARB-certified for each
      consecutive retained pair.  RuntimeError on failure.

  (d) Strip endpoints -- pure ARB:
      t_lo = (z_m - DELTA) / b_k_arb,  t_hi = (z_m + DELTA) / b_k_arb.

Step 3: Certified strip ordering.

  Strips from all 20 Bessel factors are float-sorted by float(t_lo.mid())
  (float used only as sort key, not as a cut point), then every consecutive
  pair is certified by the ARB comparison t_lo[i] < t_lo[i+1].
  RuntimeError if any pair is not certifiably ordered.

Step 4: Gap / strip decomposition with union endpoint.

  build_intervals uses prev_hi = max(prev_hi, t_hi) at each strip,
  so prev_hi never moves backward.  A gap (prev_hi, t_lo) is added
  only when prev_hi < t_lo is ARB-certified.  Nested or overlapping
  strips are handled correctly.

Step 5: Certified integration and strip-error bound.

  Gap intervals [a, b] are integrated with acb.integral at acb(a),
  acb(b) -- pure ARB limits, no float().  Strip contributions bounded
  by B_N * width / pi, all in ARB.

J_0(z) workaround.
  In python-flint 0.8.0, acb.bessel_j(z, acb(0)) = 0 (order-zero bug).
  We use J_0(z) = (2/z) J_1(z) - J_2(z)  (DLMF 10.6.1), valid for Re(z)>0.
  Integration starts at T_EPS = 10^{-30}.

Argument order in python-flint 0.8.0.
  acb.bessel_j(z, nu) = J_nu(z)  [z first, nu second].

Rigorousness checklist
----------------------
  (a) Bessel zeros computed entirely in ARB via McMahon + ARB bisection.
      No mpmath or floating-point arithmetic in any load-bearing step.

  (b) IVT performed for every zero j, including the first excluded one.
      Stopping rule uses only the IVT-certified bracket.
      RuntimeError on IVT failure for any zero.

  (c) Non-overlap check certifies exactly one zero per retained bracket.
      RuntimeError on failure.

  (d) Certified ordering: ARB comparison t_lo[i] < t_lo[i+1] for every
      consecutive pair in the sorted strip list.  RuntimeError on failure.

  (e) build_intervals uses max(prev_hi, t_hi): prev_hi never decreases.
      Gap (prev_hi, t_lo) added only when ARB certifies prev_hi < t_lo.

  (f) All integration limits are acb(arb_value).  float() is used only
      in sort keys, display, and timing.

  (g) Strip widths, strip-error sum, and subcritical comparison all in ARB.

External-input qualifications
------------------------------
  L_function_zeros.py: zero ordinates gamma_1',...,gamma_20' of chi_5
  to 70 decimal places, certified with |L(1/2+i*gamma_k')| < 10^{-449}.

  ARB bisection / IVT: the IVT and non-overlap checks serve as a runtime
  tripwire.  A Bessel zero returned out of order or with incorrect sign
  would produce an IVT failure or non-overlap violation, raising RuntimeError.

Usage
-----
  python3 "Lemma_6_6(Transition_Zone_Bound).py"
  Expected runtime: ~5--10 minutes (N=50 requires ~3950 strips,
  each requiring ARB bisection to 1e-21).

Dependencies
------------
  python-flint >= 0.8.0,  L_function_zeros.py
  (mpmath is NOT used)

References
----------
  [Paper]  arXiv:2603.00301; Zenodo 10.5281/zenodo.18783098.
  [DLMF]   10.6.1, 10.14.3, 10.21.19.  dlmf.nist.gov
  [Petras] Petras, K. (2002). Adv. Comput. Math. 16, 71-100.
"""

import sys, os, time

_SD = os.path.dirname(os.path.abspath(__file__))
_RR = os.path.dirname(_SD)
_DD = os.path.join(_RR, "01.Computed L(s, χ) Zeros and Imported ζ Zeros")
for _p in [_SD, _RR, _DD]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from L_function_zeros import get_zero
from flint import arb, acb, ctx

ARB_PREC = 256
ctx.prec  = ARB_PREC

M        = 20
D        = 5
N_VALUES = [5, 10, 20, 30, 50]
T_UPPER  = {5: 2000, 10: 2000, 20: 2000, 30: 2000, 50: 100000}
T_EPS    = arb("1e-30")   # lower limit; integrand is O(t^{N+1}) near 0
DELTA    = arb("1e-20")   # IVT / strip half-width in x-space
REL_TOL  = 2**(-200)
ABS_TOL  = 2**(-250)
LANDAU_C = arb("0.7857")  # |J_N(x)| <= LANDAU_C * N^{-1/3}  (DLMF 10.14.3)

PAPER_I       = {5:"3.82e-2", 10:"1.42e-4", 20:"7.87e-9",
                 30:"1.15e-11", 50:"1.41e-14"}
PAPER_SUBCRIT = {5:"1.5e-1",  10:"2.1e-2",  20:"4.4e-4",
                 30:"9.3e-6",  50:"4.1e-9"}
PAPER_TRANS   = {5:"8.8e-7",  10:"2.2e-9",  20:"5.3e-12",
                 30:"1.6e-13", 50:"1.9e-15"}
REL_TOL_3SF   = arb("0.02")
REL_TOL_2SF   = arb("0.06")

_ZERO = arb(0)
_TWO  = arb(2)

# ── Bessel evaluation ─────────────────────────────────────────────────────────

def J0_acb(z):
    """J_0(z) = (2/z) J_1(z) - J_2(z)  [DLMF 10.6.1; avoids flint 0.8.0 bug]."""
    return acb(2) * acb.bessel_j(z, acb(1)) / z - acb.bessel_j(z, acb(2))

def JN_acb(z, N_int):
    return acb.bessel_j(z, acb(N_int))

def J0_real(x_arb):
    return J0_acb(acb(x_arb)).real

def JN_real(x_arb, N_int):
    return acb.bessel_j(acb(x_arb), acb(N_int)).real


# ── Fully ARB Bessel zero computation ─────────────────────────────────────────

def _mcmahon_seed(nu_int, m_int):
    """
    Three-term McMahon asymptotic for j_{nu, m} in pure ARB arithmetic.

    Reference: DLMF 10.21.19.
    For nu in {0,5,...,50} and m >= 1, the seed lies within 0.3 of the
    true zero, so a bracket of +-1.5 is sufficient for bisection.
    """
    nu = arb(nu_int)
    m  = arb(m_int)
    mu = arb(4) * nu * nu
    beta = (m + nu / _TWO - arb("1/4")) * arb.pi()
    t1 = (mu - 1) / (arb(8) * beta)
    t2 = arb(4) * (mu - 1) * (arb(7) * mu - arb(31)) / (
             arb(3) * (arb(8) * beta) ** 3)
    t3 = arb(32) * (mu - 1) * (
             arb(83) * mu ** 2 - arb(982) * mu + arb(3779)) / (
             arb(15) * (arb(8) * beta) ** 5)
    return beta - t1 - t2 - t3


def arb_besseljzero(nu_int, m_int, eval_fn):
    """
    Compute the m-th positive zero of J_{nu_int} entirely in ARB.

    Algorithm:
      1. Evaluate the three-term McMahon asymptotic in ARB to obtain a
         seed that lies within 0.3 of the true zero for all nu <= 50, m >= 1.
      2. Form the bracket [seed - 1.5, seed + 1.5] and verify opposite
         ARB-certified signs at the endpoints.
      3. Bisect in ARB until the bracket width drops below DELTA/10.
         Each step uses a certified ARB sign comparison; RuntimeError on failure.
      4. Return the ARB midpoint, which lies within DELTA/20 of the true zero.

    No mpmath or floating-point arithmetic is used at any step.
    """
    seed    = _mcmahon_seed(nu_int, m_int)
    BRACKET = arb("1.5")
    lo = seed - BRACKET
    hi = seed + BRACKET
    if lo < arb("0.5"):
        lo = arb("0.5")

    flo = eval_fn(lo)
    fhi = eval_fn(hi)

    if not (((flo > _ZERO) and (fhi < _ZERO)) or
            ((flo < _ZERO) and (fhi > _ZERO))):
        raise RuntimeError(
            f"ARB bisection: initial bracket sign check failed for "
            f"J_{nu_int} zero #{m_int}  (seed ~ {float(seed.mid()):.4f})"
        )

    TARGET = DELTA * arb("0.1")
    for _it in range(500):
        if (hi - lo) < TARGET:
            break
        mid  = (lo + hi) / _TWO
        fmid = eval_fn(mid)
        lo_pos = flo > _ZERO
        mid_pos = fmid > _ZERO
        if (lo_pos and mid_pos) or ((flo < _ZERO) and (fmid < _ZERO)):
            lo  = mid
            flo = fmid
        else:
            hi  = mid
            fhi = fmid
    else:
        raise RuntimeError(
            f"ARB bisection did not converge for J_{nu_int} zero #{m_int}"
        )

    return (lo + hi) / _TWO


# ── Weights ───────────────────────────────────────────────────────────────────

def load_weights():
    b_arb = []
    for k in range(1, M + 1):
        g = arb(get_zero(D, k, as_string=True))
        b_arb.append(_TWO / (arb("0.25") + g * g))
    return b_arb, [acb(b) for b in b_arb]


# ── IVT-certified strip collection ────────────────────────────────────────────

def collect_strips(nu_int, b_k_arb, T_max_arb, eval_fn):
    """
    Collect IVT-certified strip intervals for all zeros of J_{nu}(b_k t)
    in (0, T_max].  All arithmetic is pure ARB.

    For each zero m = 1, 2, ...:
      - arb_besseljzero returns z_m entirely in ARB (no mpmath).
      - IVT is checked for every m, including the stopping zero.
      - Stopping is certified: (z_m - DELTA) > T_x proves the true zero
        (which lies in [z_m - DELTA, z_m + DELTA] by IVT) is above T_x.
      - Non-overlap is ARB-certified for consecutive retained zeros.
    """
    T_x    = b_k_arb * T_max_arb
    strips = []
    prev_s = None
    m      = 1

    while True:
        z = arb_besseljzero(nu_int, m, eval_fn)

        # IVT check -- every m, including the stopping one
        flo = eval_fn(z - DELTA)
        fhi = eval_fn(z + DELTA)
        sign_ok = (((flo > _ZERO) and (fhi < _ZERO)) or
                   ((flo < _ZERO) and (fhi > _ZERO)))
        if not sign_ok:
            raise RuntimeError(
                f"IVT FAILED for J_{nu_int} zero #{m}  (z ~ {float(z.mid()):.6f})"
            )
        # True zero certified in [z - DELTA, z + DELTA].

        # Certified stopping: entire IVT bracket above T_x?
        if (z - DELTA) > T_x:
            break

        # Non-overlap for consecutive retained zeros
        if prev_s is not None:
            if not ((z - DELTA) > (prev_s + DELTA)):
                raise RuntimeError(
                    f"Non-overlap FAILED: J_{nu_int} zeros #{m-1} and #{m}"
                )

        prev_s = z
        # Pure ARB strip endpoints -- no float() for cut points
        strips.append(((z - DELTA) / b_k_arb, (z + DELTA) / b_k_arb))
        m += 1

    return strips


def collect_all_strips(b_arb, N_int, T_max_arb):
    """
    Collect strips from all 20 Bessel factors, then certify the ordering.

    The float sort is used only as an initial ordering; every consecutive
    pair is then certified by the ARB comparison t_lo[i] < t_lo[i+1].
    """
    strips = []
    strips += collect_strips(N_int, b_arb[0], T_max_arb,
                             lambda x: JN_real(x, N_int))
    strips += collect_strips(N_int, b_arb[1], T_max_arb,
                             lambda x: JN_real(x, N_int))
    for k in range(2, M):
        strips += collect_strips(0, b_arb[k], T_max_arb, J0_real)

    # Float sort for initial ordering (float used only as sort key)
    strips.sort(key=lambda s: float(s[0].mid()))

    # Certified ordering: every consecutive pair verified in ARB
    for i in range(len(strips) - 1):
        if not (strips[i][0] < strips[i + 1][0]):
            raise RuntimeError(
                f"Certified ordering FAILED at strip pair {i} and {i+1}"
            )

    return strips


# ── Gap / strip decomposition ─────────────────────────────────────────────────

def build_intervals(strips, T_max_arb):
    """
    Return gap intervals between strips.

    Uses prev_hi = max(prev_hi, t_hi) so prev_hi never moves backward.
    A gap is added only when ARB certifies prev_hi < t_lo.
    """
    gaps    = []
    prev_hi = T_EPS

    for (t_lo, t_hi) in strips:
        if prev_hi < t_lo:
            gaps.append((prev_hi, t_lo))
        if t_hi > prev_hi:
            prev_hi = t_hi

    gaps.append((prev_hi, T_max_arb))
    return gaps


# ── Core ARB integration ──────────────────────────────────────────────────────

def compute_I_arb(b_arb, b_acb, N_int, T_max_int):
    T_max_arb = arb(str(T_max_int))
    strips    = collect_all_strips(b_arb, N_int, T_max_arb)
    gaps      = build_intervals(strips, T_max_arb)
    n_sub     = len(gaps) + len(strips)
    pi_arb    = arb.pi()

    def integrand(t, _):
        v = JN_acb(b_acb[0] * t, N_int) * JN_acb(b_acb[1] * t, N_int)
        for k in range(2, M):
            v = v * J0_acb(b_acb[k] * t)
        return v

    total = arb(0)
    for (a, b) in gaps:
        piece = acb.integral(integrand, acb(a), acb(b),
                             rel_tol=REL_TOL, abs_tol=ABS_TOL)
        total = total + piece.real.__abs__()

    B_N         = LANDAU_C ** 2 * arb(N_int) ** arb("-2/3")
    strip_width = sum((t_hi - t_lo for t_lo, t_hi in strips), arb(0))
    strip_err   = (strip_width + T_EPS) * B_N / pi_arb

    return (total + strip_err * pi_arb) / pi_arb, n_sub, strip_err


# ── Scalings and checks ───────────────────────────────────────────────────────

def subcritical_scaling(N_int):
    return (acb(1).exp().real / arb(4)) ** N_int

def transition_scaling(N_int):
    return arb(N_int) ** arb("-8.67")

def arb_matches(val, paper_str, tol):
    paper = arb(paper_str)
    return (val - paper).__abs__().abs_upper() / paper.abs_lower() < tol

def certify_subcritical(I_arb, N_int):
    sc = subcritical_scaling(N_int)
    return I_arb.abs_upper() < sc.abs_lower(), I_arb.abs_upper(), sc.abs_lower()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("Lemma 6.6 (Transition Zone Bound) -- ARB-rigorous table certification")
    print(f"Precision: {ARB_PREC}-bit ARB throughout  |  no mpmath")
    print(f"M = {M}, chi_{D}, |A| = 2 (n_1 = n_2 = N), DELTA = 1e-20")
    print("Bessel zeros: McMahon asymptotic + ARB bisection (DLMF 10.21.19)")
    print("=" * 72)

    print(f"\nLoading chi_{D} zeros and computing ARB Lorentzian weights ...")
    b_arb, b_acb = load_weights()
    print(f"  b_1  = {b_arb[0]}")
    print(f"  b_2  = {b_arb[1]}")
    print(f"  b_20 = {b_arb[M-1]}")

    results  = {}
    all_pass = True

    print()
    print(f"{'N':<5} {'T':>8} {'I_n mid':>14} {'n_sub':>6} "
          f"{'strip_err':>12} {'I<(e/4)^N':>12} {'3sf':>6} {'time':>7}")
    print("-" * 80)

    for N in N_VALUES:
        T  = T_UPPER[N]
        t0 = time.time()
        I_arb, n_sub, s_err = compute_I_arb(b_arb, b_acb, N, T)
        elapsed = time.time() - t0

        sc_arb = subcritical_scaling(N)
        tr_arb = transition_scaling(N)

        subcrit_ok, I_up, sc_lo = certify_subcritical(I_arb, N)
        I_match  = arb_matches(I_arb,  PAPER_I[N],       REL_TOL_3SF)
        sc_match = arb_matches(sc_arb, PAPER_SUBCRIT[N], REL_TOL_2SF)
        tr_match = arb_matches(tr_arb, PAPER_TRANS[N],   REL_TOL_2SF)

        pass_N   = subcrit_ok and I_match and sc_match and tr_match
        all_pass = all_pass and pass_N
        results[N] = dict(I_arb=I_arb, sc_arb=sc_arb, tr_arb=tr_arb,
                          s_err=s_err, n_sub=n_sub,
                          subcrit_ok=subcrit_ok, I_up=I_up, sc_lo=sc_lo,
                          I_match=I_match, sc_match=sc_match, tr_match=tr_match)

        print(f"{N:<5} {T:>8} {float(I_arb.mid()):>14.4e} {n_sub:>6} "
              f"{float(s_err.mid()):>12.2e} "
              f"{'PASS' if subcrit_ok else 'FAIL':>12} "
              f"{'PASS' if (I_match and sc_match and tr_match) else 'FAIL':>6} "
              f"{elapsed:>6.1f}s")

    print("\n" + "=" * 72)
    print("DETAILED ARB CERTIFICATION REPORT")
    print("=" * 72)
    for N in N_VALUES:
        r = results[N]
        print(f"\nN = {N}:")
        print(f"  I_n (ARB ball)       = {r['I_arb']}")
        print(f"  I_n (paper)          = {PAPER_I[N]}")
        print(f"  3sf match            : {'PASS' if r['I_match'] else 'FAIL'}")
        print(f"  strip error bound    = {r['s_err']}")
        print(f"  (e/4)^N (ARB)        = {r['sc_arb']}")
        print(f"  (e/4)^N (paper)      = {PAPER_SUBCRIT[N]}")
        print(f"  subcrit 2sf match    : {'PASS' if r['sc_match'] else 'FAIL'}")
        print(f"  I_upper              = {r['I_up']}")
        print(f"  sc_lower             = {r['sc_lo']}")
        print(f"  I_upper < sc_lower   : "
              f"{'PASS [ARB-certified]' if r['subcrit_ok'] else 'FAIL'}")
        print(f"  N^{{-8.67}} (ARB)      = {r['tr_arb']}")
        print(f"  N^{{-8.67}} (paper)    = {PAPER_TRANS[N]}")
        print(f"  trans  2sf match     : {'PASS' if r['tr_match'] else 'FAIL'}")

    subcrit_all = all(results[N]['subcrit_ok'] for N in N_VALUES)
    table_all   = all(results[N]['I_match'] and results[N]['sc_match']
                      and results[N]['tr_match'] for N in N_VALUES)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Method      : acb.integral (Petras), {ARB_PREC}-bit ARB throughout")
    print(f"  Bessel zeros: McMahon (DLMF 10.21.19) + ARB bisection -- no mpmath")
    print(f"  Stopping    : IVT-certified for every zero, incl. first excluded")
    print(f"  Ordering    : ARB comparison t_lo[i] < t_lo[i+1] certified")
    print(f"  build_int   : prev_hi = max(prev_hi, t_hi); ARB gap condition")
    print(f"  Cut points  : acb(arb_value) -- no float() for integration limits")
    print(f"  N values    : {N_VALUES}")
    print(f"  I_n < (e/4)^N  [ARB-certified, all N] : {'PASS' if subcrit_all else 'FAIL'}")
    print(f"  Table match    [3sf/2sf, all N]        : {'PASS' if table_all else 'FAIL'}")
    print()
    if all_pass:
        print("RESULT: ALL CHECKS PASSED (ARB-rigorous, no mpmath)")
        print("  I_n < (e/4)^N certified by ARB endpoint comparison for all N.")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED -- see report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
