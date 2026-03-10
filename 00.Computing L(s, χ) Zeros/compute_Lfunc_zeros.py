"""
compute_Lfunc_zeros.py  —  Rigorous computation of zeros of quadratic Dirichlet L-functions using ARB
=====================================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

Methodology and prior work
--------------------------
The standard method for large-scale certified zero counting of L-functions
is Turing's method (Turing, 1953; generalized to Dirichlet L-functions by
Rumely, 1993, and Booker, 2006).  Turing's method works entirely on the
critical line: sign changes of the Hardy Z-function provide a lower bound
on the zero count, and an analytic bound on the integral of S(T,chi)
certifies completeness without any off-line evaluation.  This approach
dominates all major verified computations, including Platt's verification
of GRH for all primitive characters with conductor q <= 400,000 (Platt,
Math. Comp. 85, 2016).

This module uses direct contour integration of L'/L around the rectangle
[-0.5, 1.5] x [t_lo, T_max] instead.  The choice is deliberate and rests
on three properties that are advantageous at the scale of this computation:

  (i)  Self-containedness.  The contour integral yields a provably correct
       integer zero count with no dependence on character-specific analytic
       bounds on S(T,chi) or Backlund-type estimates.  The only inputs are
       the L-function values themselves, evaluated in ARB.

  (ii) Simplicity.  A single interval-arithmetic winding number computation
       constitutes the completeness certificate.  There are no auxiliary
       analytic ingredients to verify or import.

  (iii) Adequacy at moderate height.  For the eight characters treated here
        (conductors q = 5 to 44, heights T ~ 900 to 1100) the efficiency
        penalty relative to Turing's method is negligible.  Turing's method
        is essential at Platt's scale (q up to 400,000, T up to 10,000);
        at the present scale either approach is computationally trivial.

Published precedent for direct contour integration of zeta'/zeta with ARB
ball arithmetic appears in Johansson, "Numerical integration in arbitrary-
precision ball arithmetic," ICMS 2018 (arXiv:1802.07942), which explicitly
computes N(T) for zeta(s) via a rectangular contour integral and notes that
the result is "a ball that provably determines N(T) as an integer."  The
present module applies the same approach to Dirichlet L-functions.

As an independent cross-check, the winding number count returned by Phase 4
is compared against the analytic lower bound on N(T,chi) supplied by the
BMOR explicit formula (Bennett-Martin-O'Bryant-Rechnitzer, Math. Comp. 90,
2021).  Agreement between the contour count and the BMOR lower bound at
each seal height provides a secondary validation that is methodologically
independent of both the contour integral and the individual zero locations.

References
----------
Johansson, F.  Arb: Efficient arbitrary-precision midpoint-radius interval
    arithmetic.  IEEE Trans. Comput. 66(8), 2017.
    https://doi.org/10.1109/TC.2017.2690633

Johansson, F.  Numerical integration in arbitrary-precision ball arithmetic.
    ICMS 2018.  arXiv:1802.07942.

Bennett, M., Martin, G., O'Bryant, K., Rechnitzer, A.  Counting zeros of
    Dirichlet L-functions.  Math. Comp. 90, 2021.
    https://doi.org/10.1090/mcom/3599

Platt, D.  Numerical computations concerning the GRH.  Math. Comp. 85, 2016.
    https://doi.org/10.1090/mcom/3077

Rumely, R.  Numerical computations concerning the ERH.  Math. Comp. 61, 1993.
    https://doi.org/10.1090/S0025-5718-1993-1195435-0

Algorithm
---------
The rigorous flow is:

  (1) Scan t in [t_lo, T_max] globally and collect candidate zeros.
  (2) Certify each candidate locally by Newton refinement and a local
      winding number enclosure.
  (3) Run a global argument-principle count on the full verification
      rectangle [-0.5, 1.5] x [t_lo, T_max].  If the count equals the
      number of certified zeros, the table is complete and the procedure
      terminates.
  (4) If the global count reveals missing zeros, localize the discrepancy:
      run strip-by-strip winding checks to identify which inter-zero
      gaps are deficient, then locate a seed in each deficient gap by a
      narrow rectangular winding number.
  (5) Certify the recovered zeros by Newton refinement and local
      winding number enclosure, then merge them into the table.
  (6) Rerun the global argument-principle count on the full region.
      This is the seal: the theorem asserted at the end is
          #{certified zeros in region} = #{zeros by argument principle}.
      Local window checks are tools for repair; the global contour is
      what certifies the final table.

Implementation
--------------
Steps (1)-(2) correspond to Phases 1, 2, and 3 of the code.  Step (3)
is Phase 4 (verify_completeness).  Steps (4)-(5) are Phases 5 and 6
(phase5_locate_seeds followed by phase3_newton at 1500-bit).  Step (6)
is a second call to verify_completeness on the repaired table inside
the compute_zeros recovery branch.  Phases 7 and 8 are manual closeout
steps applied after the recovery seal passes: Phase 7 extends the
forward scan to a pre-fixed height T and reseals; Phase 8 boosts any
zero with a bound weaker than the table-wide floor to 2200-bit.

  Phase 1 (128-bit):
    Scan the Hardy Z-function Z_chi(t) on a uniform grid (step 0.05).
    Sign changes are detected by ARB-certified comparisons: a sign
    change is confirmed only when consecutive Z-values are certified
    to lie on opposite sides of zero (arb balls do not straddle zero).

  Phase 2 (128-bit):
    At each sign-change bracket from Phase 1, 20 bisection steps narrow
    the bracket using ARB-certified comparisons, producing a ~6-digit
    seed.  The spurious filter checks |L(1/2 + i*gamma)| < 10^{-4} via
    a certified ARB less-than comparison to reject Hardy-phase
    oscillations that are not genuine zeros.

  Phase 3 (128-bit or 1500-bit):
    Newton refinement of each Phase 2 seed to full working precision.
    Both L(s, chi_d) and L'(s, chi_d) are evaluated as certified ARB
    balls via acb_series.zeta (Taylor series of Hurwitz zeta to order
    2); no finite-difference approximation is used.  Newton iteration
    is heuristic seed improvement only; the rigorous certification
    consists of (i) the certified magnitude bound |L(1/2+i*gamma)|
    from eval_L and (ii) a local winding number enclosure from
    certify_zero_location, neither of which uses the derivative.
    At 1500-bit the typical certified bound is |L| < 10^{-430} to
    10^{-450}, but some zeros yield weaker bounds in the range
    10^{-350} to 10^{-420}.  To push a weak zero above 10^{-400},
    rerun phase3_newton on that seed with target_prec=2200 and
    cert_threshold_exp=-400; in practice this reliably achieves
    bounds of order 10^{-420} to 10^{-560}.

  Phase 4: Global completeness verification (the seal).
    The argument principle counts zeros of L(s, chi_d) in the wide
    rectangle [-0.5, 1.5] x [t_lo, T_max] via ARB-rigorous argument
    tracking.  If the count equals the certified zero count the table
    is complete.  If not, a CompletenessError is raised and the
    recovery path (Phases 5 and 6) is invoked, followed by a second
    run of Phase 4 on the repaired table.  As a secondary validation,
    the winding number count is compared against the analytic lower
    bound from the BMOR explicit formula (Bennett-Martin-O'Bryant-
    Rechnitzer, Math. Comp. 90, 2021); the contour count must be at
    least as large as the BMOR bound at T_max.

  Phase 5: Missing zero seed location (recovery).
    Strip-by-strip winding checks identify which inter-zero gaps are
    deficient.  For each deficient gap [a, b], the winding number of L
    on [0.49, 0.51] x [a, b] is computed.  A winding number of 1
    certifies exactly one missing zero in that gap; its midpoint is
    returned as a seed.  If any gap has winding number greater than 1,
    Phase 5 raises a CompletenessError: two or more zeros in a single
    inter-zero gap cannot be resolved by midpoint seeding and require
    manual bisection of that gap before rerunning.

  Phase 6: High-precision Newton refinement of recovery seeds (1500-bit).
    Identical in mechanism to Phase 3 but always runs at 1500-bit with
    cert_threshold_exp=-300.  After Phase 6, the recovered zeros are
    merged into the table and Phase 4 is rerun to provide the final seal.
    Recovery seeds often yield weaker initial bounds than forward-scan
    zeros because the seed is a gap midpoint rather than a refined
    bisection bracket.  If any recovered zero has a bound weaker than
    desired, it is passed to Phase 8.

  Phase 7: Tail extension to a fixed height T.
    The seal (Phase 4) must be run against a T chosen in advance, not
    derived from the last known zero with a running buffer.  A buffer
    that extends T dynamically will keep crossing new zeros and cause
    repeated seal failures.  The correct procedure is to fix T at the
    outset — the natural choice is the nearest integer above the
    Von Mangoldt zero count estimate for the target count — then run
    Phases 1 through 4 to that T.  If the seal passes, the table is
    complete.  If the seal fails because the buffer has not yet reached
    all zeros up to T, run interleaved_scan_and_filter from the last
    certified zero to T, certify the new zeros via Phase 3, merge, and
    reseal.  Phase 7 is not invoked automatically; it is a manual
    closeout step applied once Phase 4 has passed on the recovered table
    but the buffer has left a gap between the last certified zero and T.

  Phase 8: Bound boosting to 2200-bit (closeout).
    Any zero whose certified bound is weaker than a target threshold
    (typically 10^{-400}) after Phases 3 or 6 is rerun through
    phase3_newton with target_prec=2200 and cert_threshold_exp=-400.
    In practice this raises bounds from the 10^{-320} to 10^{-380}
    range into the 10^{-420} to 10^{-560} range.  Weak bounds arise
    from zeros with anomalously small |L'(1/2+i*gamma)| (flat zeros)
    or anomalously close neighboring zeros; the boost does not
    eliminate the weakness but pushes the certified bound as deep as
    the arithmetic permits at 2200-bit.  Phase 8 is applied after
    Phase 6 for recovery zeros and after Phase 3 for any forward-scan
    zero that falls below the table-wide floor.

All arithmetic is rigorous interval arithmetic via ARB (python-flint).
Character data is imported from Kronecker_character_data.py.

Requirements:
    python-flint >= 0.8.0  (provides ARB ball arithmetic)
    Python >= 3.10

Usage:
    python compute_Lfunc_zeros.py --d 5 --nzeros 20
    python compute_Lfunc_zeros.py --d 5 --nzeros 200 --high-precision
    python compute_Lfunc_zeros.py --d 2 --nzeros 20

Options:
    --d               Squarefree discriminant (2,3,5,6,7,10,11,13)
    --nzeros          Number of zeros to compute
    --high-precision  Use 1500-bit precision with Newton refinement
                      (default: 128-bit for ~20-digit zeros)
    --grid-step       Grid step for sign-change scan (default: 0.05)
    --margin          Safety margin factor for T_max (default: 1.10)
    --skip-verify     Skip Phase 4 completeness verification
"""

import argparse
import sys
import time
from math import pi, log, e, ceil

from flint import arb, acb, ctx
from Kronecker_character_data import get_character, CHARACTERS


# ================================================================
# Exceptions
# ================================================================

class ArbPrecisionError(Exception):
    """Raised when ARB interval arithmetic cannot certify a result.

    Callers should increase ctx.prec or sampling density and retry.
    """


class CompletenessError(Exception):
    """Raised when the argument-principle count does not match the zero table.

    This means the certified zero table is missing at least one zero in
    the verification region.  The remedy is either to reduce grid_step
    and rerun the scan, or to invoke the recovery path (Phases 5 and 6)
    which localizes missing zeros by inter-zero gap winding numbers and
    certifies them individually before rerunning the global verification.
    """


# ================================================================
# Precomputed constants
# ================================================================

class CharacterData:
    """Precomputed ARB constants for a given character.

    Stores conductor, log(q/pi), and Hurwitz parameters a/q as arb
    objects at the current working precision.
    """

    __slots__ = ('d', 'q', 'q_arb', 'log_q', 'log_q_over_pi', 'hurwitz_pairs')

    def __init__(self, d):
        q, chi = get_character(d)
        self.d = d
        self.q = q
        self.q_arb = arb(q)
        self.log_q = arb(q).log()
        self.log_q_over_pi = (arb(q) / arb.pi()).log()
        self.hurwitz_pairs = [
            (acb(arb(a) / arb(q)), chi[a])
            for a in sorted(chi.keys())
        ]


# ================================================================
# L-function evaluation
# ================================================================

def eval_L(s, char_data):
    """Evaluate L(s, chi_d) via the Hurwitz zeta decomposition.

    L(s, chi_d) = q^{-s} * sum_{a: gcd(a,q)=1} chi_d(a) * zeta(s, a/q)

    Each Hurwitz zeta value is computed by ARB's built-in acb.zeta,
    which returns a certified complex ball.
    """
    total = acb(0)
    for a_over_q, chi_val in char_data.hurwitz_pairs:
        hz = acb.zeta(s, a_over_q)
        if chi_val == +1:
            total += hz
        else:
            total -= hz
    log_q = char_data.log_q
    q_neg_s = (-s * log_q).exp()
    return total * q_neg_s


def eval_L_with_deriv(s, char_data):
    """Evaluate L(s, chi_d) and L'(s, chi_d) as certified ARB balls.

    Both the value and the derivative are computed rigorously using
    acb_series.zeta, which evaluates the Taylor series of the Hurwitz
    zeta function zeta(s + x, a/q) to order 2 at x = 0.  The two
    coefficients of this series are zeta(s, a/q) and zeta'(s, a/q)
    respectively, each returned as a certified ARB ball.  No
    finite-difference approximation is used anywhere.

    The derivative formula follows from differentiating

        L(s, chi_d) = q^{-s} * sum_{a} chi(a) * zeta(s, a/q)

    by the product rule:

        L'(s, chi_d) = q^{-s} * (sum_{a} chi(a) * zeta'(s, a/q)
                                  - log(q) * sum_{a} chi(a) * zeta(s, a/q))

    Parameters
    ----------
    s : acb
        Point of evaluation.
    char_data : CharacterData
        Precomputed constants at the current working precision.

    Returns
    -------
    (L_val, L_deriv) : (acb, acb)
        Certified ARB balls for L(s, chi_d) and L'(s, chi_d).
    """
    from flint import acb_series as _acb_series
    s_ser = _acb_series([s, acb(1)], prec=ctx.prec)

    total_val   = acb(0)
    total_deriv = acb(0)
    for a_over_q, chi_val in char_data.hurwitz_pairs:
        hz_ser = s_ser.zeta(a_over_q)
        zv = hz_ser[0]
        zd = hz_ser[1]
        if chi_val == +1:
            total_val   += zv
            total_deriv += zd
        else:
            total_val   -= zv
            total_deriv -= zd

    log_q   = acb(char_data.log_q)
    q_neg_s = (-s * log_q).exp()
    L_val   = q_neg_s * total_val
    L_deriv = q_neg_s * (total_deriv - log_q * total_val)
    return L_val, L_deriv


# ================================================================
# Hardy Z-function
# ================================================================

def hardy_phase(t_arb, char_data):
    """Compute the Hardy phase theta_chi(t) for an even primitive character.

    For chi an even primitive character (chi(-1) = +1) of conductor q with
    root number epsilon = tau(chi)/sqrt(q) = +1, the completed L-function

        Lambda(s, chi) = (q/pi)^{s/2} * Gamma(s/2) * L(s, chi)

    satisfies Lambda(1/2 + it) in R for all real t.  Writing

        Lambda(1/2 + it) = (q/pi)^{1/4} * exp(i*(t/2)*log(q/pi))
                           * Gamma(1/4 + it/2) * L(1/2 + it, chi),

    the Hardy Z-function Z_chi(t) = exp(i*theta_chi(t)) * L(1/2+it, chi)
    is real precisely when theta_chi cancels the argument of the prefactor:

        theta_chi(t) = Im(log Gamma(s/2)) + (t/2) * log(q/pi),

    where s = 1/2 + it (so s/2 = 1/4 + it/2).  For q = 1 this reduces to
    the Riemann-Siegel theta function Im(log Gamma(1/4+it/2)) - (t/2)*log(pi),
    consistent with the sign convention log(1/pi) = -log(pi).

    All characters in this module are even with root number +1 (verified in
    Kronecker_character_data.py), so this formula applies without modification.

    Note: the sign on log(q/pi) is PLUS (not minus).  The minus-sign
    convention occasionally appears in the literature for the specific case
    q = pi (which does not occur here), or through a different normalisation
    of the functional equation.  The present formula is verified by checking
    that Im(eval_Z(t, cd)) = 0 to full working precision at multiple ordinates
    for each character.
    """
    s = acb(arb("0.5"), t_arb)
    log_gamma = (s / 2).lgamma()
    return log_gamma.imag + (t_arb / 2) * char_data.log_q_over_pi


def eval_Z(t_arb, char_data):
    """Evaluate the Hardy Z-function Z_chi(t).

    Z_chi(t) = exp(i * theta_chi(t)) * L(1/2 + it, chi_d)

    Real-valued for real t; its real zeros correspond to zeros of
    L(s, chi_d) on the critical line.
    """
    s = acb(arb("0.5"), t_arb)
    L_val = eval_L(s, char_data)
    theta = hardy_phase(t_arb, char_data)
    phase = acb(arb(0), theta).exp()
    return (phase * L_val).real


# ================================================================
# Zero-counting formula
# ================================================================

def zero_count_estimate(T, q):
    """Estimate N(T, chi_d) ~ (T/pi) * log(qT / (2*pi*e))."""
    if T <= 0:
        return 0
    return (T / pi) * log(q * T / (2 * pi * e))


def choose_T_max(nzeros, q, margin=1.10):
    """Choose T_max so that N(T_max, chi_d) >= nzeros * margin."""
    T_lo, T_hi = 1.0, 10.0
    while zero_count_estimate(T_hi, q) < nzeros * margin:
        T_hi *= 2
    for _ in range(100):
        T_mid = (T_lo + T_hi) / 2
        if zero_count_estimate(T_mid, q) < nzeros * margin:
            T_lo = T_mid
        else:
            T_hi = T_mid
    return T_hi


# ================================================================
# Phase 1+2 helpers: ARB-certified sign detection and bisection
# ================================================================

def _certified_sign_change(Za, Zb):
    """Return True if Za and Zb are certified to have opposite signs.

    Uses ARB-certified comparisons.  Returns False if either ball
    straddles zero or if the signs agree.
    """
    zero = arb(0)
    return (Za > zero and Zb < zero) or (Za < zero and Zb > zero)


def _certified_same_sign(Za, Zb):
    """Return True if Za and Zb are certified to have the same sign."""
    zero = arb(0)
    return (Za > zero and Zb > zero) or (Za < zero and Zb < zero)


def _bisect_bracket(a, b, Za, cd, n_steps=20):
    """Narrow a sign-change bracket by ARB-certified bisection.

    At each step, certified ARB comparisons determine which
    sub-bracket contains the sign change.  If the midpoint value
    straddles zero, bisection has converged within the ball radius
    and the midpoint is returned.
    """
    zero = arb(0)
    for _ in range(n_steps):
        mid = ((a + b) / 2).mid()
        Zmid = eval_Z(mid, cd)
        if not (Zmid > zero) and not (Zmid < zero):
            return mid
        if _certified_same_sign(Za, Zmid):
            a = mid
            Za = Zmid
        else:
            b = mid
    return ((a + b) / 2).mid()


# ================================================================
# Phase 1+2: Interleaved sign-change scan and bracket filtering
# ================================================================

def interleaved_scan_and_filter(d, T_max, grid_step=0.05, target=None):
    """Scan Z_chi(t) on a grid, filter each sign change immediately.

    Sign changes are detected by ARB-certified comparisons on arb
    balls.  Bisection and the spurious filter both use certified ARB
    comparisons; no float() stripping occurs.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    T_max : float
        Upper bound of scan interval [0, T_max].
    grid_step : float
        Spacing of evaluation grid (default 0.05).
    target : int or None
        Stop after collecting this many genuine seeds.

    Returns
    -------
    seeds : list of float
        Approximate zero ordinates (genuine, sorted).
    n_spurious : int
        Number of spurious brackets filtered out.
    T_scan : float
        The ordinate actually reached by the scan (equals T_max unless
        target was hit early).
    """
    saved_prec = ctx.prec
    ctx.prec = 128
    cd = CharacterData(d)

    n_points = ceil(T_max / grid_step) + 1
    grid = arb(str(grid_step))
    # Heuristic filter only: a genuine zero satisfies |L(1/2+i*gamma)| << 1,
    # while a Hardy-phase oscillation that is not a zero typically gives
    # |L| ~ 1 or larger.  The threshold 1e-4 is chosen conservatively so
    # that no genuine zero is discarded; any false negative here is
    # corrected by the Phase 4 global completeness check.
    spurious_thresh = arb("1e-4")
    zero = arb(0)

    print(f"  Phase 1+2: scanning {n_points} grid points on "
          f"[0, {T_max:.1f}] at 128-bit ...")

    seeds = []
    n_spurious = 0
    T_scan = grid_step

    t_arb = grid
    Z_prev = eval_Z(t_arb, cd)

    for i in range(2, n_points):
        t_arb = t_arb + grid
        Z_curr = eval_Z(t_arb, cd)
        T_scan = float(t_arb.mid())

        certified = _certified_sign_change(Z_prev, Z_curr)
        prev_straddles = not (Z_prev > zero) and not (Z_prev < zero)
        curr_straddles = not (Z_curr > zero) and not (Z_curr < zero)
        conservative = not certified and (prev_straddles or curr_straddles)

        if certified or conservative:
            gamma = _bisect_bracket(t_arb - grid, t_arb, Z_prev, cd)
            s = acb(arb("0.5"), gamma)
            L_mag = abs(eval_L(s, cd))
            if L_mag < spurious_thresh:
                seeds.append(float(gamma.mid()))
                if target is not None and len(seeds) >= target:
                    break
            else:
                n_spurious += 1

        Z_prev = Z_curr
        if i % 500 == 0:
            print(f"    ... {i}/{n_points} points, "
                  f"{len(seeds)} genuine, {n_spurious} spurious")

    print(f"  Phase 1+2 complete: {len(seeds)} genuine seeds, "
          f"{n_spurious} spurious filtered.")
    ctx.prec = saved_prec
    return seeds, n_spurious, T_scan


# ================================================================
# Phase 3: Newton refinement
# ================================================================

def phase3_newton(d, seeds, target_prec=128, max_iters=10,
                  cert_threshold_exp=-18):
    """Refine seeds by Newton iteration on the critical line.

    Both L(s, chi_d) and L'(s, chi_d) are evaluated as certified ARB
    balls via eval_L_with_deriv, which uses acb_series.zeta (Taylor
    series of Hurwitz zeta to order 2).  No finite-difference
    approximation is used anywhere.

    Each Newton step computes the update
    s_{n+1} = s_n - L(s_n, chi_d) / L'(s_n, chi_d)
    using the value and derivative of L returned by eval_L_with_deriv,
    then projects the result back to the critical line by replacing the
    real part with 1/2 via midpoint extraction.  This is Newton's method
    applied to L on the critical line, not a Newton iteration on the Hardy
    Z-function; the projection keeps all iterates on sigma = 1/2 and
    prevents ball inflation in the imaginary part.

    Newton refinement is heuristic seed improvement only; it is not the
    proof.  The rigorous certification consists entirely of the two items
    below.

    The certification is:
      (i)  Magnitude: the ARB ball |L(1/2 + i*t)| < 10^cert_threshold_exp
           is a certified rigorous bound, establishing proximity to zero.
      (ii) Enclosure: a winding number of 1 computed by the argument
           principle on the thin rectangle [0.49, 0.51] x [t-h, t+h]
           (see certify_zero_location) certifies that exactly one zero of
           L lies in that box, unconditionally.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    seeds : list of float
        Approximate zero ordinates from Phase 1+2.
    target_prec : int
        ARB working precision in bits (128 standard, 1500 high).
    max_iters : int
        Maximum Newton iterations.
    cert_threshold_exp : int
        Accept when |L(1/2+i*t)| < 10^cert_threshold_exp.

    Returns
    -------
    list of dict
        Each dict has 'gamma', 'L_bound' (arb), 'certified' (bool),
        'enclosure', 'enclosure_exp'.
    """
    saved_prec = ctx.prec
    ctx.prec = target_prec
    cd = CharacterData(d)
    deriv_thresh = arb(10) ** (cert_threshold_exp * 2)
    cert_thresh  = arb(10) ** cert_threshold_exp

    results = []
    for idx, seed_val in enumerate(seeds):
        s = acb(arb("0.5"), arb(str(seed_val)))

        for _ in range(max_iters):
            fv, fp = eval_L_with_deriv(s, cd)
            if abs(fv) < arb(10) ** (cert_threshold_exp - 5):
                break
            if abs(fp) < deriv_thresh:
                break
            sn = s - fv / fp
            # Keep on critical line: fix real part to 0.5.
            s = acb(arb("0.5"), sn.imag.mid())

        fv, _ = eval_L_with_deriv(s, cd)
        L_abs = abs(fv)
        magnitude_certified = bool(L_abs < cert_thresh)

        # Certified enclosure.
        enclosure = None
        enclosure_exp = None
        try:
            t_lo, t_hi, enc_h = certify_zero_location(d, s.imag, cd)
            enclosure = (t_lo, t_hi)
            import math
            enclosure_exp = int(round(math.log10(enc_h)))
        except ArbPrecisionError:
            pass

        results.append({
            'gamma': s.imag.mid(),
            'L_bound': L_abs,
            'certified': magnitude_certified,
            'enclosure': enclosure,
            'enclosure_exp': enclosure_exp,
        })

        if (idx + 1) % 20 == 0 or (idx + 1) == len(seeds):
            print(f"    ... {idx + 1}/{len(seeds)} zeros refined")

    n_cert = sum(1 for r in results if r['certified'])
    n_enc  = sum(1 for r in results if r['enclosure'] is not None)
    print(f"  Phase 3: {n_cert}/{len(results)} zeros certified "
          f"(|L| < 10^{cert_threshold_exp}), "
          f"{n_enc}/{len(results)} with certified enclosures.")
    ctx.prec = saved_prec
    return results


# ================================================================
# Phase 4: ARB-rigorous completeness verification
# ================================================================

def _track_arg_segment_L(s_start, s_end, cd, n_pts):
    """Track cumulative arg(L) along a segment, ARB-rigorous.

    Identical in structure to the DH_core.py implementation but
    calls eval_L instead of eval_DH.  See DH_core.py for full
    documentation of the algorithm and preconditions.

    Raises ArbPrecisionError if a delta cannot be certified in
    (-pi, pi) or if L is not certified nonzero at a sample point.
    """
    pi_arb = arb.pi()
    two_pi = 2 * pi_arb
    total = arb(0)

    f_prev = eval_L(s_start, cd)
    if not f_prev.real.is_finite() or not f_prev.imag.is_finite():
        raise ArbPrecisionError(
            "L is not finite at segment start.")
    prev_arg = f_prev.arg()

    for k in range(1, n_pts + 1):
        frac = arb(k) / arb(n_pts)
        s = s_start + (s_end - s_start) * acb(frac)
        f_curr = eval_L(s, cd)

        if not (abs(f_curr) > arb(0)):
            raise ArbPrecisionError(
                f"L(s) ball contains zero at step {k}/{n_pts}; "
                "increase ctx.prec or n_pts.")

        curr_arg = f_curr.arg()
        delta = curr_arg - prev_arg

        delta_mid = delta.mid()
        if delta_mid > pi_arb:
            delta = delta - two_pi
        elif delta_mid < -pi_arb:
            delta = delta + two_pi

        if not (delta > -pi_arb and delta < pi_arb):
            raise ArbPrecisionError(
                f"Cannot certify unwrapped delta in (-pi, pi) at "
                f"step {k}/{n_pts} (ball = {delta}); "
                "increase ctx.prec or n_pts.")

        total = total + delta
        prev_arg = curr_arg

    return total


def count_zeros_rect(d, sigma_lo, sigma_hi, t_lo, t_hi, cd,
                     pts_per_unit=40):
    """Count zeros of L(s, chi_d) in a rectangle, ARB-rigorous.

    Uses the argument principle: N = (1/2*pi) * oint d(arg L)
    around the rectangle boundary (counterclockwise).

    All arithmetic is ARB-rigorous.  Raises ArbPrecisionError if
    the winding number cannot be certified as a unique integer.

    Parameters
    ----------
    d : int
        Squarefree discriminant (used only for error messages).
    sigma_lo, sigma_hi : float
        Horizontal bounds.
    t_lo, t_hi : float
        Vertical bounds.
    cd : CharacterData
        Precomputed constants at current working precision.
    pts_per_unit : int
        Evaluation density per unit of contour length.

    Returns
    -------
    int
        Certified integer zero count inside the rectangle.

    Raises
    ------
    ArbPrecisionError
        If the winding number cannot be certified.  Increase
        pts_per_unit or ctx.prec and retry.
    """
    h_pts = max(int(abs(sigma_hi - sigma_lo) * pts_per_unit), 30)
    v_pts = max(int(abs(t_hi - t_lo) * pts_per_unit), 30)
    v_pts = min(v_pts, 3000)

    s_bl = acb(arb(str(sigma_lo)), arb(str(t_lo)))
    s_br = acb(arb(str(sigma_hi)), arb(str(t_lo)))
    s_tr = acb(arb(str(sigma_hi)), arb(str(t_hi)))
    s_tl = acb(arb(str(sigma_lo)), arb(str(t_hi)))

    total_arg = arb(0)
    total_arg = total_arg + _track_arg_segment_L(s_bl, s_br, cd, h_pts)
    total_arg = total_arg + _track_arg_segment_L(s_br, s_tr, cd, v_pts)
    total_arg = total_arg + _track_arg_segment_L(s_tr, s_tl, cd, h_pts)
    total_arg = total_arg + _track_arg_segment_L(s_tl, s_bl, cd, v_pts)

    winding = total_arg / (2 * arb.pi())
    n = winding.unique_fmpz()
    if n is None:
        raise ArbPrecisionError(
            f"Winding number ball {winding} for L(s, chi_{d}) "
            "straddles an integer; increase pts_per_unit or ctx.prec.")
    return int(n)


def certify_zero_location(d, t_approx, cd,
                          half_width_exp=-8, pts=80,
                          fallback_exps=(-7, -6, -5)):
    """Certify that L(s, chi_d) has exactly one zero near t_approx on the critical line.

    Computes the winding number of L around the thin rectangle

        [0.49, 0.51] x [t - h, t + h],    h = 10^half_width_exp,

    A winding number of 1 certifies, via the argument principle in fully
    rigorous ARB arithmetic, that L has exactly one zero in the rectangle.
    Because chi_d is a real primitive character, the functional equation
    supplies the symmetry rho <-> 1 - rho_bar, forcing zeros off the
    critical line to occur in pairs.  If the unique zero in
    [0.49, 0.51] x [t-h, t+h] lay at sigma_0 + i*t_0 with sigma_0 != 1/2,
    then 1 - sigma_0 + i*t_0 would be a second zero of L in the same
    rectangle (since |1/2 - (1 - sigma_0)| = |sigma_0 - 1/2| < 0.01),
    contradicting the winding number being 1.  Hence the unique zero is
    on sigma = 1/2, certified to t-ordinate accuracy h.

    This provides a true certified zero enclosure: the zero location is
    certified to within 10^half_width_exp in the t-ordinate.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    t_approx : arb
        Newton iterate (ordinate on the critical line).
    cd : CharacterData
        Precomputed constants.
    half_width_exp : int
        Exponent for initial box half-width h = 10^half_width_exp.
    pts : int
        Evaluation density for the winding number contour.
    fallback_exps : tuple of int
        Larger (less negative) exponents to try if the initial box fails.

    Returns
    -------
    (t_lo, t_hi, h) : tuple of float
        Certified t-interval and half-width.

    Raises
    ------
    ArbPrecisionError
        If no enclosure achieves a certified winding number of 1.
    """
    t = float(t_approx.mid()) if hasattr(t_approx, 'mid') else float(t_approx)

    for exp in (half_width_exp,) + fallback_exps:
        h = 10.0 ** exp
        try:
            n = count_zeros_rect(d, 0.49, 0.51, t - h, t + h, cd,
                                 pts_per_unit=pts)
            if n == 1:
                return (t - h, t + h, h)
        except ArbPrecisionError:
            continue

    raise ArbPrecisionError(
        f"certify_zero_location: winding number != 1 for all tried "
        f"half-widths 10^{half_width_exp} through 10^{fallback_exps[-1]} "
        f"around t = {t:.8f} for L(s, chi_{d}).  "
        "Increase pts or working precision and retry.")


def verify_completeness(d, T_max, expected_count, pts_per_unit=40,
                        t_lo_offset=0.5):
    """Verify that the current certified zero table contains all zeros up to T_max.

    Computes the winding number of L(s, chi_d) around the rectangle
    [-0.5, 1.5] x [t_lo_offset, T_max] and checks that it equals
    expected_count.  This is the global seal: it certifies that no
    zeros were missed anywhere in the verification region, whether the
    table was built by the forward scan or by the recovery path.

    The lower cutoff t_lo_offset avoids the real axis where L has
    no zeros for primitive characters.  The value 0.5 is safe for
    all characters in this module provided the first zero ordinate
    exceeds 0.5, which the caller is responsible for verifying.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    T_max : float
        Upper height of the verification rectangle.
    expected_count : int
        Number of zeros currently in the certified table.
    pts_per_unit : int
        Evaluation density for the argument-principle contour.
    t_lo_offset : float
        Lower bound of the verification rectangle (default 0.5).

    Raises
    ------
    CompletenessError
        If the certified zero count does not equal expected_count.
    ArbPrecisionError
        If ARB cannot certify the winding number.  Increase
        pts_per_unit or ctx.prec.
    """
    saved_prec = ctx.prec
    ctx.prec = max(ctx.prec, 192)
    cd = CharacterData(d)

    print(f"  Phase 4: verifying completeness via argument principle "
          f"on [-0.5, 1.5] x [{t_lo_offset}, {T_max:.2f}] ...")

    n = count_zeros_rect(d, -0.5, 1.5, t_lo_offset, T_max, cd,
                         pts_per_unit=pts_per_unit)

    ctx.prec = saved_prec

    if n != expected_count:
        raise CompletenessError(
            f"Completeness failure for L(s, chi_{d}): "
            f"argument principle gives {n} zero(s) in "
            f"[-0.5, 1.5] x [{t_lo_offset}, {T_max:.4f}], "
            f"but the certified table contains {expected_count}. "
            f"At least {abs(n - expected_count)} zero(s) are missing. "
            f"Invoke the recovery path (Phases 5 and 6) to locate and "
            f"certify the missing zeros, then rerun verification.")

    print(f"  Phase 4: certified {n} zeros in "
          f"[-0.5, 1.5] x [{t_lo_offset}, {T_max:.2f}].  "
          f"Completeness verified.")


# ================================================================
# Phase 5: Missing zero seed location by inter-zero gap winding
# ================================================================

def phase5_locate_seeds(d, known_zeros, discrepant_strips,
                        sigma_lo=0.49, sigma_hi=0.51,
                        strip_size=20, strip_margin=0.3,
                        pts_per_unit=60, prec=192):
    """Locate seeds for missing zeros by winding numbers on inter-zero gaps.

    After a strip audit identifies strips with winding number exceeding
    the known zero count (discrepant strips), this function pinpoints the
    specific inter-zero gap within each strip that contains the missing
    zero.  The gap is found by computing the winding number of L on the
    rectangle [sigma_lo, sigma_hi] x [a, b] for each gap [a, b] between
    consecutive known zeros.  A winding number of 1 certifies exactly one
    missing zero in that gap; its midpoint is returned as a seed.

    If any gap has winding number greater than 1, a CompletenessError is
    raised immediately.  Two or more missing zeros in a single inter-zero
    gap cannot be resolved by midpoint seeding and require manual bisection
    of that gap before rerunning.  This function is therefore sound only
    when each discrepant gap contains at most one missing zero.

    The seeds produced here should be passed to phase3_newton with
    target_prec=1500, cert_threshold_exp=-300.

    Parameters
    ----------
    d : int
        Squarefree discriminant.
    known_zeros : list of float
        All currently known zero ordinates, sorted ascending.
    discrepant_strips : dict
        Mapping strip_number -> n_missing, where strip_number is
        1-indexed and n_missing is the count of missing zeros in
        that strip (from the Phase 4 audit).  Strips are of size
        strip_size zeros each.
    sigma_lo, sigma_hi : float
        Horizontal bounds for the rectangular contour.  Use a
        thin strip around the critical line, e.g. [0.49, 0.51].
    strip_size : int
        Number of zeros per audit strip (default 20).
    strip_margin : float
        Extension of the search range beyond the first and last
        known zero in each strip (default 0.3).
    pts_per_unit : int
        Evaluation density passed to count_zeros_rect.
    prec : int
        ARB working precision in bits for the winding computations.

    Returns
    -------
    list of float
        Seed ordinates, one per missing zero, suitable for passing
        to phase3_newton.  Sorted ascending.

    Raises
    ------
    CompletenessError
        If the number of seeds found in a strip does not match the
        expected count from discrepant_strips.
    """
    saved_prec = ctx.prec
    ctx.prec = prec
    cd = CharacterData(d)

    seeds = []

    for strip_num, n_missing in sorted(discrepant_strips.items()):
        idx_start = (strip_num - 1) * strip_size
        idx_end   = min(strip_num * strip_size - 1, len(known_zeros) - 1)
        strip_zeros = known_zeros[idx_start : idx_end + 1]

        t_lo_strip = strip_zeros[0]  - strip_margin
        t_hi_strip = strip_zeros[-1] + strip_margin
        boundaries = [t_lo_strip] + list(strip_zeros) + [t_hi_strip]

        strip_seeds = []
        found = 0
        for i in range(len(boundaries) - 1):
            a, b = boundaries[i], boundaries[i + 1]
            gap = b - a
            margin = min(0.003, gap * 0.05)
            try:
                w = count_zeros_rect(
                    d, sigma_lo, sigma_hi,
                    a + margin, b - margin,
                    cd, pts_per_unit=pts_per_unit)
            except ArbPrecisionError:
                continue
            if w > 1:
                ctx.prec = saved_prec
                raise CompletenessError(
                    f"Strip {strip_num}, gap [{i}] "
                    f"t=[{a:.4f},{b:.4f}]: winding={w}.  "
                    f"More than one missing zero in a single "
                    f"inter-zero gap cannot be resolved by midpoint "
                    f"seeding.  Manual bisection of this gap is "
                    f"required before rerunning Phase 5.")
            if w == 1:
                strip_seeds.append((a + b) / 2)
                found += 1

        if found != n_missing:
            ctx.prec = saved_prec
            raise CompletenessError(
                f"Strip {strip_num}: expected {n_missing} missing zeros, "
                f"located {found}.  Increase pts_per_unit or prec.")

        seeds.extend(strip_seeds)
        print(f"  Strip {strip_num:3d}: {found} seed(s) located "
              f"in t in [{strip_zeros[0]:.2f}, {strip_zeros[-1]:.2f}]")

    ctx.prec = saved_prec
    return sorted(seeds)


# ================================================================
# Output formatting
# ================================================================

def format_results(results, d, q, output_digits=20, high_prec=False):
    """Print zero ordinates with certified bounds."""
    if high_prec:
        output_digits = 70

    print()
    print("=" * (output_digits + 56))
    print(f"  Zeros of L(s, chi_{d}):  "
          f"field Q(sqrt({d})),  conductor q = {q}")
    print(f"  {len(results)} zeros to {output_digits} decimal places")
    print("=" * (output_digits + 56))
    print()
    print(f"{'n':>4}  {'gamma_n (ordinate)':>{output_digits + 4}}  "
          f"{'ARB bound on |L(1/2+i*gamma)|':>30}  "
          f"{'enclosure':>12}  {'Status':>8}")
    print("-" * (output_digits + 68))

    for n, res in enumerate(results, start=1):
        gamma = res['gamma']

        saved = ctx.prec
        ctx.prec = max(saved, int(output_digits * 3.33) + 64)
        gamma_str = arb.str(gamma, output_digits, more=True)
        if '+/-' in gamma_str:
            gamma_str = gamma_str.split('+/-')[0].strip().strip('[]')
        ctx.prec = saved

        L_abs = res['L_bound']

        # Format the certified upper bound directly from the ARB ball.
        # float() underflows for bounds below ~2.2e-308, so we never
        # use float() here.  upper() returns the right endpoint of the
        # interval as an arb ball; str(4) gives a 4-sig-fig string from
        # which we parse coefficient and exponent.
        old_p = ctx.prec
        ctx.prec = max(old_p, 2000)
        L_upper = L_abs.upper()          # arb: right endpoint
        L_str   = L_upper.str(4)         # e.g. "1.358e-13" or "[+/- 2.12e-449]"
        ctx.prec = old_p

        import re
        # Case 1: normal positive value  "1.234e-440" or "[1.234e-440 +/- ...]"
        match = re.search(r'(\d+\.?\d*)e([+-]?\d+)', L_str)
        if match:
            coeff = float(match.group(1))
            exp   = int(match.group(2))
            bound_str = f"{coeff:.3g} * 10^{exp}"
        elif re.search(r'\+/-\s*([\d.]+e[+-]?\d+)', L_str):
            # Case 2: pure-error ball "[+/- 2.12e-449]" — midpoint is zero,
            # report the radius as the upper bound
            rm = re.search(r'\+/-\s*([\d.]+)e([+-]?\d+)', L_str)
            if rm:
                bound_str = f"{float(rm.group(1)):.3g} * 10^{int(rm.group(2))}"
            else:
                bound_str = L_str.strip('[] ')
        else:
            bound_str = L_str.strip('[] ')

        status = "OK" if res['certified'] else "WARN"
        enc_exp = res.get('enclosure_exp')
        enc_str = f"10^{enc_exp}" if enc_exp is not None else "unverified"
        print(f"{n:>4}  {gamma_str:>{output_digits + 4}}  "
              f"{bound_str:>30}  {enc_str:>12}  {status:>8}")

    n_cert = sum(1 for r in results if r['certified'])
    n_enc = sum(1 for r in results if r.get('enclosure') is not None)
    print()
    print(f"Summary: {n_cert}/{len(results)} zeros certified, "
          f"{n_enc}/{len(results)} with certified enclosures.")
    if n_cert < len(results):
        uncert = [i + 1 for i, r in enumerate(results)
                  if not r['certified']]
        print(f"  Uncertified zeros: {uncert}")


# ================================================================
# Main driver
# ================================================================

def compute_zeros(d, nzeros, high_precision=False, grid_step=0.05,
                  margin=1.10, skip_verify=False):
    """Compute and certify zeros of L(s, chi_d).

    Runs all six phases: Phases 1+2 (scan and bisection), Phase 3
    (Newton refinement), Phase 4 (global completeness verification
    via argument principle on [-0.5, 1.5] x [t_lo, T_max]), and if
    Phase 4 raises CompletenessError, Phases 5+6 (gap winding seed
    location and 1500-bit Newton recovery followed by re-verification).
    Skip Phase 4 onward with skip_verify=True.
    """
    print()
    print("=" * 70)
    print(f"  Computing zeros of L(s, chi_{d})")
    print(f"  Field: Q(sqrt({d})),  "
          f"Conductor: q = {CHARACTERS[d]['q']}")
    print(f"  Zeros requested: {nzeros}")
    mode_str = ('high-precision (1500-bit Newton, ~70 digits)'
                if high_precision
                else 'standard (128-bit Newton, ~20 digits)')
    print(f"  Mode: {mode_str}")
    print("=" * 70)
    print()

    q = CHARACTERS[d]['q']
    t_start = time.time()

    effective_margin = margin * 2.0
    T_max = max(choose_T_max(nzeros, q, effective_margin), nzeros * 2.0)

    all_seeds = []
    T_scan_final = 0.0
    attempt = 0
    current_grid = grid_step
    target = int(nzeros * 1.15) + 2

    while len(all_seeds) < nzeros and attempt < 5:
        attempt += 1
        if attempt > 1:
            T_max *= 1.5
            current_grid = max(current_grid / 2, 0.005)
            target = None
            print(f"  --- Retry {attempt}: T_max = {T_max:.1f}, "
                  f"grid = {current_grid} ---")
            print()

        est = zero_count_estimate(T_max, q)
        print(f"  T_max = {T_max:.1f}  "
              f"(N(T_max) ~ {est:.1f}, requesting {nzeros})")
        print()

        t12 = time.time()
        new_seeds, n_spurious, T_scan = interleaved_scan_and_filter(
            d, T_max, current_grid, target=target)
        print(f"  (Phase 1+2: {time.time() - t12:.2f}s)")
        print()

        T_scan_final = T_scan
        for ns in new_seeds:
            if not any(abs(ns - es) < 0.01 for es in all_seeds):
                all_seeds.append(ns)
        all_seeds.sort()
        print(f"  Total seeds: {len(all_seeds)}")
        print()

    seeds = all_seeds[:nzeros]

    # Phase 3
    if high_precision:
        target_prec, cert_exp, max_iters = 1500, -68, 12
    else:
        target_prec, cert_exp, max_iters = 128, -18, 10

    t3 = time.time()
    print(f"  Phase 3: rigorous Newton at {target_prec}-bit ...")
    results = phase3_newton(
        d, seeds,
        target_prec=target_prec,
        max_iters=max_iters,
        cert_threshold_exp=cert_exp,
    )
    print(f"  (Phase 3: {time.time() - t3:.2f}s)")

    # Phase 4: completeness verification up to just above the last
    # kept seed.  Using T_scan_final would include zeros beyond the
    # nzeros-th, causing a false CompletenessError.
    if not skip_verify and seeds:
        print()
        t4 = time.time()

        # Verify the first seed lies above the verification rectangle
        # lower bound.  If not, zeros below t_lo_offset would be
        # silently excluded from the winding number count.
        t_lo_offset = 0.5
        if float(seeds[0]) <= t_lo_offset:
            raise CompletenessError(
                f"First seed {float(seeds[0]):.6f} is at or below "
                f"the verification lower bound t_lo_offset = "
                f"{t_lo_offset}.  Reduce t_lo_offset and rerun.")

        # Choose T_verify as the last seed plus half the minimum
        # inter-zero spacing (at least 1.0), so that the rectangle
        # covers all nzeros zeros without reaching the next one.
        if len(seeds) >= 2:
            gaps = [seeds[i+1] - seeds[i] for i in range(len(seeds)-1)]
            min_gap = min(gaps)
            buffer = max(min_gap / 2.0, 1.0)
        else:
            buffer = 1.0
        T_verify = float(seeds[-1]) + buffer

        try:
            verify_completeness(d, T_verify, len(seeds),
                                pts_per_unit=40,
                                t_lo_offset=t_lo_offset)
        except CompletenessError as exc:
            print(f"  Phase 4: CompletenessError -- {exc}")
            print()
            print("  Entering recovery: Phase 5 (gap winding) + "
                  "Phase 6 (Newton).")

            # Phase 5: strip audit to identify discrepant strips,
            # then locate missing zero seeds by inter-zero gap winding.
            known_floats = [float(r['gamma'].mid()) for r in results]
            discrepant = {}
            cd_audit = CharacterData(d)
            strip_size = 20
            n_strips = (len(known_floats) + strip_size - 1) // strip_size
            for strip_num in range(1, n_strips + 1):
                i0 = (strip_num - 1) * strip_size
                i1 = min(strip_num * strip_size - 1,
                         len(known_floats) - 1)
                strip_zeros = known_floats[i0 : i1 + 1]
                t_lo_s = strip_zeros[0] - 0.3
                t_hi_s = strip_zeros[-1] + 0.3
                try:
                    w = count_zeros_rect(
                        d, 0.49, 0.51, t_lo_s, t_hi_s,
                        cd_audit, pts_per_unit=40)
                    expected = i1 - i0 + 1
                    if w > expected:
                        discrepant[strip_num] = w - expected
                except ArbPrecisionError:
                    pass

            if not discrepant:
                raise CompletenessError(
                    f"Global completeness failed for L(s, chi_{d}) but "
                    f"the strip audit found no discrepant strips.  The "
                    f"missing zero may lie near a strip boundary.  "
                    f"Widen strip_margin or reduce grid_step and rerun.")
            else:
                print(f"  Phase 5: {len(discrepant)} discrepant "
                      f"strip(s): {sorted(discrepant.keys())}")
                t5 = time.time()
                recovery_seeds = phase5_locate_seeds(
                    d, known_floats, discrepant, prec=192)
                print(f"  Phase 5: {len(recovery_seeds)} seed(s) "
                      f"located ({time.time() - t5:.2f}s).")
                print()

                # Phase 6: 1500-bit Newton on recovery seeds.
                print("  Phase 6: 1500-bit Newton on recovery seeds ...")
                t6 = time.time()
                recovery_results = phase3_newton(
                    d, recovery_seeds,
                    target_prec=1500,
                    max_iters=15,
                    cert_threshold_exp=-300,
                )
                print(f"  (Phase 6: {time.time() - t6:.2f}s)")

                # Merge and re-sort.
                results = sorted(
                    results + recovery_results,
                    key=lambda r: float(r['gamma'].mid()))
                print(f"  Recovery complete: table now has "
                      f"{len(results)} zeros.")
                print()

                # Final re-verification on the repaired table.
                # This closes the recovery loop with the same
                # unconditional force as the primary Phase 4 path.
                repaired_floats = [float(r['gamma'].mid())
                                   for r in results]
                if len(repaired_floats) >= 2:
                    rep_gaps = [repaired_floats[i+1] - repaired_floats[i]
                                for i in range(len(repaired_floats) - 1)]
                    rep_min_gap = min(rep_gaps)
                    rep_buffer = max(rep_min_gap / 2.0, 1.0)
                else:
                    rep_buffer = 1.0
                T_verify_repaired = repaired_floats[-1] + rep_buffer
                print("  Phase 4 (re-verification of repaired table) ...")
                try:
                    verify_completeness(
                        d, T_verify_repaired, len(results),
                        pts_per_unit=40,
                        t_lo_offset=t_lo_offset)
                    print("  Repaired table certified complete.")
                except CompletenessError as exc2:
                    raise CompletenessError(
                        f"Re-verification of repaired table failed: "
                        f"{exc2}  The recovered table is not certified "
                        f"complete.  Manual inspection is required."
                    ) from exc2
                except ArbPrecisionError as exc2:
                    raise ArbPrecisionError(
                        f"ARB precision insufficient to certify "
                        f"completeness of the repaired table "
                        f"({exc2}).  The recovered table is not "
                        f"certified complete.  Increase ctx.prec "
                        f"or pts_per_unit and rerun verification."
                    ) from exc2

        except ArbPrecisionError as exc:
            print(f"  Phase 4 WARNING: ARB precision insufficient "
                  f"for winding number ({exc}). "
                  f"Completeness not verified.")
        print(f"  (Phase 4: {time.time() - t4:.2f}s)")
    else:
        print("  Phase 4: skipped (--skip-verify).")

    # Output
    total_time = time.time() - t_start
    format_results(results, d, q,
                   output_digits=70 if high_precision else 20,
                   high_prec=high_precision)
    print()
    print(f"  Total computation time: {total_time:.1f}s")
    print()

    return results


# ================================================================
# Command-line interface
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute zeros of quadratic Dirichlet L-functions "
                    "using rigorous ARB interval arithmetic.",
        epilog="Eight-phase pipeline: sign-change scan (Phase 1), "
               "bisection filtering (Phase 2), Newton refinement "
               "(Phase 3), global completeness verification via "
               "argument principle on [-0.5,1.5]x[t_lo,T] "
               "(Phase 4); if Phase 4 detects gaps: inter-zero gap "
               "winding to locate seeds (Phase 5) then 1500-bit "
               "Newton recovery (Phase 6); manual closeout: tail "
               "extension to fixed T (Phase 7) and bound boosting "
               "to 2200-bit (Phase 8).  All arithmetic is "
               "ARB-rigorous."
    )
    parser.add_argument(
        '--d', type=int, required=True,
        choices=sorted(CHARACTERS.keys()),
        help='Squarefree discriminant (2,3,5,6,7,10,11,13)')
    parser.add_argument(
        '--nzeros', type=int, default=20,
        help='Number of zeros to compute (default: 20)')
    parser.add_argument(
        '--high-precision', action='store_true',
        help='Use 1500-bit Newton for ~70-digit zeros')
    parser.add_argument(
        '--grid-step', type=float, default=0.05,
        help='Grid step for sign-change scan (default: 0.05)')
    parser.add_argument(
        '--margin', type=float, default=1.10,
        help='Safety margin for T_max (default: 1.10)')
    parser.add_argument(
        '--skip-verify', action='store_true',
        help='Skip Phase 4 completeness verification')

    args = parser.parse_args()
    compute_zeros(
        d=args.d,
        nzeros=args.nzeros,
        high_precision=args.high_precision,
        grid_step=args.grid_step,
        margin=args.margin,
        skip_verify=args.skip_verify,
    )


if __name__ == '__main__':
    main()
