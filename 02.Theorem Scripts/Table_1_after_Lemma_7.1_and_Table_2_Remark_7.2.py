"""
Lemma_7_1(Monotonicity).py  --  ARB-rigorous certification of tables in Section 7
===================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

Lemma 7.1 itself (monotonicity of the unsigned main-term integral) requires
no computation: |J_0(b_{M+1} t)| <= 1 pointwise is purely analytic.

This script certifies the two numerical tables associated with Section 7.

TABLE 1 (unsigned main-term integrals, after Lemma 7.1):

    (1/pi) int_0^{2000} prod_{k=1}^M |J_0(b_k t)| dt

for M in {5, 10, 15, 20, 25, 30}, both L-function (chi_5) and Riemann
zeta weights.

    M     L-function    Zeta
    -------------------------
    5     10.877        47.702
    10    10.585        46.119
    15    10.535        45.727
    20    10.517        45.564
    25    10.508        45.479
    30    10.504        45.430

TABLE 2 (unsigned vs signed, Remark 7.2):

    (1/pi) int_0^{2000} prod_{k=1}^M |J_0(b_k t)| dt   (unsigned)
    (1/pi) int_0^{2000} prod_{k=1}^M  J_0(b_k t)  dt   (signed)

for M in {3, 10, 20}, L-function (chi_5) weights only.

    M     Unsigned    Signed
    -------------------------
    3     12.367      7.838
    10    10.585      8.292
    20    10.517      8.313

All arithmetic is ARB interval arithmetic throughout.  No mpmath or
floating-point library is used in any load-bearing computation.

Tail bound: the analytic envelope bound |J_0(x)| <= sqrt(2/(pi*x))
(DLMF 10.14.1) gives a rigorous ARB upper bound on the tail
(1/pi) int_T^inf prod|J_0| dt, reported per row.  For M >= 10 the
bound is tight; for M = 3 and M = 5 it is O(1) to O(0.1) and
conservative -- the table values are the [0, T] integrals as stated.

Algorithm
---------
UNSIGNED integrals: strip decomposition at every J_0 zero, identical
architecture to Remark_6_12(J0_Role).py.  acb.integral is applied
only on gap intervals where the product is analytic and sign-definite.

SIGNED integrals: the integrand prod J_0(b_k t) / pi is analytic on
(0, T] (no absolute values).  acb.integral is applied directly
over the full interval [T_EPS, T].

J_0 computed via J_0(z) = (2/z)J_1(z) - J_2(z) (DLMF 10.6.1).
J_0 zeros: McMahon (DLMF 10.21.19, nu=0) with BRACKET = 1.5.

PASS/FAIL: certified ARB predicate  rel_err < REL_TOL,
where < is the ARB operator (returns True only when the entire ball of
rel_err lies strictly below the lower bound of REL_TOL).

Zero data:
  L-function: first 30 certified zeros of L(s, chi_5) from
    L_function_zeros.py (Zenodo doi:10.5281/zenodo.18783098),
    70 decimal places, |L(1/2 + i*gamma_k', chi_5)| < 10^{-449}.
  Zeta: first 30 zeros from zeta_zeros.py (LMFDB, 31 decimal places).

Working precision: 256-bit ARB throughout.
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros as get_L_zeros
from zeta_zeros import get_zeros as get_Z_zeros
from flint import arb, acb, ctx

# ── Precision and parameters ──────────────────────────────────────────────────
ARB_PREC        = 256
MAX_BISECT_PREC = 1536
T_UPPER         = 2000
T_EPS           = arb("1e-30")
T_MAX_ARB       = arb(T_UPPER)
REL_TOL_INT     = 2**(-200)
DELTA           = arb("1e-20")

# 0.1% relative tolerance; certified ARB comparison throughout.
REL_TOL = arb("1e-3")

# ── Paper table values (ARB-certified strip-decomposition results) ─────────────
# These are the values used in the paper.

# Table 1: unsigned main-term integrals, M in {5,10,15,20,25,30}
PAPER_T1_L = {5: "10.877", 10: "10.585", 15: "10.535",
              20: "10.517", 25: "10.508", 30: "10.504"}
PAPER_T1_Z = {5: "47.702", 10: "46.119", 15: "45.727",
              20: "45.564", 25: "45.479", 30: "45.430"}
M_VALUES_T1 = [5, 10, 15, 20, 25, 30]

# Table 2: unsigned and signed, M in {3,10,20}, L-function only
PAPER_T2_UNSIGNED = {3: "12.367", 10: "10.585", 20: "10.517"}
PAPER_T2_SIGNED   = {3:  "7.838", 10:  "8.292", 20:  "8.313"}
M_VALUES_T2 = [3, 10, 20]

# ── ARB constants ─────────────────────────────────────────────────────────────
_ZERO = arb(0)
_TWO  = arb(2)
# No module-level pi: arb.pi() called fresh inside functions.

# ── Bessel helpers ────────────────────────────────────────────────────────────

def J0_real(x_arb):
    """J_0(x) as ARB ball via DLMF 10.6.1 recurrence."""
    z = acb(x_arb)
    return (acb(_TWO) * acb.bessel_j(z, acb(1)) / z
            - acb.bessel_j(z, acb(_TWO))).real


def J0_acb(z_acb):
    """J_0(z) for complex argument via DLMF 10.6.1."""
    return (acb(_TWO) * acb.bessel_j(z_acb, acb(1)) / z_acb
            - acb.bessel_j(z_acb, acb(_TWO)))


# ── J_0 zero computation ──────────────────────────────────────────────────────

def _mcmahon_seed_J0(m_int):
    """McMahon seed for m-th zero of J_0, pure ARB (DLMF 10.21.19, nu=0)."""
    m    = arb(m_int)
    beta = (m - arb("1/4")) * arb.pi()   # fresh pi at current ctx.prec
    # nu=0: mu = 4*0^2 = 0, so (mu-1) = -1
    t1   = arb(-1) / (arb(8) * beta)
    t2   = arb(4) * arb(-1) * arb(-31) / (arb(3) * (arb(8) * beta) ** 3)
    t3   = arb(32) * arb(-1) * arb(3779) / (arb(15) * (arb(8) * beta) ** 5)
    return beta - t1 - t2 - t3


def arb_J0_zero(m_int):
    """
    m-th positive zero of J_0 via certified trichotomy bisection, pure ARB.
    McMahon seed with BRACKET = 1.5; precision doubled to MAX_BISECT_PREC
    if sign is undecidable at base precision.
    """
    seed    = _mcmahon_seed_J0(m_int)
    BRACKET = arb("1.5")
    lo      = seed - BRACKET
    hi      = seed + BRACKET
    if lo < arb("0.5"):
        lo = arb("0.5")

    flo = J0_real(lo)
    fhi = J0_real(hi)
    if not (((flo > _ZERO) and (fhi < _ZERO)) or
            ((flo < _ZERO) and (fhi > _ZERO))):
        raise RuntimeError(
            f"Initial bracket failed for J_0 zero #{m_int} "
            f"(seed ~ {float(seed.mid()):.4f})"
        )

    TARGET = DELTA * arb("0.1")
    for _it in range(500):
        if (hi - lo) < TARGET:
            break
        mid  = (lo + hi) / _TWO
        prec = ctx.prec
        while prec <= MAX_BISECT_PREC:
            old_prec  = ctx.prec
            ctx.prec  = prec
            fmid      = J0_real(mid)
            ctx.prec  = old_prec
            mid_pos   = fmid > _ZERO
            mid_neg   = fmid < _ZERO
            if mid_pos or mid_neg:
                break
            prec *= 2
        if not mid_pos and not mid_neg:
            raise RuntimeError(
                f"Sign undecidable at MAX_BISECT_PREC={MAX_BISECT_PREC} "
                f"for J_0 zero #{m_int}."
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

def collect_strips_J0(b_k_arb, T_max_arb):
    """
    IVT-certified strip intervals for all zeros of J_0(b_k t) in (0, T_max].
    Architecture identical to Lemma_6_6 and Remark_6_12.
    """
    T_x    = b_k_arb * T_max_arb
    strips = []
    prev_s = None
    m      = 1

    while True:
        z   = arb_J0_zero(m)
        flo = J0_real(z - DELTA)
        fhi = J0_real(z + DELTA)
        if not (((flo > _ZERO) and (fhi < _ZERO)) or
                ((flo < _ZERO) and (fhi > _ZERO))):
            raise RuntimeError(
                f"IVT FAILED for J_0 zero #{m} (z ~ {float(z.mid()):.6f})"
            )
        if (z - DELTA) > T_x:
            break
        if prev_s is not None and not ((z - DELTA) > (prev_s + DELTA)):
            raise RuntimeError(f"Non-overlap FAILED: J_0 zeros #{m-1} and #{m}")
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
    Collect strips from all M J_0 factors and merge into a single
    certified-ascending list using a K-way merge on ARB comparisons.

    Each per-factor list is already in certified ascending order
    (m increments, j_{0,m} is strictly increasing).  The K-way merge
    uses only ARB < comparisons to select the next strip at each step.
    """
    # Collect one sorted list per factor; each is ARB-ascending by construction.
    per_factor = [collect_strips_J0(b_arb[k], T_max_arb) for k in range(M_int)]
    ptrs       = [0] * M_int   # current index into each per-factor list

    merged = []
    while True:
        # Find all factors that still have strips remaining.
        active = [k for k in range(M_int) if ptrs[k] < len(per_factor[k])]
        if not active:
            break

        # Select the factor whose current strip starts earliest.
        # Start with the first active factor as candidate.
        best = active[0]
        for k in active[1:]:
            # ARB certified comparison: is k's strip start < best's strip start?
            if per_factor[k][ptrs[k]][0] < per_factor[best][ptrs[best]][0]:
                best = k
            elif not (per_factor[best][ptrs[best]][0] < per_factor[k][ptrs[k]][0]):
                # Neither is certified smaller: the two strip starts overlap in ARB.
                # This cannot happen for distinct Bessel zeros of different factors
                # (the zeros of J_0(b_i t) and J_0(b_j t) are certified distinct in this computation
                # and DELTA = 1e-20 is far smaller than any inter-zero gap), but we
                # guard explicitly.
                raise RuntimeError(
                    f"ARB ordering undecidable between factor {best} strip {ptrs[best]} "
                    f"and factor {k} strip {ptrs[k]}: strip starts overlap at 256-bit "
                    f"precision.  Increase ARB_PREC or decrease DELTA."
                )
        merged.append(per_factor[best][ptrs[best]])
        ptrs[best] += 1

    # Paranoia check: verify the merged list is certified ascending.
    for i in range(len(merged) - 1):
        if not (merged[i][0] < merged[i + 1][0]):
            raise RuntimeError(
                f"Post-merge certified ordering FAILED at positions {i} and {i+1}"
            )
    return merged


def build_gaps(strips, T_max_arb):
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


# ── Integration ───────────────────────────────────────────────────────────────

def integrate_unsigned(b_arb, M_int, T_upper=None):
    """
    (1/pi) int_0^T prod_{k=1}^M |J_0(b_k t)| dt  via strip decomposition.
    acb.integral applied on gap intervals only (analytic, sign-definite).
    T_upper defaults to the module-level T_UPPER (2000).
    Returns ARB ball.
    """
    ctx.prec = ARB_PREC
    T_arb  = arb(T_upper) if T_upper is not None else T_MAX_ARB
    b_acb  = [acb(bk) for bk in b_arb[:M_int]]
    strips = collect_all_strips(b_arb, M_int, T_arb)
    gaps   = build_gaps(strips, T_arb)

    I = _ZERO
    for (t_lo, t_hi) in gaps:
        def f(t, _, M=M_int, b=b_acb):
            r = J0_acb(t * b[0])
            for k in range(1, M):
                r = r * J0_acb(t * b[k])
            return r / acb(arb.pi())
        I = I + abs(acb.integral(f, acb(t_lo), acb(t_hi),
                                  rel_tol=REL_TOL_INT, eval_limit=10**7).real)

    # Strip error bound: each J_0 factor is <= 1, strip width is 2*DELTA/b_k
    se = _ZERO
    for (t_lo, t_hi) in strips:
        se = se + (t_hi - t_lo) / acb(arb.pi()).real   # product of M factors <= 1
    return I + se


def integrate_signed(b_arb, M_int):
    """
    (1/pi) int_0^T prod_{k=1}^M J_0(b_k t) dt  -- no absolute values.
    Integrand is analytic; acb.integral applied over full [T_EPS, T].
    """
    ctx.prec = ARB_PREC
    b_acb    = [acb(bk) for bk in b_arb[:M_int]]

    def f(t, _):
        r = J0_acb(t * b_acb[0])
        for k in range(1, M_int):
            r = r * J0_acb(t * b_acb[k])
        return r / acb(arb.pi())

    return acb.integral(f, acb(T_EPS), acb(T_MAX_ARB),
                        rel_tol=REL_TOL_INT, eval_limit=10**7).real


# ── Certified analytic tail bound ────────────────────────────────────────────

def certified_tail_bound(b_arb, M_int, T_arb):
    """
    ARB-rigorous upper bound on (1/pi) int_T^inf prod_{k=1}^M |J_0(b_k t)| dt.

    Uses Watson's bound |J_0(x)| <= sqrt(2/(pi x)) for x > 0 (valid for all
    nu >= -1/2 by Watson 1944).  Factors with b_k * T >= 1 (certified ARB)
    are classed as "decaying" and contribute sqrt(2/(pi b_k t)); any remaining
    factors are bounded by 1 (conservative).

    For m decaying factors (m >= 3), integrating from T to infinity gives:
      (1/pi) (2/pi)^{m/2} (prod_{decaying} b_k)^{-1/2} T^{1-m/2} / (m/2 - 1).

    This bound can be very conservative for small M because Watson's bound
    ignores the oscillatory cancellation that makes the actual integral small.
    It is a valid rigorous upper bound regardless.

    INTERPRETATION: the paper values are T_UPPER = 2000 truncated integrals.
    The PASS/FAIL in this script compares T=2000 script values to T=2000 paper
    values and is therefore self-consistent regardless of the tail.  This
    function documents the unverified truncation error for a reader who wants
    to know how close the T=2000 values are to the true integrals on [0, inf).
    For the unsigned integrals used as upper bounds in the paper's theorems,
    the T=2000 value is a lower bound on int_0^inf, so the reported bound
    is conservative (weaker than the truth).
    """
    decaying = [b_arb[k] for k in range(M_int)
                if bool(b_arb[k] * T_arb >= arb(1))]
    m = len(decaying)
    if m < 3:
        # Not enough certified-decaying factors to bound the integral; return
        # infinity as a placeholder (should not occur for T=2000, M>=3, chi_5/zeta).
        return arb("inf")
    prod_b = arb(1)
    for b in decaying:
        prod_b = prod_b * b
    two_over_pi = arb(2) / arb.pi()
    return (arb(1) / arb.pi())            * two_over_pi ** (arb(m) / arb(2))            * prod_b ** arb("-1/2")            * T_arb ** (arb(1) - arb(m) / arb(2))            / (arb(m) / arb(2) - arb(1))


# ── Weight loading ────────────────────────────────────────────────────────────

def arb_tail_bound(b_arb, M_int, T_val):
    """
    ARB-certified upper bound on (1/pi) int_T^inf prod_{k=1}^M |J_0(b_k t)| dt.

    Uses the uniform envelope bound |J_0(x)| <= sqrt(2/(pi*x)) (DLMF 10.14.1),
    giving prod|J_0(b_k t)| <= (2/pi)^{M/2} * (prod b_k)^{-1/2} * t^{-M/2}
    and hence

        tail <= (1/pi) * (2/pi)^{M/2} * (prod b_k)^{-1/2} * T^{1-M/2} / (M/2 - 1).

    This is a valid ARB-rigorous upper bound for all M >= 3.  It is conservative
    by a factor of roughly (pi/2)^M relative to the true tail because the envelope
    is the oscillation peak rather than the cycle average; for M >= 10 the bound
    is tight to within two significant figures, while for M = 3 and M = 5 it is
    O(1) or O(0.1) even though the actual tail is smaller.
    """
    ctx.prec = ARB_PREC
    M_arb   = arb(M_int)
    T_arb   = arb(T_val)
    prod_b  = arb(1)
    for k in range(M_int):
        prod_b = prod_b * b_arb[k]
    two_over_pi = arb(2) / arb.pi()
    coeff  = two_over_pi ** (M_arb / arb(2)) * prod_b ** arb("-1/2")
    return coeff * T_arb ** (arb(1) - M_arb / arb(2)) / (M_arb / arb(2) - arb(1)) / arb.pi()


def load_weights_L():
    """First 30 Lorentzian weights for chi_5, ARB at ARB_PREC bits."""
    ctx.prec = ARB_PREC
    return [arb(2) / (arb("1/4") + arb(g)**2)
            for g in get_L_zeros(5, 30, as_strings=True)]


def load_weights_Z():
    """First 30 Lorentzian weights for zeta, ARB at ARB_PREC bits."""
    ctx.prec = ARB_PREC
    return [arb(2) / (arb("1/4") + arb(g)**2)
            for g in get_Z_zeros(30, as_strings=True)]


# ── Certified PASS/FAIL ───────────────────────────────────────────────────────

def arb_matches(val_arb, paper_str):
    """
    Return True iff |val_arb - paper_val| / paper_val < REL_TOL,
    where < is the ARB certified predicate.
    """
    pv      = arb(paper_str)
    rel_err = abs(val_arb - pv) / abs(pv)
    return bool(rel_err < REL_TOL)


# ── Certification ─────────────────────────────────────────────────────────────

def certify_table1(b_L, b_Z):
    """Certify Table 1: unsigned integrals for both weight sets."""
    ctx.prec = ARB_PREC
    results  = []
    for M in M_VALUES_T1:
        t0  = time.time()
        IL  = integrate_unsigned(b_L, M)
        IZ  = integrate_unsigned(b_Z, M)
        elapsed = time.time() - t0
        TBL = arb_tail_bound(b_L, M, T_UPPER)
        TBZ = arb_tail_bound(b_Z, M, T_UPPER)
        results.append(dict(
            M=M, IL=IL, IZ=IZ,
            IL_f=float(IL.mid()), IZ_f=float(IZ.mid()),
            IL_match=arb_matches(IL, PAPER_T1_L[M]),
            IZ_match=arb_matches(IZ, PAPER_T1_Z[M]),
            TBL=TBL, TBZ=TBZ,
            elapsed=elapsed,
        ))
    return results


def certify_table2(b_L):
    """Certify Table 2 (Remark 7.2): unsigned and signed, L-function."""
    ctx.prec = ARB_PREC
    results  = []
    for M in M_VALUES_T2:
        t0  = time.time()
        Iu  = integrate_unsigned(b_L, M)
        Is  = integrate_signed(b_L, M)
        elapsed = time.time() - t0
        TB = arb_tail_bound(b_L, M, T_UPPER)
        results.append(dict(
            M=M, Iu=Iu, Is=Is,
            Iu_f=float(Iu.mid()), Is_f=float(Is.mid()),
            Iu_match=arb_matches(Iu, PAPER_T2_UNSIGNED[M]),
            Is_match=arb_matches(Is, PAPER_T2_SIGNED[M]),
            TB=TB,
            elapsed=elapsed,
        ))
    return results


# ── Output ────────────────────────────────────────────────────────────────────

def print_table1(results):
    print()
    print("Table 1 -- unsigned main-term integrals (1/pi) int prod|J_0| dt")
    print("=" * 68)
    print(f"  {'M':>4}  {'L-fn':>9}  {'Paper':>8}  {'Zeta':>9}  "
          f"{'Paper':>8}  {'L':>6}  {'Z':>6}  {'tail_L':>10}  {'tail_Z':>10}  {'Time':>6}")
    print("  " + "-" * 82)
    all_pass = True
    for r in results:
        M = r['M']
        print(f"  {M:>4}  {r['IL_f']:>9.3f}  {PAPER_T1_L[M]:>8}  "
              f"{r['IZ_f']:>9.3f}  {PAPER_T1_Z[M]:>8}  "
              f"{'PASS' if r['IL_match'] else 'FAIL':>6}  "
              f"{'PASS' if r['IZ_match'] else 'FAIL':>6}  "
              f"{float(r['TBL'].mid()):>10.3e}  {float(r['TBZ'].mid()):>10.3e}  "
              f"{r['elapsed']:>5.1f}s")
        if not r['IL_match'] or not r['IZ_match']:
            all_pass = False
    print()
    print("  ARB balls:")
    for r in results:
        print(f"    M={r['M']:2d}: L = {r['IL']}  Z = {r['IZ']}")
    return all_pass


def print_table2(results):
    print()
    print("Table 2 -- Remark 7.2: unsigned vs signed (L-function only)")
    print("=" * 68)
    print(f"  {'M':>4}  {'Unsigned':>10}  {'Paper':>8}  {'Signed':>10}  "
          f"{'Paper':>8}  {'U':>6}  {'S':>6}  {'tail(ub)':>10}  {'Time':>6}")
    print("  " + "-" * 76)
    all_pass = True
    for r in results:
        M = r['M']
        print(f"  {M:>4}  {r['Iu_f']:>10.3f}  {PAPER_T2_UNSIGNED[M]:>8}  "
              f"{r['Is_f']:>10.3f}  {PAPER_T2_SIGNED[M]:>8}  "
              f"{'PASS' if r['Iu_match'] else 'FAIL':>6}  "
              f"{'PASS' if r['Is_match'] else 'FAIL':>6}  "
              f"{float(r['TB'].mid()):>10.3e}  "
              f"{r['elapsed']:>5.1f}s")
        if not r['Iu_match'] or not r['Is_match']:
            all_pass = False
    print()
    print("  ARB balls:")
    for r in results:
        print(f"    M={r['M']:2d}: unsigned = {r['Iu']}  signed = {r['Is']}")
    return all_pass


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ctx.prec = ARB_PREC
    print(f"Loading weights at {ARB_PREC}-bit ARB precision ...")
    b_L = load_weights_L()
    b_Z = load_weights_Z()
    print(f"  chi_5  b_1={float(b_L[0].mid()):.6f}, b_2={float(b_L[1].mid()):.6f}")
    print(f"  zeta   b_1={float(b_Z[0].mid()):.6f}, b_2={float(b_Z[1].mid()):.6f}")
    print()
    print("Certifying Table 1 (unsigned, both weight sets) ...")
    r1 = certify_table1(b_L, b_Z)
    pass1 = print_table1(r1)
    print()
    print("Certifying Table 2 (Remark 7.2, L-function only) ...")
    r2 = certify_table2(b_L)
    pass2 = print_table2(r2)
    print()
    if pass1 and pass2:
        print("RESULT: ALL PASS  [ARB-rigorous]")
        print("  - certified trichotomy bisection at all J_0 zeros")
        print("  - strip decomposition at every cusp (unsigned integrals)")
        print("  - signed integrals via direct acb.integral (analytic)")
        print("  - PASS/FAIL via decisive ARB predicate, no float threshold")
        print("  - tail(ub): ARB-certified envelope bound on (1/pi)int_T^inf prod|J_0|")
        print("    (conservative for M=3,5; tight for M>=10)")
    else:
        raise RuntimeError("Certification failed")

