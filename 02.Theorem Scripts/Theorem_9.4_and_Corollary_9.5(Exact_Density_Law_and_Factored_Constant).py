"""
Theorem_9.4_and_Corollary_9.5(Exact_Density_Law_and_Factored_Constant).py
===================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the three numerical constants in Theorem 9.4 and Corollary 9.5
(Exact density law for quadratic norm-form energies) and
Corollary 9.5 (Factored form of the leading constant):

    f_{S_L}(0)  = 8.3129   (central density of L-function spectral sum)
    E[|S_zeta|] = 0.00717  (mean absolute value of zeta spectral sum)
    C(5)        = 0.1193   (leading constant in the density law)

via the formulas (Corollary 9.5, equations (eq:f0-bessel) and (eq:Eabs-bessel)):

    f_{S_L}(0)  = (1/pi) int_0^inf prod_{k=1}^M J_0(b_k t) dt

    E[|S_zeta|] = (2/pi) int_0^inf (1 - prod_{k=1}^M J_0(a_k t)) / t^2 dt

    C(5)        = 2 * f_{S_L}(0) * E[|S_zeta|]

where the spectral weights are:

    b_k = 2 / (1/4 + gamma'_k^2)   (chi_5 zero ordinates, L-function)
    a_k = 2 / (1/4 + gamma_k^2)    (zeta zero ordinates)

and M = 20 throughout.

RIGOUR ARCHITECTURE
-------------------
All arithmetic is ARB interval arithmetic throughout.  No Python floats
are used in any load-bearing computation.

f_{S_L}(0): computed as (1/pi) * int_0^{T_L} prod J_0(b_k t) dt + tail,
    where T_L = 2000 and the signed tail is bounded in absolute value by
    arb_tail_bound(b, M, T_L) using the uniform Bessel envelope
    |J_0(x)| <= sqrt(2/(pi*x)) (DLMF 10.14.1).  The tail bound at T=2000
    with M=20 chi_5 weights is certified in ARB.

E[|S_zeta|]: split as
    (2/pi) * [int_0^{T_Z} (1 - phi_z(t)) / t^2 dt  +  1/T_Z  +  phi-tail],
    where T_Z = 50000, phi_z(t) = prod_{k=1}^M J_0(a_k t), and the
    analytic correction 1/T_Z = 2e-5 accounts for int_{T_Z}^inf 1/t^2 dt.
    The residual phi-tail |int_{T_Z}^inf phi_z(t)/t^2 dt| is bounded by
    the Bessel envelope; the certified ARB bound at T=50000 is < 3e-25.
    The integrand (1-phi_z)/t^2 has a removable singularity at t=0 with
    limit sigma_zeta^2/2; the Petras algorithm in acb.integral handles
    this by adaptive subdivision.

PASS criterion: certified ARB predicate
    bool(abs(computed - paper) / abs(paper) < REL_TOL) == True
for each of the three values, with REL_TOL = 1e-3.

Zero data:
    First 20 zeros of L(s, chi_5): from L_function_zeros.py,
        Zenodo doi:10.5281/zenodo.18783098, 70 decimal places,
        certified |L(1/2 + i*gamma'_k, chi_5)| < 10^{-449}.
    First 20 zeros of zeta(s): from zeta_zeros.py, LMFDB data,
        31 decimal places.

Working precision: 256-bit ARB / ACB throughout.
Integration: acb.integral (Petras adaptive algorithm), rel_tol = 2^{-200}.
"""

import os
import sys
import time

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros as get_L_zeros
import zeta_zeros as _zz
from flint import arb, acb, ctx

# ── Parameters ────────────────────────────────────────────────────────────────
ARB_PREC    = 256
M           = 20       # truncation level

T_L         = 2000     # upper limit for f_{S_L}(0) integral
T_Z         = 50000    # upper limit for E[|S_zeta|] main integral
T_EPS_STR   = "1e-30"  # lower limit (avoids t=0 singularity)

REL_TOL_INT = 2**(-200)  # quadrature relative tolerance
REL_TOL     = arb("1e-3")  # PASS/FAIL relative tolerance for paper values

# Paper values to certify
PAPER_F0   = "8.3129"
PAPER_EABS = "0.00717"
PAPER_C5   = "0.1193"


# ── Load spectral weights ─────────────────────────────────────────────────────

def load_weights():
    """
    Return (b, a): ARB lists of M Lorentzian weights.
        b_k = 2 / (1/4 + gamma'_k^2)   for chi_5 zeros
        a_k = 2 / (1/4 + gamma_k^2)    for zeta zeros
    """
    ctx.prec = ARB_PREC
    quarter = arb("1/4")
    chi5 = get_L_zeros(5, M, as_strings=True)
    zeta = _zz.get_zeros(M, as_strings=True)
    b = [arb(2) / (quarter + arb(g)**2) for g in chi5]
    a = [arb(2) / (quarter + arb(g)**2) for g in zeta]
    return b, a


# ── Bessel tail bound for signed integral ─────────────────────────────────────

def bessel_tail_bound(weights, M_int, T_arb):
    """
    ARB upper bound on (1/pi) * int_T^inf prod_{k=1}^M |J_0(w_k t)| dt.

    Uses |J_0(x)| <= sqrt(2/(pi*x)) (DLMF 10.14.1) for x >= 1, giving
        prod |J_0(w_k t)| <= (2/pi)^{M/2} * (prod w_k)^{-1/2} * t^{-M/2}
    and integrating:
        tail <= (1/pi) * (2/pi)^{M/2} * (prod w_k)^{-1/2}
                * T^{1-M/2} / (M/2 - 1).

    Valid for M >= 3 and all w_k * T >= 1.
    """
    ctx.prec = ARB_PREC
    M_arb = arb(M_int)
    prod_w = arb(1)
    for w in weights[:M_int]:
        prod_w = prod_w * w
    two_over_pi = arb(2) / arb.pi()
    coeff = two_over_pi ** (M_arb / arb(2)) * prod_w ** arb("-1/2")
    return coeff * T_arb ** (arb(1) - M_arb / arb(2)) / (M_arb / arb(2) - arb(1)) / arb.pi()


def phi_tail_bound(weights, M_int, T_arb):
    """
    ARB upper bound on int_T^inf prod_{k=1}^M |J_0(w_k t)| / t^2 dt.

    Same envelope: integrand <= (2/pi)^{M/2} * (prod w_k)^{-1/2} * t^{-(M/2+2)}.
    Integrating gives T^{-(M/2+1)} / (M/2+1).

    Valid for M >= 1 and all w_k * T >= 1.
    """
    ctx.prec = ARB_PREC
    M_arb = arb(M_int)
    prod_w = arb(1)
    for w in weights[:M_int]:
        prod_w = prod_w * w
    two_over_pi = arb(2) / arb.pi()
    coeff = two_over_pi ** (M_arb / arb(2)) * prod_w ** arb("-1/2")
    exponent = -(M_arb / arb(2) + arb(1))
    denom = M_arb / arb(2) + arb(1)
    return coeff * T_arb ** exponent / denom


# ── Compute f_{S_L}(0) ────────────────────────────────────────────────────────

def compute_f0(b):
    """
    Certify f_{S_L}(0) = (1/pi) int_0^inf prod_{k=1}^M J_0(b_k t) dt.

    Method: signed ARB quadrature on [T_EPS, T_L] plus certified tail bound.
    """
    ctx.prec = ARB_PREC
    b_acb = [acb(bk) for bk in b]
    pi_acb = acb(arb.pi())

    def integrand(t, _):
        ctx.prec = ARB_PREC
        r = acb(1)
        for bk in b_acb:
            r = r * (bk * t).bessel_j(acb(0))
        return r / pi_acb

    integral = acb.integral(integrand,
                            acb(arb(T_EPS_STR)), acb(arb(T_L)),
                            rel_tol=REL_TOL_INT,
                            eval_limit=10**7).real

    T_L_arb = arb(T_L)
    tail    = bessel_tail_bound(b, M, T_L_arb)

    # The tail bound (~9.5e-9 at T=2000) is the dominant uncertainty;
    # the quadrature ball radius (~5e-73) is negligible by comparison.
    # Return them separately so main() can report both.
    return integral, tail, T_L_arb


# ── Compute E[|S_zeta|] ───────────────────────────────────────────────────────

def compute_eabs(a):
    """
    Certify E[|S_zeta|] = (2/pi) int_0^inf (1 - prod J_0(a_k t)) / t^2 dt.

    Split:
        (2/pi) * [int_0^{T_Z} (1 - phi_z(t))/t^2 dt  +  1/T_Z  +  phi-tail]
    where phi-tail = int_{T_Z}^inf phi_z(t)/t^2 dt is bounded by
    phi_tail_bound(a, M, T_Z).  The analytic correction (2/pi)/T_Z accounts
    for int_{T_Z}^inf 1/t^2 dt = 1/T_Z.
    """
    ctx.prec = ARB_PREC
    a_acb = [acb(ak) for ak in a]
    two_over_pi = arb(2) / arb.pi()

    def integrand(t, _):
        ctx.prec = ARB_PREC
        phi = acb(1)
        for ak in a_acb:
            phi = phi * (ak * t).bessel_j(acb(0))
        return (acb(1) - phi) / (t * t)

    integral = acb.integral(integrand,
                            acb(arb(T_EPS_STR)), acb(arb(T_Z)),
                            rel_tol=REL_TOL_INT,
                            eval_limit=10**7).real

    T_Z_arb       = arb(T_Z)
    analytic_tail = arb(1) / T_Z_arb            # int_{T_Z}^inf 1/t^2 dt
    ptail         = phi_tail_bound(a, M, T_Z_arb)

    # E[|S_zeta|] = (2/pi) * (integral + analytic_tail) + (2/pi) * phi-tail-error
    eabs_main = two_over_pi * (integral + analytic_tail)
    return eabs_main, two_over_pi * ptail, T_Z_arb


# ── PASS/FAIL ─────────────────────────────────────────────────────────────────

def arb_matches(val, paper_str, label):
    pv  = arb(paper_str)
    rel = abs(val - pv) / abs(pv)
    ok  = bool(rel < REL_TOL)
    print(f"        ARB rel err vs {paper_str}: {rel}  -->  {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ctx.prec = ARB_PREC
    t0 = time.time()

    print()
    print("Theorem 9.4 and Corollary 9.5 -- Exact Density Constants -- ARB Certification")
    print("=" * 78)
    print(f"  M = {M} zeros; precision = {ARB_PREC}-bit ARB; quadrature rel_tol = 2^(-200).")
    print()

    print("  Loading spectral weights ...")
    b, a = load_weights()
    W1L  = sum(b)
    W1Z  = sum(a)
    sL2  = sum(bk**2 for bk in b) / arb(2)
    sZ2  = sum(ak**2 for ak in a) / arb(2)
    print(f"    W_1(L)     = {W1L}")
    print(f"    W_1(zeta)  = {W1Z}")
    print(f"    sigma_L    = {sL2**arb('1/2')}")
    print(f"    sigma_zeta = {sZ2**arb('1/2')}")
    print()

    # ── f_{S_L}(0) ───────────────────────────────────────────────────────────
    print(f"  [1/3] f_{{S_L}}(0): (1/pi) int_0^inf prod J_0(b_k t) dt ...")
    print(f"        Signed quadrature on [{T_EPS_STR}, {T_L}]; Bessel tail at T={T_L}.")
    t1 = time.time()
    f0_integral, f0_tail, _ = compute_f0(b)
    # Inflate the quadrature ball by the certified tail bound to form the
    # full rigorous enclosure: f_{S_L}(0) in [f0_integral - f0_tail, f0_integral + f0_tail].
    f0_certified = f0_integral + arb(0, f0_tail)
    print(f"        Quadrature ball (ARB): {f0_integral}")
    print(f"        Tail bound:            {f0_tail}")
    print(f"        Full enclosure:        {f0_certified}")
    ok_f0 = arb_matches(f0_certified, PAPER_F0, "f0")
    print(f"        [{time.time()-t1:.1f}s]")
    print()

    # ── E[|S_zeta|] ──────────────────────────────────────────────────────────
    print(f"  [2/3] E[|S_zeta|]: (2/pi) int_0^inf (1 - phi_z(t)) / t^2 dt ...")
    print(f"        Split: integral on [{T_EPS_STR}, {T_Z}] + (2/pi)/T_Z + phi-tail.")
    t1 = time.time()
    eabs_main, eabs_ptail, _ = compute_eabs(a)
    # Inflate by the certified phi-tail bound.
    eabs_certified = eabs_main + arb(0, eabs_ptail)
    print(f"        Main value (ARB):  {eabs_main}")
    print(f"        Phi-tail bound:    {eabs_ptail}")
    print(f"        Full enclosure:    {eabs_certified}")
    print(f"        Analytic correction (2/pi)/T_Z = {arb(2)/arb.pi()/arb(T_Z)}")
    ok_eabs = arb_matches(eabs_certified, PAPER_EABS, "eabs")
    print(f"        [{time.time()-t1:.1f}s]")
    print()

    # ── C(5) ─────────────────────────────────────────────────────────────────
    print("  [3/3] C(5) = 2 * f_{S_L}(0) * E[|S_zeta|] ...")
    # Propagate full certified enclosures into C(5).
    C5 = arb(2) * f0_certified * eabs_certified
    print(f"        C(5) (ARB, full enclosure): {C5}")
    ok_C5 = arb_matches(C5, PAPER_C5, "C5")
    print()

    # ── Gaussian comparison ───────────────────────────────────────────────────
    sL  = sL2**arb("1/2")
    sZ  = sZ2**arb("1/2")
    two = arb(2)
    C_gaussian = two * sZ / (arb.pi() * sL)
    print(f"  Gaussian reference: 2*sigma_zeta/(pi*sigma_L) = {C_gaussian}")
    print(f"  Overestimate factor: C_gaussian / C(5) = {C_gaussian / C5}")
    print()

    # ── Summary ───────────────────────────────────────────────────────────────
    all_pass = ok_f0 and ok_eabs and ok_C5
    print("  Summary:")
    print(f"    f_{{S_L}}(0):  {'PASS' if ok_f0 else 'FAIL'}   (paper: {PAPER_F0})")
    print(f"    E[|S_zeta|]: {'PASS' if ok_eabs else 'FAIL'}   (paper: {PAPER_EABS})")
    print(f"    C(5):        {'PASS' if ok_C5 else 'FAIL'}   (paper: {PAPER_C5})")
    print(f"    Total time: {time.time()-t0:.1f}s")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    All three constants certified via ARB quadrature (acb.integral,")
        print("    Petras algorithm) at 256-bit precision with certified tail bounds.")
    else:
        print("  RESULT: FAIL")
        raise RuntimeError("Theorem 9.4 and Corollary 9.5 ARB certification failed.")
    print()


if __name__ == "__main__":
    main()
