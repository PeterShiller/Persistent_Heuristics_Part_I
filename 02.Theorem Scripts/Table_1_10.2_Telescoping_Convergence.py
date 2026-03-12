"""
Table_1_10.2_Telescoping_Convergence.py
========================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights.
    arXiv:2603.00301.  Zenodo: https://doi.org/10.5281/zenodo.18783098

This script certifies the numerical table in Section 10.2
(Telescoping convergence), which displays the convergence of the
density bound as the truncation level M increases:

    M  | f_main^(M)(0) | Predicted tail | Total bound
    ---|---------------|----------------|------------
     5 |        10.877 |          1.365 |      12.242
    10 |        10.585 |          0.951 |      11.536
    15 |        10.535 |          0.763 |      11.298
    20 |        10.517 |          0.649 |      11.166
    25 |        10.508 |          0.571 |      11.079
    30 |        10.504 |          0.513 |      11.017

Column definitions:
    f_main^(M)(0) = (1/pi) int_0^T prod_{k=1}^M |J_0(b_k t)| dt  (unsigned)
    epsilon_M     = sum_{k=M+1}^{200} b_k  +  analytic BMOR tail
    Predicted tail = 2 * f_main^(M)(0) * epsilon_M
    Total bound    = f_main^(M)(0) + Predicted tail

RIGOUR ARCHITECTURE
-------------------
All arithmetic is ARB interval arithmetic throughout.  No Python floats
are used in any load-bearing computation.

f_main^(M)(0): computed via integrate_unsigned imported from
    Table_1_after_Lemma_7.1_and_Table_2_Remark_7.2.py, which applies
    the strip decomposition / acb.integral architecture certified by
    that script.  The integration upper limit T is chosen per M so that
    the analytic Bessel tail bound falls within the PASS tolerance:
    T = 10000 for M = 5 (tail ~ 0.003 at T = 2000 would fail at 1e-3),
    T = 2000 for M >= 10.  The certified enclosure is
        f_main^(M)_certified = integrate_unsigned(b, M, T) + arb(0, tail_bound(M, T)).

epsilon_M: ARB sum of b_k for k = M+1, ..., 200 (70-digit certified zeros)
    plus the analytic tail sum_{k>200} b_k bounded via Abel summation against
    BMOR (Bennett-Martin-O'Bryant-Rechnitzer, Math. Comp. 90, 2021, Thm 1.1):
    tail = (1/pi)(log(5*T/2*pi)/T + 1/T) evaluated at T = gamma'_200 = 283.935.

PASS criterion: certified ARB predicate
    bool(abs(computed - paper) / abs(paper) < REL_TOL) == True
for each column entry, with REL_TOL = 1e-3.

Zero data:
    First 200 zeros of L(s, chi_5): from L_function_zeros.py,
        Zenodo doi:10.5281/zenodo.18783098, 70 decimal places.

Working precision: 256-bit ARB throughout.
"""

import os
import sys
import time
import importlib.util

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "..", "01.Computed L(s, \u03c7) Zeros and Imported \u03b6 Zeros")
sys.path.insert(0, _DATA)

from L_function_zeros import get_zeros as get_L_zeros
from flint import arb, ctx

# Import the certified unsigned integrator and tail bound from the existing
# script (importlib.util required because the filename contains parentheses).
_t1_path = os.path.join(
    _HERE, "Table_1_after_Lemma_7.1_and_Table_2_Remark_7.2.py"
)
_spec  = importlib.util.spec_from_file_location("_t1mod", _t1_path)
_t1mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_t1mod)
integrate_unsigned = _t1mod.integrate_unsigned
arb_tail_bound     = _t1mod.arb_tail_bound

# ── Parameters ────────────────────────────────────────────────────────────────
ARB_PREC = 256
REL_TOL  = arb("1e-3")

M_VALUES = [5, 10, 15, 20, 25, 30]

# Integration upper limit per M.
# At T=2000 the Bessel tail for M=5 has radius ~0.030, which exceeds the
# 1e-3 relative tolerance against f_main ~ 10.877.  T=10000 reduces the
# tail to ~0.0027, comfortably within tolerance.  For M >= 10 the tail at
# T=2000 is already negligible.
T_PER_M = {5: 10000, 10: 2000, 15: 2000, 20: 2000, 25: 2000, 30: 2000}

# Paper values: (f_main, predicted_tail, total_bound)
PAPER = {
     5: ("10.877", "1.365", "12.242"),
    10: ("10.585", "0.951", "11.536"),
    15: ("10.535", "0.763", "11.298"),
    20: ("10.517", "0.649", "11.166"),
    25: ("10.508", "0.571", "11.079"),
    30: ("10.504", "0.513", "11.017"),
}


# ── Load weights ─────────────────────────────────────────────────────────────

def load_weights_200():
    ctx.prec = ARB_PREC
    zeros = get_L_zeros(5, 200, as_strings=True)
    quarter = arb("1/4")
    b = [arb(2) / (quarter + arb(g)**2) for g in zeros]
    return b, zeros


# ── epsilon_M ────────────────────────────────────────────────────────────────

def compute_epsilon(b200, zeros_str, M_int):
    """
    epsilon_M = sum_{k=M+1}^{200} b_k  +  BMOR analytic tail.

    BMOR tail at T = gamma'_200 = 283.935 (70-digit certified):
        (1/pi)(log(q*T/2*pi)/T + 1/T)  with q = 5.
    """
    ctx.prec = ARB_PREC
    partial   = sum(b200[M_int:200])
    T200      = arb(zeros_str[199])
    pi_arb    = arb.pi()
    bmor_tail = (arb.log(arb(5) * T200 / (arb(2) * pi_arb)) / T200
                 + arb(1) / T200) / pi_arb
    return partial + bmor_tail


# ── PASS/FAIL ─────────────────────────────────────────────────────────────────

def arb_matches(val, paper_str):
    pv  = arb(paper_str)
    rel = abs(val - pv) / abs(pv)
    return bool(rel < REL_TOL), rel


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ctx.prec = ARB_PREC
    t0 = time.time()

    print()
    print("Section 10.2 -- Telescoping Convergence Table -- ARB Certification")
    print("=" * 68)
    print(f"  Precision: {ARB_PREC}-bit ARB.  REL_TOL = 1e-3.")
    print(f"  T per M: {T_PER_M}")
    print()

    print("  Loading 200 chi_5 weights ...")
    b200, zeros_str = load_weights_200()
    print()

    results  = []
    all_pass = True

    for M in M_VALUES:
        t1 = time.time()
        ctx.prec = ARB_PREC
        T = T_PER_M[M]

        # f_main^(M)(0): unsigned integral + certified tail inflation
        f_int  = integrate_unsigned(b200, M, T_upper=T)
        f_tail = arb_tail_bound(b200, M, T)
        f_cert = f_int + arb(0, f_tail)

        # epsilon_M
        eps = compute_epsilon(b200, zeros_str, M)

        # Predicted tail and total bound propagated through certified enclosures
        pred  = arb(2) * f_cert * eps
        total = f_cert + pred

        pf_str, ppt_str, ptot_str = PAPER[M]
        ok_f,   rel_f   = arb_matches(f_cert, pf_str)
        ok_pt,  rel_pt  = arb_matches(pred,   ppt_str)
        ok_tot, rel_tot = arb_matches(total,  ptot_str)
        ok       = ok_f and ok_pt and ok_tot
        all_pass = all_pass and ok

        elapsed = time.time() - t1
        print(f"  M={M:2d}  T={T}  [{elapsed:.1f}s]  {'PASS' if ok else 'FAIL'}")
        print(f"         f_main^({M})(0)  = {f_cert}")
        print(f"         epsilon_{M:<2}      = {eps}")
        print(f"         Pred tail       = {pred}")
        print(f"         Total bound     = {total}")
        if not ok_f:   print(f"         FAIL f_main:  rel={rel_f}")
        if not ok_pt:  print(f"         FAIL pred:    rel={rel_pt}")
        if not ok_tot: print(f"         FAIL total:   rel={rel_tot}")
        print()

        results.append((M, f_cert, eps, pred, total, ok))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"  {'M':>3}  {'f_main^(M)(0)':>14}  {'eps_M':>9}  "
          f"{'Pred tail':>10}  {'Total':>7}  status")
    print(f"  {'-'*3}  {'-'*14}  {'-'*9}  {'-'*10}  {'-'*7}  ------")
    for M, f_cert, eps, pred, total, ok in results:
        print(f"  {M:>3}  {float(f_cert.mid()):>14.3f}  {float(eps.mid()):>9.5f}  "
              f"{float(pred.mid()):>10.3f}  {float(total.mid()):>7.3f}  "
              f"{'PASS' if ok else 'FAIL'}")

    print(f"\n  Total time: {time.time()-t0:.1f}s")
    print()

    if all_pass:
        print("  RESULT: ALL PASS  [ARB-rigorous]")
        print("    All columns certified: f_main via acb.integral strip decomposition")
        print("    (imported from Table_1_after_Lemma_7.1 script), epsilon_M via ARB")
        print("    partial sum + BMOR analytic tail.  Predicted tail and Total bound")
        print("    use full certified enclosures throughout.")
    else:
        print("  RESULT: FAIL")
        raise RuntimeError(
            "Section 10.2 telescoping convergence ARB certification failed.")
    print()


if __name__ == "__main__":
    main()
