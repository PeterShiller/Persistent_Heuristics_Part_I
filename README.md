# Unconditional Density Bounds for Quadratic Norm-Form Energies via Lorentzian Spectral Weights

### Persistent Heuristics I: Mini Monograph I

Part of the [Persistent Heuristics](https://github.com/PeterShiller/Persistent_Heuristics) series.

**Author:** Peter Shiller (Independent Researcher)

**arXiv:** [https://arxiv.org/abs/2603.00301](https://arxiv.org/abs/2603.00301)

**Zenodo:** [https://doi.org/10.5281/zenodo.18783098](https://doi.org/10.5281/zenodo.18783098)

arXiv contains the main paper only. Zenodo has an exact copy of the paper, scripts, and data in this repository.

## Abstract

For a real quadratic field Q(sqrt(d)), we study the norm-form energy N = S_zeta^2 - d S_L^2, where S_zeta and S_L are Lorentzian-weighted zero sums with w(rho) = 2/(1/4 + gamma^2). Three main results:

1. **Spacelike spectral data.** N < 0 unconditionally for all squarefree d > 1, via a low-lying zero dominance theorem proved by explicit zero-counting.

2. **Effective density bound.** At each verified truncation level M, dens{N > 0} <= 2 ||f_{S_L^(M)}||_inf (W_1(zeta)/sqrt(d) + epsilon_M), established unconditionally via Jacobi--Anger resonance analysis. The O(1/sqrt(d)) rate requires M to grow with d under a computationally verified finite-rank condition on the resonance lattice.

3. **Exact asymptotic.** Under the hypothesis that the infinite resonance lattice has finite rank (verified to have rank 0 for M <= 20), the sharp asymptotic dens{N > 0} = C(d)/sqrt(d) + o(1/sqrt(d)) holds. For d = 5, C(5) = 0.1193.

Appendix F tabulates between 1004 and 1044 zeros at 70 decimal places for eight quadratic Dirichlet L-functions (8244 zeros total), all rigorously certified by ARB interval arithmetic.

## Repository Structure

```
00.Computing L(s, χ) Zeros/       ARB zero computation pipeline (8-phase)
01.Computed L(s, χ) Zeros.../     Sealed zero tables: 8 characters + Riemann zeta
02.Theorem Scripts/                17 self-contained ARB certification scripts
03.LaTeX/                          Paper and Appendix F source (.tex)
04.LaTeX Figures/                  Figure PDFs (included by LaTeX)
05.PDF/                            Compiled paper (63 pp) and Appendix F (186 pp)
06.Library/                        pip-installable Python data API
```

### `00.Computing L(s, χ) Zeros/`

The eight-phase computation pipeline: sign-change scan, bisection filtering, Newton refinement at 1500-bit, global completeness verification via argument principle, gap recovery, and bound boosting to 2200-bit. Requires `python-flint >= 0.8.0`.

### `01.Computed L(s, χ) Zeros and Imported ζ Zeros/`

Sealed zero data for all eight quadratic characters (chi_2 through chi_13) at 70 decimal places, with certified |L(1/2 + i gamma)| bounds. Also contains 6000 Riemann zeta zeros from the LMFDB at 31 decimal places.

| Character | Conductor | Zeros | Worst bound |
|-----------|-----------|-------|-------------|
| chi_2     | 8         | 1016  | < 2e-409    |
| chi_3     | 12        | 1043  | < 5e-419    |
| chi_5     | 5         | 1004  | < 3e-428    |
| chi_6     | 24        | 1039  | < 7e-430    |
| chi_7     | 28        | 1044  | < 2e-421    |
| chi_10    | 40        | 1040  | < 9e-421    |
| chi_11    | 44        | 1038  | < 7e-425    |
| chi_13    | 13        | 1020  | < 2e-447    |

### `02.Theorem Scripts/`

Seventeen self-contained Python scripts, one per theorem, proposition, or table in the paper. Each script certifies its result using ARB interval arithmetic and prints PASS/FAIL. Dependencies: `python-flint >= 0.8.0`, `sympy >= 1.12` (two scripts only). No external data files needed beyond the zero tables in `01.*/`.

### `03.LaTeX/`

LaTeX source files for the main paper (63 pages) and Appendix F (186 pages).

### `04.LaTeX Figures/`

Figure PDFs included by the LaTeX source: the norm-form plot and the density convergence plot.

### `05.PDF/`

Compiled PDFs of the main paper and Appendix F.

### `06.Library/`

A pip-installable data API for the zero tables. See [`06.Library/README.md`](06.Library/README.md) for installation and usage.

```python
from persistent_heuristics_I import get_zeros, get_bound, get_seal
zeros = get_zeros(5, n=20, as_strings=True)   # first 20 zeros of L(s, chi_5)
bound = get_bound(5, 1)                        # certified |L(1/2+i*gamma_1)| bound
```

## Requirements

**Data access only** (zero tables, library API): Python >= 3.10, no compiled dependencies.

**Running theorem scripts or the compute pipeline:** `python-flint >= 0.8.0` (ARB interval arithmetic). Two scripts additionally require `sympy >= 1.12`.

## Citation

```bibtex
@misc{shiller2026unconditional,
  author = {Shiller, Peter},
  title  = {Unconditional Density Bounds for Quadratic Norm-Form Energies
            via Lorentzian Spectral Weights},
  year   = {2026},
  eprint = {2603.00301},
  archiveprefix = {arXiv},
  primaryclass  = {math.NT},
  doi    = {10.5281/zenodo.18783098},
  url    = {https://doi.org/10.5281/zenodo.18783098}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
