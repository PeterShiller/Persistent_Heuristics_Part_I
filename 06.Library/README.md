# persistent-heuristics-I

ARB-certified zero tables and computation pipeline for quadratic Dirichlet
L-functions, ancillary to:

> Shiller, P. (2026). *Unconditional Density Bounds for Quadratic Norm-Form
> Energies via Lorentzian Spectral Weights.* arXiv:2603.00301.
> Zenodo: https://doi.org/10.5281/zenodo.18783098

## Installation

This package must be installed in editable mode from a cloned copy of the
repository. The zero data files are not bundled; they are loaded at runtime
from their canonical locations in the repository tree.

**Data API only** (no binary dependencies):

```bash
git clone https://github.com/PeterShiller/Persistent_Heuristics_Part_I
pip install -e Persistent_Heuristics_Part_I/06.Library
```

**With computation pipeline** (requires python-flint):

```bash
pip install -e 'Persistent_Heuristics_Part_I/06.Library[compute]'
```

## Usage

```python
from persistent_heuristics_I import (
    get_zero, get_zeros, get_bound, get_seal, available_characters
)

# Available characters
available_characters()          # [2, 3, 5, 6, 7, 10, 11, 13]

# First 20 zeros of L(s, chi_5) as Decimal objects at 70 decimal places
zeros = get_zeros(5, n=20)

# As strings
zeros = get_zeros(5, n=20, as_strings=True)

# Single zero
zero = get_zero(5, 1)

# Certified |L(1/2 + i*gamma)| bound for zero k, returned as (mantissa, exponent)
# meaning |L(1/2 + i*gamma)| < mantissa * 10^exponent
bound = get_bound(5, 1)         # e.g. (1.56, -450)

# Seal metadata (height, zero count, working precision, date)
seal = get_seal(5)

# Riemann zeta zeros (6000 zeros, LMFDB, 31 decimal places)
from persistent_heuristics_I import zeta
zeta.get_zeros(n=60)
zeta.get_zero(1)
```

**CLI** (requires `[compute]`):

```bash
ph1-compute-zeros --d 5 --nzeros 1000 --high-precision
```

## Data

| d  | conductor q | zeros | sealed | T_seal |
|----|-------------|-------|--------|--------|
| 2  | 8           | 1016  | yes    | 1033   |
| 3  | 12          | 1043  | yes    | 1000   |
| 5  | 5           | 1004  | yes    | 1094   |
| 6  | 24          | 1039  | yes    | 912    |
| 7  | 28          | 1000  | —      | —      |
| 10 | 40          | 1000  | —      | —      |
| 11 | 44          | 1000  | —      | —      |
| 13 | 13          | 1000  | —      | —      |

All sealed zeros are certified by ARB interval arithmetic at 1500-bit working
precision. Certified bounds |L(1/2 + i*gamma)| are returned as (mantissa,
exponent) tuples by get_bound(). The table-wide floor is 10^{-409} (chi_2);
most zeros have bounds in the range 10^{-420} to 10^{-660}. Completeness is
guaranteed by a global argument-principle winding number on the rectangle
[-0.5, 1.5] x [0.5, T_seal].

## License

CC BY 4.0. Cite the Zenodo deposit when using this data.
