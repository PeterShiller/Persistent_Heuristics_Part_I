"""
Kronecker_character_data.py  —  Kronecker character data for quadratic Dirichlet L-functions
============================================================================================
Ancillary data module for:
    Shiller, P. (2026). Unconditional Density Bounds for Quadratic
    Norm-Form Energies via Lorentzian Spectral Weights. Zenodo.
    https://doi.org/10.5281/zenodo.18783098

This module tabulates the Kronecker character chi_d for each squarefree
discriminant d covered by the above reference.  The character values feed the Hurwitz
zeta decomposition

    L(s, chi_d) = q^{-s} * sum_{a=1}^{q} chi_d(a) * zeta(s, a/q),

where q = |Delta_K| is the conductor of Q(sqrt(d)).  Terms with
gcd(a, q) > 1 vanish; only nonzero values chi_d(a) = +/-1 are stored.
All characters are real, primitive, and even (chi_d(-1) = +1).

Verification:
    chi_d(a) equals the Kronecker symbol (Delta_K | a), where
    Delta_K = d if d = 1 (mod 4), Delta_K = 4d otherwise.
    Running this module as a script checks every entry against
    an independent Kronecker symbol computation and verifies
    that the nonzero values exhaust the residues coprime to q.

    Cross-checked against LMFDB (www.lmfdb.org) on 27 February 2026
       """

CHARACTERS = {
    5: {
        "q": 5,
        "field": "Q(sqrt(5))",
        "chi": {1: +1, 4: +1, 2: -1, 3: -1},
    },
    2: {
        "q": 8,
        "field": "Q(sqrt(2))",
        "chi": {1: +1, 7: +1, 3: -1, 5: -1},
    },
    3: {
        "q": 12,
        "field": "Q(sqrt(3))",
        "chi": {1: +1, 11: +1, 5: -1, 7: -1},
    },
    13: {
        "q": 13,
        "field": "Q(sqrt(13))",
        "chi": {1: +1, 3: +1, 4: +1, 9: +1, 10: +1, 12: +1,
                2: -1, 5: -1, 6: -1, 7: -1, 8: -1, 11: -1},
    },
    6: {
        "q": 24,
        "field": "Q(sqrt(6))",
        "chi": {1: +1, 5: +1, 19: +1, 23: +1,
                7: -1, 11: -1, 13: -1, 17: -1},
    },
    7: {
        "q": 28,
        "field": "Q(sqrt(7))",
        "chi": {1: +1, 3: +1, 9: +1, 19: +1, 25: +1, 27: +1,
                5: -1, 11: -1, 13: -1, 15: -1, 17: -1, 23: -1},
    },
    10: {
        "q": 40,
        "field": "Q(sqrt(10))",
        "chi": {1: +1, 3: +1, 9: +1, 13: +1, 27: +1, 31: +1, 37: +1, 39: +1,
                7: -1, 11: -1, 17: -1, 19: -1, 21: -1, 23: -1, 29: -1, 33: -1},
    },
    11: {
        "q": 44,
        "field": "Q(sqrt(11))",
        "chi": {1: +1, 5: +1, 7: +1, 9: +1, 19: +1, 25: +1, 35: +1, 37: +1,
                39: +1, 43: +1,
                3: -1, 13: -1, 15: -1, 17: -1, 21: -1, 23: -1, 27: -1, 29: -1,
                31: -1, 41: -1},
    },
}


# ------------------------------------------------------------------
# Access functions
# ------------------------------------------------------------------

def get_character(d):
    """Return (conductor q, chi_dict) for squarefree d."""
    if d not in CHARACTERS:
        raise ValueError(
            f"d={d} not available. Have: {sorted(CHARACTERS.keys())}")
    entry = CHARACTERS[d]
    return entry["q"], entry["chi"]


def hurwitz_terms(d):
    """Return sorted list of (a/q, chi_d(a)) pairs for the Hurwitz sum."""
    q, chi = get_character(d)
    return [(a / q, chi[a]) for a in sorted(chi.keys())]


# ------------------------------------------------------------------
# Self-verification
# ------------------------------------------------------------------

def _kronecker(D, n):
    """Kronecker symbol (D|n) for fundamental discriminant D, positive n."""
    if n == 1:
        return 1
    result = 1
    while n % 2 == 0:
        n //= 2
        if D % 2 == 0:
            return 0
        if D % 8 in (3, 5):
            result = -result
    if n == 1:
        return result
    a, b = D % n, n
    if a < 0:
        a += b
    while a != 0:
        while a % 2 == 0:
            a //= 2
            if b % 8 in (3, 5):
                result = -result
        a, b = b, a
        if a % 4 == 3 and b % 4 == 3:
            result = -result
        a = a % b
    return result if b == 1 else 0


def _verify():
    """Check every stored value against the Kronecker symbol."""
    from math import gcd
    ok = True
    for d in sorted(CHARACTERS.keys()):
        q, chi = get_character(d)
        Delta = d if d % 4 == 1 else 4 * d
        coprime = [a for a in range(1, q) if gcd(a, q) == 1]
        if set(chi.keys()) != set(coprime):
            print(f"  d={d}: residue set mismatch")
            ok = False
            continue
        bad = [a for a in coprime if chi[a] != _kronecker(Delta, a)]
        if bad:
            print(f"  d={d}: wrong values at a = {bad}")
            ok = False
        else:
            n_plus = sum(1 for v in chi.values() if v == +1)
            n_minus = sum(1 for v in chi.values() if v == -1)
            print(f"  d={d:>2}  q={q:>2}  phi(q)={len(coprime):>2}  "
                  f"(+1: {n_plus}, -1: {n_minus})  {CHARACTERS[d]['field']}  OK")
    return ok


if __name__ == "__main__":
    print("Verifying character data against Kronecker symbol computation:\n")
    if _verify():
        print("\nAll entries verified.")
    else:
        print("\nERRORS DETECTED.")
        raise SystemExit(1)
