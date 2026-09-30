# Reference data

Numbers in here come from ERKALE, not from NuKEXC. They exist so the tests can
check *parity* rather than self-consistency: the rest of the GTO suite
validates NuKEXC against its own closed forms, which catches implementation
slips but not a shared misunderstanding of the conventions. If the evaluator,
its reference and the normalization all agreed on the wrong phase or component
ordering, every one of those tests would still pass.

Each reference file carries **both the basis and the quantity computed from
it**. That is deliberate: reading the basis out of the same file as the matrix
removes any question of whether the two codes were handed the same basis, which
is otherwise the easiest way to make such a comparison meaningless.

## Files

| file | what | consumed by |
|---|---|---|
| `water.xyz` | water geometry, Ångström | both sides |
| `water_ccpvdz_erkale.ref` | cc-pVDZ basis + analytic overlap | `tests/test_erkale_overlap.cxx` |
| `water_631gdp_erkale.ref` | 6-31G(d,p) basis + analytic overlap | `tests/test_erkale_overlap.cxx` |
| `radial_chebyshev_{25,50,75}.ref` | ERKALE radial grids, weights excluding r² | `tests/test_koster_radial.cxx` |
| `becke_erkale_cellfn.ref` | ERKALE's f_s(mu, a=0.7), 2001 samples | `tests/test_partition_schemes.cxx` |
| `integratorxx-upstream.md` | two IntegratorXX issues found here, to be filed | — |

`water.xyz` is a copy of `input/water.xyz` with the leading blank line removed:
NuKEXC's `read_xyz` tolerates it, ERKALE's `load_xyz` asserts.

## Provenance

- **ERKALE** commit `ea638a54` ("emd: normalize accessors, underscore members"),
  built in `/home/work/erkale/build-verify` (OpenMP, `liberkale_omp.a`).
- **Bases** `/home/work/erkale/basis/cc-pVDZ.gbs` and
  `/home/work/erkale/basis/6-31G(d,p).gbs`, loaded with
  `BasisSetLibrary::load_gaussian94`. ERKALE's library has no bare `6-31G*`;
  `6-31G(d,p)` is a superset of it.
- **Defaults** as constructed by `settings.add_scf_settings()`; in particular
  ERKALE's OptLM default, under which s and p shells are stored Cartesian and
  d and higher spherical. The `shell` records in the file record the
  per-shell choice explicitly, so the test does not have to assume it.
- Geometry converted to bohr by ERKALE (`load_xyz(..., true)`); the centers in
  the file are therefore already in bohr.

## Regenerating

`erkale_dump_basis_overlap.cpp` is **not** part of the NuKEXC build — it links
against ERKALE and is run by hand.

The radial reference comes from a second, smaller dumper built the same way
against `chebyshev.h`, calling `radial_chebyshev(n, r, w)` for
n = 25, 50, 75.

```sh
E=/home/work/erkale
B=$E/build-verify/src

g++ -std=c++17 -O2 -fopenmp -o erk_dump \
    tests/reference/erkale_dump_basis_overlap.cpp \
    -I$E/src -I$B \
    -DARMA_DONT_USE_WRAPPER -DARMA_NO_DEBUG -DARMA_USE_BLAS \
    -DARMA_USE_LAPACK -DARMA_BLAS_LONG_LONG \
    $B/completeness/liberkale_cmp_omp.a $B/emd/liberkale_emd_omp.a \
    $B/liberkale_omp.a \
    /usr/lib64/libcint.so /usr/lib64/libflexiblas64.so /usr/lib64/libgmp.so \
    /usr/lib64/libgsl.so /usr/lib64/libgslcblas.so /usr/lib64/libhdf5.so \
    /usr/lib64/libmpfr.so /usr/lib64/libwignernj.so.0.8.0 \
    /usr/lib64/libxc.so.15 -lquadmath -ldl -lm

./erk_dump tests/reference/water.xyz $E/basis/cc-pVDZ.gbs \
           tests/reference/water_ccpvdz_erkale.ref
./erk_dump tests/reference/water.xyz "$E/basis/6-31G(d,p).gbs" \
           tests/reference/water_631gdp_erkale.ref
```

Two things that are easy to trip over when relinking: ERKALE's library code
reads a global `Settings settings` that its *drivers* define, so the dumper
defines one; and `liberkale_omp.a` alone is not enough — the completeness and
emd archives plus libcint are needed.

## File format

Plain text, one record per line.

```
nshell <n>
nbf    <n>
shell  <l> <spherical> <ox> <oy> <oz> <nprim> <nctr>
  exp  <alpha>                        x nprim
  coef <c>                            x nprim*nctr, contraction-major
...
overlap <nbf*nbf>
<value>                               x nbf*nbf, row-major
```

Coefficients are as ERKALE holds them internally: coefficients of
**unnormalized** primitives after `GaussianShell::normalize()`. That is exactly
what `NormalizationConvention::ERKALE` produces, so the test asks for that
convention rather than `None`, which additionally checks that renormalizing an
already-normalized shell is idempotent.

## Current agreement

Water / cc-pVDZ, 11 shells, 24 functions, including a generally contracted
oxygen s shell (9 primitives, 2 contractions):

- overlap **diagonal** from `GTOShell::function_norms()` matches ERKALE to
  `1e-12` — this pins the contraction normalization and the spherical
  `1/sqrt(S(0))` rescaling without involving any two-centre machinery;
- full 24×24 **quadrature** overlap on a (120, L=41) Treutler–Ahlrichs /
  Lebedev grid matches to **2.6e-8**, which is grid error. A wrong phase,
  ordering or normalization would show up at O(1), not here.

Water / 6-31G(d,p), 12 shells, 24 functions:

- the **density integrates to the electron count** to **1.0e-8**. No SCF is
  needed to make that a real test: for any S-orthonormal set of vectors,
  ∫ρ = Σᵢ occᵢ cᵢᵀ S cᵢ = Σᵢ occᵢ exactly, so orthonormalizing arbitrary
  vectors against ERKALE's overlap gives an exactly known target and the
  deviation is purely NuKEXC's quadrature.
