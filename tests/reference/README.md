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
| `water_631gdp_xc_erkale.ref` | 6-31G(d,p) basis + spin densities + unrestricted XC energies and matrices, 7 functionals, 2 grids | `tests/test_erkale_xc.cxx` |
| `radial_chebyshev_{25,50,75}.ref` | ERKALE radial grids, weights excluding r² | `tests/test_koster_radial.cxx` |
| `becke_erkale_cellfn.ref` | ERKALE's f_s(mu, a=0.7), 2001 samples | `tests/test_partition_schemes.cxx` |
| `lobatto_l{5,11,17,29}.ref` | ERKALE Lobatto shells, unit sphere | `tests/test_lobatto_angular.cxx` |
| `integratorxx-upstream.md` | two IntegratorXX issues found here, to be filed | — |

`water.xyz` is a copy of `input/water.xyz` with the leading blank line removed:
NuKEXC's `read_xyz` tolerates it, ERKALE's `load_xyz` asserts.

## Provenance

- **ERKALE** commit `ea638a54` ("emd: normalize accessors, underscore members")
  for the overlap, radial, Becke and Lobatto files, built in
  `/home/work/erkale/build-verify` (OpenMP, `liberkale_omp.a`).
- **ERKALE** commit `15d0f506` ("basis: correct the integrals of Cartesian
  functions, and add moments") for `water_631gdp_xc_erkale.ref`, same build
  directory, rebuilt. The older files were not regenerated: they are static
  text and remain exactly what `ea638a54` produced.
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

`erkale_dump_xc.cpp` is the third, for the XC reference, and links the same
way with one addition: ERKALE now vendors libxckernel itself, so
`src/xckernel/libxckernel.a` has to be on the link line or
`xckernel_dispatch.cpp` leaves `xckernel_kernels` undefined.

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

## The XC reference, and what it can and cannot settle

`water_631gdp_xc_erkale.ref` holds, for water / 6-31G(d,p):

- the basis, as every file here does;
- `Pa` and `Pb`, a fixed pair of spin density matrices -- rank one plus a
  positive diagonal shift, so positive semidefinite and the density is
  nonnegative everywhere. No SCF is involved: the quantity under test is the
  functional derivative at a given density, and a fixed arithmetic recipe is
  reproducible in a way converged orbitals are not. They are written into the
  file rather than recomputed on each side, for the same reason the basis is;
- for each of `LDA_X`, `GGA_X_PBE`, `GGA_XC_PBE`, `MGGA_X_TPSS`,
  `MGGA_XC_TPSS`, `MGGA_X_SCAN` and `MGGA_X_BR89_EXPLICIT`, on a coarse
  (75, lmax 29; 56910 points) and a fine (150, lmax 41; 220304 points) grid:
  `Exc`, `Nel`, the point count, and both spin channels of the XC matrix.

Two grids, because the comparison cannot be better than the reference's own
grid error and that error varies hugely between functionals -- 1.1e-7
relative on `LDA_X`'s energy against 2.6e-4 on `BR89`'s. The test scales its
tolerance with the measured coarse-to-fine change instead of carrying a
number someone chose.

**The caveat that matters.** At commit `15d0f506` ERKALE assembles its XC
matrices through libxckernel, the same generator NuKEXC uses -- its
`AngularGrid::eval_Fxc` calls `xck_contract("ua_o1")` and `("ub_o1")`. The
two vendor different libxckernel commits (ERKALE `2ae20f4b`, NuKEXC `5d84d7a`
= v0.5.0), but that is not independence: an error in the generator's
meta-GGA derivation would sit on both sides and cancel. So this file does
**not** settle whether the generated meta-GGA chain rule is right.

What it does settle, because the two codes share no code for any of it: the
basis evaluation and ordering, the grid, the construction of rho, sigma, tau
and the Laplacian from a density matrix, the Libxc conventions for a
polarized functional, the energy expression, and the operand wiring -- ERKALE
reaches the kernels through a string-keyed dispatch table and NuKEXC through
libxckernel's named structs, so a swapped same-spin and cross-spin sigma
derivative, or a `vlapl` in the `vrho` slot, cannot survive the comparison.

Closing the remaining gap needs an ERKALE from before the libxckernel switch,
whose `increment_lda` / `increment_gga` / `increment_mgga_kin` assembly was
hand-written and therefore genuinely independent.

### Current agreement

Relative deviations from ERKALE's fine grid, with ERKALE's own
coarse-to-fine change for comparison:

| functional | Exc | Ha | Hb | ERKALE Exc / Ha |
|---|---|---|---|---|
| `LDA_X` | 1.7e-8 | 6.1e-8 | 6.9e-8 | 1.1e-7 / 6.0e-7 |
| `GGA_X_PBE` | 5.0e-8 | 2.0e-6 | 1.4e-6 | 6.4e-7 / 6.5e-6 |
| `GGA_XC_PBE` | 1.1e-8 | 2.0e-6 | 1.4e-6 | 7.2e-7 / 5.5e-6 |
| `MGGA_X_TPSS` | 8.9e-8 | 1.9e-6 | 3.3e-6 | 1.1e-6 / 5.0e-6 |
| `MGGA_XC_TPSS` | 5.5e-8 | 2.7e-6 | 3.5e-6 | 1.2e-6 / 6.5e-6 |
| `MGGA_X_SCAN` | 4.1e-6 | 2.7e-4 | 7.1e-5 | 9.0e-6 / 7.0e-4 |
| `MGGA_X_BR89` | 2.4e-5 | 4.1e-4 | 6.3e-4 | 2.6e-4 / 9.6e-4 |

Every entry is below the reference's own grid uncertainty, on a NuKEXC grid
of 128810 points (99 radial, lmax 35, Treutler-Ahlrichs / Lebedev). The
electron count, which no functional enters, agrees to 1.4e-9 absolute.

`MGGA_X_BR89_EXPLICIT` is in the list because it is the only functional here
that needs the density Laplacian, i.e. the only one that exercises the
`MetaGGALapl` family and the Hessian collocation tower at all. Its poor grid
convergence is a property of the functional, not of either code.
