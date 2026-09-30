# IntegratorXX: two findings from wiring up a logarithmic radial map

Found while adding Köster's parameter-free radial mapping
(`r(x) = ln(2/(1-x))/ln 2`, as used by ERKALE) on top of
`GaussChebyshev2Modified`. Both are upstream issues rather than NuKEXC ones,
so they are written up here to be filed.

Versions: IntegratorXX at the revision pinned in
`cmake/nukexc-dep-versions.cmake` (`1369be58`), GCC 16, x86-64, double
precision.

---

## 1. `GaussChebyshev2Modified` cannot be used with `RadialTransformQuadrature`

`RadialTransformQuadrature` reads `base_quad_traits::bound_inclusive`:

```cpp
// quadratures/radial/radial_transform.hpp
const auto npts_base   = base_quad_traits::bound_inclusive ? npts+2 : npts;
const auto ipts_offset = !!base_quad_traits::bound_inclusive;
```

but only two of the seven primitives declare it:

| primitive | declares `bound_inclusive` |
|---|---|
| `gausschebyshev1` | no |
| `gausschebyshev2` | **yes** |
| `gausschebyshev2modified` | no |
| `gausschebyshev3` | no |
| `gausslegendre` | no |
| `gausslobatto` | no |
| `uniform` | **yes** |

So any radial transform can currently be composed only with
`GaussChebyshev2` or `Uniform`. Attempting
`RadialTransformQuadrature<GaussChebyshev2Modified<double,double>, …>`
fails to compile with

```
error: 'bound_inclusive' is not a member of 'base_quad_traits'
```

### Suggested fix

Add the member to each primitive's `quadrature_traits`. The correct value
is whether the rule places nodes at the interval endpoints:

| primitive | `bound_inclusive` | why |
|---|---|---|
| `gausschebyshev1` | `false` | open rule |
| `gausschebyshev2modified` | `false` | open; nodes are interior |
| `gausschebyshev3` | `false` | open |
| `gausslegendre` | `false` | open |
| `gausslobatto` | `true` | Lobatto includes both endpoints by construction |

`gausslobatto` is the one that matters beyond bookkeeping: it is the only
primitive here for which the value is `true`, and it is the natural base for
a Gauss–Lobatto × uniform-φ product sphere rule.

A defaulted member in a traits base, rather than a required one in each
specialization, would prevent this class of omission recurring.

---

## 2. Catastrophic cancellation in `1 - x` at the outer node

This one bites any *logarithmic* radial map, which is why it shows up here
and not with the Becke or Treutler–Ahlrichs transforms.

The modified Gauss–Chebyshev nodes are generated as (Pérez-Jordá,
San-Fabián and Moscardó, *J. Chem. Phys.* **100** (1994) 6520, Eq. 32)

```
x_i = 1 - 2i/(n+1) + (2/π)(1 + (2/3)sin²t)·cos t·sin t,    t = iπ/(n+1)
```

For the node nearest `x = +1` this is `1 - (something very close to 1)`, and
a consumer that then needs `1 - x` — as every log-type map does — gets it
with badly degraded precision. Measured relative error in `1 - x`, comparing
`1.0 - x_i` against the algebraically identical but cancellation-free

```
1 - x_i = 2i/(n+1) - (2/π)(1 + (2/3)sin²t)·cos t·sin t
```

| n | `x_max` | `1-x` | relative error |
|---|---|---|---|
| 25 | 0.999991315558714 | 8.68e-06 | 4.8e-12 |
| 50 | 0.999999699397504 | 3.01e-07 | 2.3e-11 |
| 75 | 0.999999959054435 | 4.09e-08 | 4.2e-10 |
| 150 | 0.999999998676713 | 1.32e-09 | **2.6e-08** |

At n = 150 the outermost radius and its Jacobian are wrong in the eighth
significant figure — for a mapping whose Jacobian is `1/(ln 2 (1-x))`, so
the weight inherits the full relative error.

### Suggested fix

Expose the complement alongside the node. Something like a
`generate_complement()` returning `1 - x_i` computed directly, or having the
primitive store both, would let a transform written as `f(1-x)` stay
accurate. The change is confined to the primitive; transforms that do not
need `1 - x` are unaffected.

A narrower alternative is to give `RadialTransformQuadrature` an optional
`radial_transform_complement(1-x)` hook that transforms opt into.

### It is measurable in molecular integrals, not just in the radial grid

Integrating a sum of normalized Slater densities on a water grid, the
diffuse case does not converge -- it hits a non-monotonic floor:

| nrad | lmax | error (ζ = 0.8) |
|---|---|---|
| 99 | 59 | 1.6e-09 |
| 99 | 71 | 8.7e-11 |
| 99 | 83 | 5.2e-09 |
| 150 | 95 | 3.0e-09 |

A diffuse density weights the outermost radial points most heavily, which
is exactly where the cancellation lives. Compact densities (ζ = 3) converge
smoothly to ~5e-11 over the same range.

### Note

ERKALE computes the same expression the same way, so this is not a
disagreement between the two codes — they lose the same digits in the same
place. It only becomes visible when comparing either against an analytically
exact result, or against an implementation that avoids the cancellation.

---

## What NuKEXC does meanwhile

`nukexc/koster_radial.hpp` assembles the quadrature directly from the
primitive rather than through `RadialTransformQuadrature`, because of (1).
`KosterRadialTraits` is nevertheless written against IntegratorXX's
`RadialTraits` interface, so once (1) is fixed the generator collapses to a
type alias and the class can be contributed upstream unchanged.

Nothing is done about (2): matching ERKALE bit-for-bit is currently worth
more than being more accurate than it, and the two must not be fixed
independently or the parity tests become meaningless.
