# libxckernel: the header-only ABI cannot see its own operand order

> **Resolved in v0.5.0** (`5d84d7a`, 2026-10-06). Findings 1 and 2 are
> both fixed; finding 3's recommendation was followed, in that the
> ordering was left alone. Kept here for the rationale and for the
> measurements, which are still the evidence that NuKEXC's XC path and
> the generated kernels agree. See "Status after v0.5.0" at the end for
> what changed and what it means for NuKEXC.

Found while wiring NuKEXC's XC path onto the generated kernels (Phase 4,
restricted `o1` Fock matrices for `lda`, `gga`, `mgga_tau`, `mgga`). One
real defect, one hardening suggestion, and one tempting fix that would be
a mistake.

Versions: libxckernel at `71c7226` (branch `changelog`, 2026-10-06),
generated with

    python3 -m xckernel.catalog <out> lda,gga,mgga_tau,mgga 1 c

GCC 16, x86-64, double precision.

**First, the part that is right.** The generated kernels reproduce
NuKEXC's own hand-written LDA and GGA Fock assembly -- which is
independently validated against ERKALE -- to machine precision:

| functional | NuKEXC screening on | screening off |
|---|---|---|
| `XC_LDA_X` | 2.5e-10 | **1.2e-14** |
| `XC_GGA_X_PBE` | 3.4e-10 | **8.5e-15** |

31 basis functions, 13180 grid points, water/TZP. The first column is not
a kernel error: it is NuKEXC's own density and basis screening at 1e-10
(`XCThresholds`, matching ERKALE's `DFTDensityThr`/`DFTBasisThr`). With
screening disabled the two agree to rounding. So the ABI, the operand
ordering and the tower layout are all correct as documented -- the issue
below is purely one of discoverability.

---

## 1. The `.hpp` directs the caller to a symbol it does not ship

`emit_tower_hpp()` (cbackend.py:875) writes this into every kernel header:

```c
/* fields: the per-point tower operands (type T), in the order
 * of xck_mgga_r_o1_scal_names; xc: the functional-derivative arrays
 * (type Txc; Libxc computes in double whatever T is). */
template <typename T, typename Txc = T>
int xck_mgga_r_o1_t(int64_t npts, int64_t nbf, const T* chi,
                    const T* const* fields, const Txc* const* xc,
                    T* out, T* work = nullptr);
```

But `xck_mgga_r_o1_scal_names` is emitted only by `emit_tower_cpp()`
(cbackend.py:1488), inside `extern "C"`:

```c
/* src/xck_mgga_r_o1.cpp */
const char* xck_mgga_r_o1_scal_names[8] = {
    "w", "rho_x", "rho_y", "rho_z", "vlapl", "vrho", "vsigma", "vtau" };
const int xck_mgga_r_o1_n_scal   = 8;
const int xck_mgga_r_o1_n_fields = 4;
```

A header-only consumer never compiles that translation unit, so the
header points at a symbol it cannot resolve. Measured over the 60
kernels generated above:

| | declares `scal_names` | declares `n_fields` / `n_scal` |
|---|---|---|
| `include/xckernel/kernels/*.hpp` | **0** | **0** |
| `src/*.cpp` | all | all |

This is not an unsupported configuration: `evaluator.hpp` explicitly
blesses it ("Header-only use without it falls back to the portable
loops"), and header-only is the only route for a header-only host --
NuKEXC is one, so it is the route taken here.

Two further wrinkles make the comment harder to act on than it looks:

* the two ABIs differ in shape. The C ABI takes **one** concatenated
  `scal` array with `n_fields` as the split point; the header-only
  signature takes **separate** `fields` and `xc`. So `scal_names` is not
  directly indexable against the signature the comment is attached to --
  the reader has to know to split it at `n_fields`.
* there is no `n_xc`. It is `n_scal - n_fields`, which is fine once
  known, but both of those live only in the `.cpp` too.

### Suggested fix

Emit the names and counts in the header, as the comment already implies,
and let the C ABI initialise from them so there is one source of truth:

```c
/* in emit_tower_hpp(), per kernel */
inline constexpr const char* xck_mgga_r_o1_field_names[] = {
    "w", "rho_x", "rho_y", "rho_z" };
inline constexpr const char* xck_mgga_r_o1_xc_names[] = {
    "vlapl", "vrho", "vsigma", "vtau" };
inline constexpr int xck_mgga_r_o1_n_fields = 4;
inline constexpr int xck_mgga_r_o1_n_xc     = 4;
```

Splitting them as `field_names` / `xc_names` matches the header-only
signature rather than the C one, which removes the split-at-`n_fields`
step. `emit_tower_cpp()` then keeps its `extern "C"` concatenated
`scal_names[]` for ABI compatibility, built from the two lists, so
nothing downstream changes and the two can no longer drift.

The same comment appears in `emit_batch_hpp()` (cbackend.py:1084) and
wants the same treatment.

---

## 2. Positional arrays of identically-typed pointers

A secondary point, and the reason finding 1 matters more than a doc bug
normally would.

For `xck_mgga_r_o1` the four `xc` entries are all `const double*`, all of
length `npts`. Passing them in the wrong order is therefore undetectable:
no type error, no arity error, no length error -- just wrong numbers. The
order here is `vlapl, vrho, vsigma, vtau`, and `mgga` is the one family
where that disagrees with Libxc's own declaration order
(`vrho, vsigma, vlapl, vtau`), so a host that wires it from memory or by
analogy with its GGA path gets it wrong and gets a plausible-looking
answer.

### Suggested fix

Emit a named operand struct per kernel alongside the array form:

```c
template <typename Txc>
struct xck_mgga_r_o1_xc_t {
    const Txc* vlapl; const Txc* vrho;
    const Txc* vsigma; const Txc* vtau;
};
```

with an overload taking it. `.vtau = ...` then makes a mis-ordering a
compile error. Purely additive; the array overload stays.

---

## 3. What NOT to fix: the ordering itself

The tempting fix is to renumber the `xc` arrays into Libxc's declaration
order. That would be a mistake.

The current order is neither arbitrary nor unstable. It is
`sorted(libxc_used, key=_libxc_sort_key)` with (codegen.py:383)

```python
def _libxc_sort_key(n: str):
    if n in LIBXC_MULTISET:
        return (len(LIBXC_MULTISET[n]), n)
    return (int(n[1]) if n[1].isdigit() else 1, n)
```

i.e. group by derivative order, then alphabetically within the group. At
first order `n[1]` is `r`/`s`/`t`/`l` -- never a digit -- so every entry
keys `(1, name)` and sorts alphabetically; at second order `v2rho2`,
`v2rhosigma`, `v2sigma2` group correctly by the leading digit. It is
deterministic across runs and principled.

Renumbering would silently break every consumer that wires `scal`
positionally, in exactly the undetectable way described in finding 2,
and would buy nothing that findings 1 and 2 do not already buy.

Who that is, checked rather than assumed:

| consumer | wires `scal` positionally? |
|---|---|
| external C users of the generated library | yes |
| generated Fortran module `xckernel_f03.f90` | yes -- `type(c_ptr) :: scal(*)`, no names exposed |
| the seven host writers in `xckernel/emitters/` | **no** -- none reference `scal_order` or `scal[`; they emit host-idiom code from the collapsed form and bypass the C ABI entirely |
| `xckernel/tests/cbackend_validate.py` | no -- builds operands **by name** through `scal_order(ck)` |

So the in-tree surface is already safe, and it is safe precisely because
it wires by name. That is the pattern finding 1 asks to make available to
header-only consumers, who currently cannot follow it. If the order ever
does change it should carry a version bump consumers can branch on.

---

## Status after v0.5.0

Both findings are fixed, and the fix for 1 took the shape suggested here:
the names are split to match the header-only signature rather than the C
one, and the C ABI now builds its concatenated array *from* the header's
lists, so the two cannot drift.

```c
/* include/xckernel/kernels/xck_mgga_r_o1.hpp, v0.5.0 */
inline constexpr int xck_mgga_r_o1_n_fields = 4;
inline constexpr int xck_mgga_r_o1_n_xc     = 4;
inline constexpr const char* xck_mgga_r_o1_field_names[] = {
    "w", "rho_x", "rho_y", "rho_z"};
inline constexpr const char* xck_mgga_r_o1_xc_names[] = {
    "vlapl", "vrho", "vsigma", "vtau"};
```

```c
/* src/xck_mgga_r_o1.cpp, v0.5.0 -- one source of truth */
const char* xck_mgga_r_o1_scal_names[8] = {
    xckernel::xck_mgga_r_o1_field_names[0], ...
    xckernel::xck_mgga_r_o1_xc_names[0], ... };
```

Finding 2 is fixed too, with named operand structs and an overload that
packs the positional arrays itself:

```c
template <typename Txc> struct xck_mgga_r_o1_xc {
    const Txc* vlapl; const Txc* vrho;
    const Txc* vsigma; const Txc* vtau; };
```

### What this changes for NuKEXC

The plan here had been to generate NuKEXC's binding from
`manifest.json`, because that was the only machine-readable place a
header-only consumer could read the operand order. **That is no longer
needed.** The struct overload moves the ordering from something NuKEXC
must get right into something the compiler enforces:

```cpp
xckernel::xck_gga_r_o1_xc<double> xc{.vrho = vrho.data(),
                                     .vsigma = vsigma.data()};
```

so the generator is dropped from the Phase 4 plan -- it would now be
complexity bought for nothing. The counts are additionally checkable at
compile time, which is what guards against a regeneration at a different
revision quietly changing arity:

```cpp
static_assert(xckernel::xck_gga_r_o1_n_fields == 4);
static_assert(xckernel::xck_gga_r_o1_n_xc == 2);
```

Re-measured on v0.5.0 through the struct overload, water/TZP, 31 basis
functions, 13180 points, NuKEXC screening disabled:

| functional | worst abs | relative |
|---|---|---|
| `XC_LDA_X` | 1.95e-14 | 1.5e-14 |
| `XC_GGA_X_PBE` | 1.55e-14 | 8.5e-15 |

unchanged from the positional form, as expected -- the overload only
packs the arrays.

Also relevant from the 0.5.0 notes: `--kinds` landed, so NuKEXC can
generate `exc,matrix` alone instead of all kinds. For the four families
at order 1 that is 16 kernels and 240 KiB rather than 60 and 1.5 MiB.
The `xck_kernel_info`
layout change in the same release does not affect a header-only
consumer, which never sees the index.
