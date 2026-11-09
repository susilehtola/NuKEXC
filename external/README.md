# Vendored libxckernel headers

GENERATED -- do not edit. Regenerate with:

    python3 python/generate_xc_kernels.py

## Provenance

| | |
|---|---|
| source | https://github.com/susilehtola/libxckernel |
| version | `v0.7.0` (`5ae8376`) |
| families | `lda,gga,mgga_tau,mgga` |
| max order | `1` |
| kinds | `exc,matrix` |

Produced by, from a libxckernel checkout:

    python -m xckernel.catalog <out> lda,gga,mgga_tau,mgga 1 c --kinds exc,matrix

Only `include/xckernel/` is vendored, and only the kinds above. See the
header of `python/generate_xc_kernels.py` for why, and
`tests/reference/libxckernel-upstream.md` for the measurements that
establish these kernels reproduce NuKEXC's own XC assembly.

## What NuKEXC relies on here

* the templated header-only entry points `xck_<family>_<spin>_o<n>_t`,
  which need only `<cstdint>` and `<new>`;
* the named operand structs `xck_..._fields` / `xck_..._xc` and the
  overload taking them, so operand ORDER is enforced by the compiler
  rather than by NuKEXC getting an array right -- `xck_mgga_r_o1`'s xc
  order is `vlapl, vrho, vsigma, vtau`, which is deliberately not
  Libxc's declaration order;
* the `constexpr` counts `_n_fields` / `_n_xc`, which
  `nukexc/xc_kernels.hpp` static_asserts so a regeneration that changed
  a kernel's arity fails to compile instead of reading past an array;
* `xck_<name>_work(npts, nbf)`, asked rather than bounded. 0.7.0 grew it
  by nbf^2 per part for the half product its symmetric kernels now
  accumulate, and NuKEXC sizes a work slice PER SPATIAL BOX from it
  (`PackedCollocationStore`), so a hand-rolled bound would have silently
  started overlapping those slices.

The named structs and the counts arrived in 0.5.0; older releases will not
compile.
