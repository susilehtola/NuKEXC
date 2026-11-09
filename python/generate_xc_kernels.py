#!/usr/bin/env python3
"""
Vendor libxckernel's generated XC kernel headers into external/xckernel/.

Why vendor rather than depend
-----------------------------
libxckernel is a *generator*: the C/C++ library is a generated artifact,
not something one finds installed. Generating it needs Python and sympy,
which NuKEXC's build does not otherwise require and CI should not have to
grow. Generation is also deterministic and takes about a second, so the
output is exactly the kind of thing that belongs in the tree next to
nukexc/thakkar_data.hpp: checked in, with the command that produced it
recorded.

Only the HEADER-ONLY half is vendored. Each kernel header is a set of
constexpr coefficient tables plus a templated `XCK_HD inline` entry point
needing nothing but <cstdint> and <new>, which is what lets a header-only
library like NuKEXC consume it at all. The C ABI (src/*.cpp, the Fortran
module, the index) is not vendored: it would make NuKEXC link a library
to call functions it can already inline, and its `xck_kernel_info` index
is the one part of the ABI whose layout moves between releases.

Only the kinds NuKEXC uses are generated. `exc` is the energy
(Exc = sum_g w_g rho_g zk_g) and `matrix` is the Fock matrix; everything
else in the catalog -- gradients, Hessians, MO-projected response, GIAO
-- belongs to phases NuKEXC has not reached. For the four families at
order 1 that is 16 kernels and 74 KiB as the generator reports it, against
60 and 1.5 MiB for all kinds.

Usage
-----
    python3 python/generate_xc_kernels.py [--libxckernel DIR] [--check]

--check regenerates into a temporary directory and diffs against the
vendored copy without writing, which is what CI would run to catch a
tree that no longer matches its recorded provenance.
"""

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "external", "xckernel")

#: What NuKEXC consumes. Order 1 is energy and Fock; response is Phase 5.
FAMILIES = "lda,gga,mgga_tau,mgga"
MAX_ORDER = "1"
KINDS = "exc,matrix"

#: The release this tree was generated from. Bump deliberately: the
#: operand-name constants the binding static_asserts against arrived in
#: 0.5.0, so anything older will not compile. 0.7.0 is where the symmetric
#: matrix kernels stopped paying a GEMM for both members of a transposed
#: pattern pair -- gga o1 went from four products to one -- which is worth
#: re-measuring after, not just rebuilding.
WANT_TAG = "v0.7.0"


def libxckernel_version(src):
    """(tag, sha) of the libxckernel checkout, for the provenance note."""
    def git(*a):
        return subprocess.run(["git", "-C", src, *a], capture_output=True,
                              text=True).stdout.strip()
    sha = git("rev-parse", "--short", "HEAD")
    tag = git("describe", "--tags", "--exact-match", "HEAD") or git(
        "describe", "--tags")
    return tag, sha


def generate(src, outdir):
    """Run libxckernel's catalog generator. Returns (tag, sha)."""
    tag, sha = libxckernel_version(src)
    if not sha:
        raise SystemExit(f"{src} is not a git checkout of libxckernel")
    if tag != WANT_TAG:
        print(f"warning: libxckernel is at {tag or sha}, not {WANT_TAG}; "
              f"the vendored tree records whatever is generated",
              file=sys.stderr)

    cmd = [sys.executable, "-m", "xckernel.catalog", outdir,
           FAMILIES, MAX_ORDER, "c", "--kinds", KINDS]
    r = subprocess.run(cmd, cwd=src, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"generation failed:\n{r.stdout}\n{r.stderr}")
    return tag, sha


def readme(tag, sha):
    return f"""\
# Vendored libxckernel headers

GENERATED -- do not edit. Regenerate with:

    python3 python/generate_xc_kernels.py

## Provenance

| | |
|---|---|
| source | https://github.com/susilehtola/libxckernel |
| version | `{tag}` (`{sha}`) |
| families | `{FAMILIES}` |
| max order | `{MAX_ORDER}` |
| kinds | `{KINDS}` |

Produced by, from a libxckernel checkout:

    python -m xckernel.catalog <out> {FAMILIES} {MAX_ORDER} c --kinds {KINDS}

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
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--libxckernel", default="/home/work/libxckernel",
                    help="path to a libxckernel checkout")
    ap.add_argument("--check", action="store_true",
                    help="regenerate and diff; write nothing")
    args = ap.parse_args()

    src = os.path.abspath(args.libxckernel)
    if not os.path.isdir(os.path.join(src, "xckernel")):
        raise SystemExit(f"no libxckernel checkout at {src}")

    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "cat")
        tag, sha = generate(src, out)
        inc = os.path.join(out, "include", "xckernel")
        if not os.path.isdir(inc):
            raise SystemExit("generator produced no include/xckernel")

        n = len([f for f in os.listdir(os.path.join(inc, "kernels"))
                 if f.endswith(".hpp")])

        # config.h is written by libxckernel's CMake for the compiled
        # library. Header-only use falls back to the portable loops and
        # evaluator.hpp guards the include with __has_include, so the
        # template is dropped on vendoring -- and must therefore be
        # ignored when diffing, or --check always reports a difference.
        SKIP = {"config.h.in"}

        if args.check:
            if not os.path.isdir(DEST):
                raise SystemExit(f"{DEST} does not exist; run without --check")
            diff = []
            for dirpath, _, files in os.walk(inc):
                for f in files:
                    a = os.path.join(dirpath, f)
                    b = os.path.join(DEST, os.path.relpath(a, inc))
                    rel = os.path.relpath(a, inc)
                    if rel in SKIP:
                        continue
                    if not os.path.exists(b) or not filecmp.cmp(a, b,
                                                                shallow=False):
                        diff.append(rel)
            if diff:
                print(f"vendored tree differs from {tag} ({sha}):",
                      file=sys.stderr)
                for d in sorted(diff)[:20]:
                    print("  " + d, file=sys.stderr)
                return 1
            print(f"vendored tree matches {tag} ({sha}): {n} kernels")
            return 0

        if os.path.isdir(DEST):
            shutil.rmtree(DEST)
        shutil.copytree(inc, DEST)
        for f in SKIP:
            cfg = os.path.join(DEST, f)
            if os.path.exists(cfg):
                os.remove(cfg)
        with open(os.path.join(os.path.dirname(DEST), "README.md"), "w") as fh:
            fh.write(readme(tag, sha))

        kib = sum(os.path.getsize(os.path.join(dp, f))
                  for dp, _, fs in os.walk(DEST) for f in fs) // 1024
        print(f"vendored {n} kernels from {tag} ({sha}) -> {DEST} ({kib} KiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
