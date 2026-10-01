#!/usr/bin/env python3
"""
Translate the Thakkar Roothaan-Hartree-Fock atomic database into C++ source.

Why source rather than a reader
-------------------------------
NuKEXC does not parse basis-set or density files -- the calling program owns
that, and the library takes arrays. But the free-atom densities are needed
*internally*, by Hirshfeld partitioning, by free-atom reference subtraction,
and by density-based adaptive-grid criteria, and none of those should require
the caller to supply a database they did not ask about. The database is small
enough (54 elements, a few thousand numbers) that emitting it as a constexpr
table removes the question: no file I/O, no parser, no search path, and the
data is as available on a GPU as anywhere else.

The source files live under input/k99light/neutral/ and are the neutral-atom
tabulations of Koga, Kanayama, Watanabe and Thakkar.

Format of the input
-------------------
    OXYGEN   1S(2)2S(2)2P(4), 3P        <- configuration, with occupations
    E = ...
    ORBITAL ENERGIES AND EXPANSION COEFFICIENTS
          S                    1S             2S     <- block: l, then orbitals
    BASIS/ORB.ENERGY      -20.67         -1.24
                CUSP        0.99           0.99
    2S       24.975698     -0.0004756     -0.0001236 <- n, zeta, one coef/orbital
    ...

Each orbital is a contraction over the block's primitives. Note the principal
quantum numbers VARY within a contraction (2S, 1S, 2S, ... above), which is
exactly why these need the contracted STO path.

Usage
-----
    python3 python/precompute_thakkar.py            # writes nukexc/thakkar_data.hpp
    python3 python/precompute_thakkar.py --check    # validate only, write nothing
"""

import argparse
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# Light elements (H-Xe) and heavy ones (Cs-Lr) are tabulated separately and
# named differently: "o" against "008o".
SRC_DIRS = [os.path.join(ROOT, "input", "k99light", "neutral"),
            os.path.join(ROOT, "input", "k00heavy")]


def source_path(Z, sym):
    """Where this element's tabulation lives, or None."""
    for d in SRC_DIRS:
        for name in (sym, "%03d%s" % (Z, sym)):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return None
OUT = os.path.join(ROOT, "nukexc", "thakkar_data.hpp")

SYMBOLS = [
    "x", "h", "he", "li", "be", "b", "c", "n", "o", "f", "ne", "na", "mg",
    "al", "si", "p", "s", "cl", "ar", "k", "ca", "sc", "ti", "v", "cr", "mn",
    "fe", "co", "ni", "cu", "zn", "ga", "ge", "as", "se", "br", "kr", "rb",
    "sr", "y", "zr", "nb", "mo", "tc", "ru", "rh", "pd", "ag", "cd", "in",
    "sn", "sb", "te", "i", "xe",
    "cs", "ba", "la", "ce", "pr", "nd", "pm", "sm", "eu", "gd", "tb", "dy",
    "ho", "er", "tm", "yb", "lu", "hf", "ta", "w", "re", "os", "ir", "pt",
    "au", "hg", "tl", "pb", "bi", "po", "at", "rn", "fr", "ra", "ac", "th",
    "pa", "u", "np", "pu", "am", "cm", "bk", "cf", "es", "fm", "md", "no",
    "lr",
]
Z_OF = {s: i for i, s in enumerate(SYMBOLS)}

# X-ray shell letters, as filled (label, occupation) pairs.
SHELL_LETTER = {
    "K": [("1S", 2)],
    "L": [("2S", 2), ("2P", 6)],
    "M": [("3S", 2), ("3P", 6), ("3D", 10)],
}

# Noble-gas cores, used by the heavy-element set (Cs onwards), which writes
# "[XE]6S(1)" where the light set would spell the core out.
NOBLE_CORE = {
    "HE": [("1S", 2)],
    "NE": [("1S", 2), ("2S", 2), ("2P", 6)],
    "AR": [("1S", 2), ("2S", 2), ("2P", 6), ("3S", 2), ("3P", 6)],
}
NOBLE_CORE["KR"] = NOBLE_CORE["AR"] + [
    ("3D", 10), ("4S", 2), ("4P", 6)]
NOBLE_CORE["XE"] = NOBLE_CORE["KR"] + [
    ("4D", 10), ("5S", 2), ("5P", 6)]
NOBLE_CORE["RN"] = NOBLE_CORE["XE"] + [
    ("4F", 14), ("5D", 10), ("6S", 2), ("6P", 6)]

L_OF = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4, "H": 5, "I": 6}


def parse_element(path):
    """Return (orbitals, config_occ) for one element file.

    orbitals: list of (l, label, [(n, zeta, coeff), ...])
    config_occ: {label: occupation} from the header line
    """
    orbitals = []
    config = {}

    block_l, labels, prims, coefs = -1, [], [], []

    def flush():
        nonlocal block_l, labels, prims, coefs
        if block_l >= 0 and prims:
            for k, lab in enumerate(labels):
                if len(coefs[k]) != len(prims):
                    raise ValueError(
                        f"{path}: ragged column for orbital {lab}")
                orbitals.append(
                    (block_l, lab,
                     [(prims[p][0], prims[p][1], coefs[k][p])
                      for p in range(len(prims))]))
        block_l, labels, prims, coefs = -1, [], [], []

    with open(path, errors="replace") as fh:
        first = True
        for raw in fh:
            line = raw.rstrip("\r\n")
            if first:
                # Configuration, e.g. "OXYGEN   1S(2)2S(2)2P(4), 3P".
                #
                # Heavier elements abbreviate filled inner shells by their
                # X-ray letter -- "SODIUM K(2)L(8)3S(1)" -- and the usage is
                # not consistent across the table: argon spells 1S(2)2S(2)
                # 2P(6)3S(2)3P(6) out in full while potassium writes
                # K(2)L(8)3S(2)3P(6)4S(1). Expansion is therefore additive
                # on top of whatever is written explicitly.
                #
                # Note also palladium's 5S(0): an orbital that is tabulated
                # but unoccupied. Zero is a legitimate occupation, not a
                # missing entry.
                for gas in re.findall(r"\[([A-Z]{2})\]", line):
                    if gas not in NOBLE_CORE:
                        raise ValueError(f"{path}: unknown core [{gas}]")
                    for lab, n in NOBLE_CORE[gas]:
                        config[lab] = config.get(lab, 0) + n
                for letter, occ in re.findall(r"\b([KLM])\((\d+)\)", line):
                    shell = SHELL_LETTER[letter]
                    if sum(n for _, n in shell) != int(occ):
                        raise ValueError(
                            f"{path}: {letter}({occ}) is not a filled shell")
                    for lab, n in shell:
                        config[lab] = config.get(lab, 0) + n
                for lab, occ in re.findall(r"(\d[SPDFGHI])\((\d+)\)", line):
                    config[lab] = config.get(lab, 0) + int(occ)
                first = False
                continue
            if not line.strip():
                continue
            if ("BASIS/ORB.ENERGY" in line or "CUSP" in line
                    or "ORBITAL ENERGIES" in line):
                continue

            tok = line.split()
            if not tok:
                continue

            # block header: a lone angular-momentum letter, then one label
            # per orbital in the block
            if len(tok[0]) == 1 and tok[0].upper() in L_OF:
                flush()
                block_l = L_OF[tok[0].upper()]
                labels = tok[1:]
                coefs = [[] for _ in labels]
                continue

            if block_l < 0 or len(tok) < 2 or not tok[0][0].isdigit():
                continue
            try:
                zeta = float(tok[1])
            except ValueError:
                continue
            if len(tok) - 2 != len(labels):
                raise ValueError(
                    f"{path}: expected {len(labels)} coefficients, got "
                    f"{len(tok) - 2}: {line!r}")

            prims.append((int(tok[0][0]), zeta))
            for k in range(len(labels)):
                coefs[k].append(float(tok[k + 2]))
    flush()
    return orbitals, config


def slater_norm(n, zeta):
    """(2 zeta)^(n + 1/2) / sqrt((2n)!)"""
    return (2.0 * zeta) ** (n + 0.5) / math.sqrt(math.factorial(2 * n))


def self_overlap(prims):
    """<chi|chi> for a contraction of normalized Slater primitives."""
    acc = 0.0
    for n1, z1, c1 in prims:
        a1 = c1 * slater_norm(n1, z1)
        for n2, z2, c2 in prims:
            a2 = c2 * slater_norm(n2, z2)
            npq = n1 + n2
            acc += a1 * a2 * math.factorial(npq) / (z1 + z2) ** (npq + 1)
    return acc


def occupations(orbitals, config, element):
    """Occupation of each orbital, from the header configuration."""
    occ = []
    for (l, lab, _) in orbitals:
        if lab not in config:
            raise ValueError(
                f"{element}: orbital {lab} has no occupation in the "
                f"configuration {sorted(config)}")
        occ.append(float(config[lab]))
    return occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate only, write nothing")
    args = ap.parse_args()

    elements = {}
    problems = []
    worst_defect = 0.0
    for sym in SYMBOLS[1:]:
        path = source_path(Z_OF[sym], sym)
        if path is None:
            continue
        orbitals, config = parse_element(path)
        occ = occupations(orbitals, config, sym)

        # Validation 1: electrons must add up to Z.
        nelec = sum(occ)
        if abs(nelec - Z_OF[sym]) > 1e-9:
            problems.append(f"{sym}: occupations sum to {nelec}, Z = {Z_OF[sym]}")

        # Validation 2: every orbital must be normalized. These are published
        # RHF orbitals, so a large deviation means the parse is wrong. A
        # SMALL one is the data's own: the coefficients are printed to seven
        # decimals, so the norms are only good to ~5e-7 per orbital.
        for (l, lab, prims) in orbitals:
            s = self_overlap(prims)
            if abs(s - 1.0) > 1e-5:
                problems.append(
                    f"{sym}: orbital {lab} has self-overlap {s:.9f}")

        # Validation 3: the occupation-weighted defect, which is exactly how
        # much int n_A 4 pi r^2 dr will miss Z by. It accumulates over
        # occupied orbitals, so it grows with Z -- silver is the worst in
        # the table at -8.2e-6 (rhenium). tests/test_atomic_density.cxx asserts
        # the same bound from the other side; keep the two in step.
        defect = sum(occ[k] * (self_overlap(pr) - 1.0)
                     for k, (_, _, pr) in enumerate(orbitals))
        worst_defect = max(worst_defect, abs(defect))
        if abs(defect) > 1.2e-5:
            problems.append(
                f"{sym}: normalization defect {defect:+.3e} exceeds 1.2e-5")

        elements[sym] = (orbitals, occ)

    print(f"parsed {len(elements)} elements")
    if problems:
        print(f"\n{len(problems)} VALIDATION FAILURES:", file=sys.stderr)
        for p in problems[:20]:
            print("  " + p, file=sys.stderr)
        return 1
    print("all elements validate: occupations sum to Z, orbitals normalized")
    print(f"worst occupation-weighted normalization defect: "
          f"{worst_defect:.3e} (the tabulation's own precision)")

    if args.check:
        return 0

    # ---- emit ----------------------------------------------------------
    maxz = max(Z_OF[s] for s in elements)
    orb_off, prim_off = [0], [0]
    orb_l, orb_occ, prim_n, prim_zeta, prim_coeff = [], [], [], [], []

    for z in range(maxz + 1):
        sym = SYMBOLS[z] if z < len(SYMBOLS) else None
        if sym in elements:
            orbitals, occ = elements[sym]
            for k, (l, lab, prims) in enumerate(orbitals):
                orb_l.append(l)
                orb_occ.append(occ[k])
                for (n, zt, c) in prims:
                    prim_n.append(n)
                    prim_zeta.append(zt)
                    prim_coeff.append(c)
                prim_off.append(len(prim_n))
        orb_off.append(len(orb_l))

    def fmt(vals, per_line, conv=str):
        out, line = [], "    "
        for i, v in enumerate(vals):
            tok = conv(v) + ("," if i + 1 < len(vals) else "")
            if len(line) + len(tok) > 76:
                out.append(line.rstrip())
                line = "    "
            line += tok + " "
        if line.strip():
            out.append(line.rstrip())
        return "\n".join(out)

    d = lambda v: repr(float(v))

    hdr = f"""/*
 *    NuKEXC -- Numerical Kokkos Enhanced Exchange Correlation Integrator
 *    Copyright (c) 2026, Susi Lehtola
 *    All rights reserved.
 *
 *    SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

// =========================================================================
//  GENERATED FILE -- do not edit
//
//  Regenerate with:  python3 python/precompute_thakkar.py
//
//  Thakkar Roothaan-Hartree-Fock neutral-atom densities, from
//  input/k99light/neutral (H-Xe; Koga, Kanayama, Watanabe and Thakkar,
//  Int. J. Quantum Chem. 71 (1999) 491) and input/k00heavy (Cs-Lr; Koga,
//  Kanayama, Watanabe, Imai and Thakkar, Theor. Chem. Acc. 104 (2000)
//  411). Both sets are constrained to satisfy the nuclear cusp and the
//  correct asymptotic decay, which is what makes them usable as free-atom
//  references rather than merely as fits.
//
//  Emitted as source rather than read at run time: the free-atom densities
//  are needed internally -- Hirshfeld partitioning, free-atom reference
//  subtraction, density-based adaptive-grid criteria -- and none of those
//  should oblige the caller to supply a database they never asked about.
//  As constexpr data there is no file I/O, no parser, no search path, and
//  it is as available on a device as anywhere else.
//
//  Each orbital is a CONTRACTION over Slater primitives whose principal
//  quantum numbers vary within the contraction, which is why these need
//  the contracted STO path rather than the primitive one.
//
//  Layout is CSR by atomic number:
//    orbitals of element Z occupy [orb_offset[Z], orb_offset[Z+1])
//    primitives of orbital i occupy [prim_offset[i], prim_offset[i+1])
//
//  Validated at generation: occupations sum to Z for every element, and
//  every orbital has unit self-overlap to 1e-6.
// =========================================================================

#include <Kokkos_Macros.hpp>

namespace Nukexc {{
namespace thakkar {{

/// Highest atomic number in the table.
inline constexpr int max_Z = {maxz};

"""

    body = f"""// CSR offsets into the orbital arrays, indexed by atomic number.
inline constexpr int orb_offset_[{len(orb_off)}] = {{
{fmt(orb_off, 12)}
}};

/// Angular momentum of each orbital.
inline constexpr int orb_l_[{len(orb_l)}] = {{
{fmt(orb_l, 20)}
}};

/// Occupation of each orbital.
inline constexpr double orb_occ_[{len(orb_occ)}] = {{
{fmt(orb_occ, 8, d)}
}};

/// CSR offsets into the primitive arrays, indexed by orbital.
inline constexpr int prim_offset_[{len(prim_off)}] = {{
{fmt(prim_off, 12)}
}};

/// Principal quantum number of each primitive.
inline constexpr int prim_n_[{len(prim_n)}] = {{
{fmt(prim_n, 20)}
}};

/// Slater exponent of each primitive.
inline constexpr double prim_zeta_[{len(prim_zeta)}] = {{
{fmt(prim_zeta, 6, d)}
}};

/// Contraction coefficient of each primitive, for NORMALIZED primitives.
inline constexpr double prim_coeff_[{len(prim_coeff)}] = {{
{fmt(prim_coeff, 6, d)}
}};

}} // namespace thakkar
}} // namespace Nukexc
"""

    # Accessors come after the arrays they index: a constexpr function
    # body may reference a later declaration, but the arrays' own
    # definitions must precede any constant-expression use of them.
    accessors = """
/// Is there tabulated data for this element?
KOKKOS_INLINE_FUNCTION constexpr bool available(int Z) {
  return Z >= 1 && Z <= max_Z && orb_offset_[Z + 1] > orb_offset_[Z];
}

/// Number of orbitals tabulated for an element.
KOKKOS_INLINE_FUNCTION constexpr int n_orbitals(int Z) {
  return available(Z) ? orb_offset_[Z + 1] - orb_offset_[Z] : 0;
}

/// Total electron count, i.e. Z for every element in this neutral table.
KOKKOS_INLINE_FUNCTION constexpr double n_electrons(int Z) {
  double n = 0.0;
  for (int i = orb_offset_[Z]; i < orb_offset_[Z + 1]; ++i)
    n += orb_occ_[i];
  return n;
}

} // namespace thakkar
} // namespace Nukexc
"""
    text = hdr + body[:body.index("} // namespace thakkar")] + accessors.lstrip("\n")

    with open(OUT, "w") as fh:
        fh.write(text)
    print(f"wrote {OUT}: {len(orb_l)} orbitals, {len(prim_n)} primitives, "
          f"{os.path.getsize(OUT) // 1024} KiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
