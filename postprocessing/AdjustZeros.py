#!/usr/bin/env python
"""
Replace gated zero entries in a pRF func.gii with a small epsilon.

Assumed GIfTI data-array order:
    darray[0]: polar angle
    darray[1]: variance explained
    darray[2]: eccentricity

For each selected map (polar angle and/or eccentricity), entries satisfying
    (map == 0) & (variance explained > 0)
are replaced with EPS (1e-6), since a literal zero is ambiguous between
"no signal" and "a real zero-valued fit" once variance explained is nonzero.
"""
import argparse
import sys

import numpy as np
import nibabel as nib

EPS = 1e-6


def adjust_zeros(img, *, do_ecc, do_pol):
    if len(img.darrays) < 3:
        raise ValueError(
            f"Expected at least 3 GIfTI data arrays, found {len(img.darrays)}."
        )

    polar_angle = img.darrays[0].data
    variance_explained = img.darrays[1].data
    eccentricity = img.darrays[2].data

    if polar_angle.shape != variance_explained.shape:
        raise ValueError(
            "Polar-angle and variance-explained arrays have different shapes: "
            f"{polar_angle.shape} vs {variance_explained.shape}"
        )
    if eccentricity.shape != variance_explained.shape:
        raise ValueError(
            "Eccentricity and variance-explained arrays have different shapes: "
            f"{eccentricity.shape} vs {variance_explained.shape}"
        )

    reports = []

    if do_pol:
        mask = (polar_angle == 0) & (variance_explained > 0)
        indices = np.flatnonzero(mask)
        polar_angle[mask] = np.asarray(EPS, dtype=polar_angle.dtype)
        n_remaining = int(np.count_nonzero((polar_angle == 0) & (variance_explained > 0)))
        reports.append(("polar-angle", indices, n_remaining))

    if do_ecc:
        mask = (eccentricity == 0) & (variance_explained > 0)
        indices = np.flatnonzero(mask)
        eccentricity[mask] = np.asarray(EPS, dtype=eccentricity.dtype)
        n_remaining = int(np.count_nonzero((eccentricity == 0) & (variance_explained > 0)))
        reports.append(("eccentricity", indices, n_remaining))

    return reports


def main():
    parser = argparse.ArgumentParser(
        description="Replace gated zero entries (map == 0 & varexp > 0) with 1e-6.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", help="Input .func.gii file.")
    out_group = parser.add_mutually_exclusive_group(required=True)
    out_group.add_argument("output", nargs="?", default=None, help="Output .func.gii file.")
    out_group.add_argument("--in-place", action="store_true", help="Overwrite the input file.")

    map_group = parser.add_argument_group(
        "map selection (default: both, if neither flag is given)"
    )
    map_group.add_argument("--ecc", action="store_true", help="Adjust the eccentricity map.")
    map_group.add_argument("--pol", action="store_true", help="Adjust the polar-angle map.")

    args = parser.parse_args()

    if args.in_place and args.output is not None:
        parser.error("cannot pass both OUTPUT and --in-place.")
    if not args.in_place and args.output is None:
        parser.error("must pass either OUTPUT or --in-place.")

    if args.in_place:
        out_file = args.input
        print("WARNING:", file=sys.stderr)
        print("    Overwriting input file in place.", file=sys.stderr)
        print(f"    Input/output are identical: {args.input}", file=sys.stderr)
        print("Continuing but be wary", file=sys.stderr)
    else:
        out_file = args.output
        if args.input == out_file:
            parser.error("input and output paths are identical. Use --in-place if overwriting is intentional.")

    do_ecc, do_pol = args.ecc, args.pol
    if not do_ecc and not do_pol:
        do_ecc = do_pol = True

    img = nib.load(args.input)
    reports = adjust_zeros(img, do_ecc=do_ecc, do_pol=do_pol)
    nib.save(img, out_file)

    print(f"Input: {args.input}")
    print(f"Output: {out_file}")
    for name, indices, n_remaining in reports:
        n_changed = int(indices.size)
        print(f"Changed {n_changed} {name} vertices from 0 to {EPS} where variance explained > 0.")
        print(f"Remaining gated zero-{name} vertices: {n_remaining}")
        if n_changed > 0:
            print(f"First up-to-20 changed {name} vertex indices: {indices[:20].tolist()}")


if __name__ == "__main__":
    main()
