#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ADJUST_PY="$SCRIPT_DIR/AdjustZeros.py"

usage() {
    cat <<'EOF'
Usage:
  AdjustZerosBatch.sh -d DIR [-g GLOB] [-o SUFFIX | -i] [-e] [-p]

Applies AdjustZeros.py to every gifti matching GLOB in DIR.

  -d DIR     Directory containing input giftis. (required)
  -g GLOB    Glob pattern to match within DIR. (default: *.func.gii)
  -o SUFFIX  Write output alongside each input, named
             "<input-basename><SUFFIX>.func.gii". (default: _zerofix)
  -i         Overwrite each input file in place instead of writing
             a suffixed copy. Mutually exclusive with -o.
  -e         Only adjust the eccentricity map (passes --ecc).
  -p         Only adjust the polar-angle map (passes --pol).
             If neither -e nor -p is given, both maps are adjusted.

Examples:
  # Non-destructive: write *_zerofix.func.gii next to each input.
  ./AdjustZerosBatch.sh -d ./sub-01/pRF

  # Overwrite every matching gifti in place.
  ./AdjustZerosBatch.sh -d ./sub-01/pRF -i

  # Only fix eccentricity, custom glob, custom suffix.
  ./AdjustZerosBatch.sh -d ./sub-01/pRF -g '*hemi-L*.func.gii' -o _eccfix -e
EOF
}

dir=""
glob="*.func.gii"
suffix="_zerofix"
in_place=0
do_ecc=0
do_pol=0
explicit_suffix=0

while getopts "d:g:o:iep h" opt; do
    case "$opt" in
        d ) dir="$OPTARG" ;;
        g ) glob="$OPTARG" ;;
        o ) suffix="$OPTARG"; explicit_suffix=1 ;;
        i ) in_place=1 ;;
        e ) do_ecc=1 ;;
        p ) do_pol=1 ;;
        h ) usage; exit 0 ;;
        * ) usage >&2; exit 1 ;;
    esac
done

if [[ -z "$dir" ]]; then
    echo "ERROR: -d DIR is required." >&2
    usage >&2
    exit 1
fi

if [[ ! -d "$dir" ]]; then
    echo "ERROR: Directory not found: $dir" >&2
    exit 1
fi

if [[ "$in_place" -eq 1 && "$explicit_suffix" -eq 1 ]]; then
    echo "ERROR: -o and -i are mutually exclusive." >&2
    usage >&2
    exit 1
fi

map_args=()
if [[ "$do_ecc" -eq 1 ]]; then map_args+=(--ecc); fi
if [[ "$do_pol" -eq 1 ]]; then map_args+=(--pol); fi

shopt -s nullglob
files=("$dir"/$glob)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
    echo "No files matching '$glob' found in $dir." >&2
    exit 1
fi

echo "Found ${#files[@]} file(s) matching '$glob' in $dir."

for in_gii in "${files[@]}"; do
    if [[ "$in_place" -eq 1 ]]; then
        echo "--- Adjusting (in place): $in_gii ---"
        python "$ADJUST_PY" "$in_gii" --in-place "${map_args[@]+"${map_args[@]}"}"
    else
        base="$(basename -- "$in_gii")"
        if [[ "$base" == *.func.gii ]]; then
            stem="${base%.func.gii}"
            out_gii="$dir/${stem}${suffix}.func.gii"
        else
            stem="${base%.*}"
            ext="${base##*.}"
            out_gii="$dir/${stem}${suffix}.${ext}"
        fi
        echo "--- Adjusting: $in_gii -> $out_gii ---"
        python "$ADJUST_PY" "$in_gii" "$out_gii" "${map_args[@]+"${map_args[@]}"}"
    fi
done
