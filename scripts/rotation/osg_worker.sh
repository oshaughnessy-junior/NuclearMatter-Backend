#!/bin/bash
set -euo pipefail
energy=$1
axis_ratio=$2
export NMB_RNS_Q=$axis_ratio
mkdir result
cp rns_export result/
cd result
timeout 120 ./rns_export -q poly -N 1 -e "$energy" > run.log 2>&1
rm rns_export
printf 'energy=%s axis_ratio=%s host=%s\n' "$energy" "$axis_ratio" "$(hostname -f)" > request.txt
sha256sum rns_spherical.csv rns_rotating.csv > sha256.txt
cd ..
tar -czf result.tar.gz result
