#!/bin/bash
# F9 full campaign: one job per (predictor, arm). Run from the card's checkout once, after the release is pinned.
#   H100: CARD=h100 NODE=node02 benchmarks/paper/F9_inference_limit/submit_all.sh      A100 (cssb3): CARD=a100 ...      5090: see README (shell loop under the GPU lock)
# Arms: engine torch (+ engine_fp8 on the H100) + the kits (H100 / A100 with a kit harness). A refused submit stops the script (ask; no other job shape).
set -euo pipefail
CARD=${CARD:?CARD=h100|a100}; HERE=$(cd "$(dirname "$0")" && pwd)
MODELS="protenix1 protenix2 opendde esmfold2 esmfold2-fast intellifold2 af3 boltz2 chai1 openfold3 rosettafold3"
KITS="af3_jax af3_torch openfold3 chai1 esmfold2 esmfold2_fast boltz2 protenix_v1 protenix_v2 opendde"   # rf3 and intellifold2: input builders not automated (README)
ARMS="engine torch"; [ "$CARD" = h100 ] && ARMS="engine engine_fp8 torch"
SB=(); [ -n "${NODE:-}" ] && SB+=(-w "$NODE")
for m in $MODELS; do for a in $ARMS; do sbatch "${SB[@]}" -J "f9-$a-$m" --export=ALL,MODEL=$m,ARM=$a "$HERE/$CARD.sbatch"; done; done
[ "${WITH_KITS:-1}" = 1 ] && for k in $KITS; do sbatch "${SB[@]}" -J "f9-kit-$k" --export=ALL,ARM=kit,KIT=$k "$HERE/$CARD.sbatch"; done
