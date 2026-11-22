#!/bin/bash
# b200_ready.sh <model>: exit 0 when the weights files of the predictor are complete on the B200 (rsync renames a file only when it is complete).
C=${FOLDFORGE_HOME:-/NHNHOME/WORKSPACE/26mohw002_A/psk6950/paper-b200/ff/home}/checkpoints
need() { for f in "$@"; do [ -s "$C/$f" ] || exit 1; done; exit 0; }
case $1 in
  af3) need af3/af3.miniworld.pt ;;
  boltz2) need boltz2/boltz2.miniworld.pt ;;
  chai1) need chai1/chai1.miniworld.pt chai1/release/esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt chai1/release/conformers_v1.apkl ;;
  esmfold2) need esmfold2/esmfold2.miniworld.pt esmfold2/esmfold2.lm.npz esmc-6b/tokenizer.json ;;
  esmfold2-fast) need esmfold2-fast/esmfold2_fast.miniworld.pt esmfold2-fast/esmfold2_fast.lm.npz esmc-6b/tokenizer.json ;;
  intellifold2) need intellifold2/intellifold2.miniworld.pt ;;
  opendde) need opendde/opendde.miniworld.pt ;;
  openfold3) need openfold3/openbind0.miniworld.pt ;;
  openfold3-preview2) need openfold3/openfold3.miniworld.pt ;;
  protenix1) need protenix/protenix1.miniworld.pt ;;
  protenix2) need protenix/protenix2.miniworld.pt ;;
  rosettafold3) need rosettafold3/rosettafold3.miniworld.pt ;;
  *) exit 1 ;;
esac
