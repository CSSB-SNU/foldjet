#!/bin/bash
#SBATCH -p h100 -A cssb --qos=cssb_h100 --gres=gpu:h100:1 -c 8 --mem=120G --time=02:00:00 -J F11inv-s3
#SBATCH -o /home/psk6950/foldforge-scratch/paper/F11_investigation/logs/%x_%j.log
# Step 3: per-block relative error of the trunk pair / single (last pass) bf16 vs Transition-only FP8, eager, seed 42.
R=/home/psk6950/foldforge-scratch/paper-306; B=$R/benchmarks/foldbench; D=/home/psk6950/foldforge-scratch/paper/F11_investigation
export FFENV=$R/.pixi/envs/default ENGINE_OVERLAY=$D/overlay FF_ALLOW_SIZE_MISMATCH=1
O=$D/step3
for t in ${TARGETS:-8f5n-assembly1 8dtn-assembly1 8im0-assembly1 7uvs-assembly1 8pb3-assembly1 8iha-assembly1}; do
  ( unset FOLDJET_ENGINE_FP8 FOLDJET_ENGINE_FP8_MODEL; bash $B/run_py.sh $D/scripts/step3.py ref --target $t --out $O ) 2>&1 | grep -E "^REF|Error|Traceback" 
  ( export FOLDJET_ENGINE_FP8=safe FOLDJET_ENGINE_FP8_MODEL=af3 FOLDJET_ENGINE_TRANSITION_FP8_GUARD=lazy F11INV_TRIATTN=none FOLDJET_ENGINE_TRIMUL_FP8= FOLDJET_ENGINE_OPM_FP8= FOLDJET_ENGINE_PWA_FP8= FOLDJET_ENGINE_TOKEN_DIT_FP8= FOLDJET_ENGINE_ATOMDIT_FP8= F11INV_TR_LOG=$O/$t/trcalls.txt CFG=only_transition
    rm -f $O/$t/trcalls.txt; bash $B/run_py.sh $D/scripts/step3.py cmp --target $t --out $O ) 2>&1 | grep -E "^CMP|Error|Traceback"
done
echo END
