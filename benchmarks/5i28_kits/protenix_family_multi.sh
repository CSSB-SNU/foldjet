#!/bin/bash
# usage: protenix_family_multi.sh KIT   (KIT = protenix_v1 | protenix_v2 | opendde)
# The 5I28 benchmark, ALL sizes in ONE kit process: one input JSON with the six items 5i28_x1 .. 5i28_x16 (ascending n); the kit loads once and
# runs each item with seeds 0-3 (first seed = cold for that shape, warm = seeds 1-3). Same kit command, mode, flags and timer span (PHASE total_s) as
# protenix_family_run.sh. Writes one result line per n to logs/all.results and done markers $DONE/kit_<KIT>_x<n>.
KIT=$1; K=$KITS_DIR; B=$BENCH_DIR; D=$B/$KIT; DONE=${DONE:-$B/done_final}
IN=$B/in/5i28_all.json
case $KIT in
  protenix_v1) VENV=$K/protenix_v1/venv130; KDIR=$K/protenix_v1/kit/protenix_v1; ROOTVAR=PROTENIX_ROOT_DIR; MODEL=protenix_base_default_v1.0.0; EXTRA="--kalign_binary_path $FF_SCRATCH/kalign_build/install/bin/kalign";;
  opendde)     VENV=$K/opendde/venv;        KDIR=$K/opendde/kit/opendde;         ROOTVAR=OPENDDE_ROOT_DIR;  MODEL=opendde_v1; EXTRA="--template_mmcif_dir $D/root/mmcif";;
  protenix_v2) VENV=$K/protenix_v2/venv;      KDIR=$K/protenix_v2/kit/protenix_v2;  ROOTVAR=PROTENIX_ROOT_DIR; MODEL=protenix-v2; EXTRA="--kalign_binary_path $FF_SCRATCH/kalign_build/install/bin/kalign";;
esac
mkdir -p $D/logs $D/tmp $D/out $DONE
cat > $D/tmp/inner_all.sh <<EOT
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip HF_HOME=$D/hf TMPDIR=$D/tmp
export MODEL_OPT_JIT_ROOT=$D/jit TRITON_CACHE_DIR=$D/jit/triton TORCH_EXTENSIONS_DIR=$D/jit/torch_ext TORCHINDUCTOR_CACHE_DIR=$D/jit/inductor
export PYTHONHASHSEED=0 CUDA_HOME=$K/cuda13home CC=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/gcc CXX=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/g++
export PATH=$VENV/bin:$FF_SCRATCH/kalign_build/install/bin:$K/cuda13home/bin:/usr/bin:/bin:/usr/local/bin $ROOTVAR=$D/root
. $VENV/bin/activate
cd $KDIR
bash run.sh install > $D/logs/install_all.log 2>&1 || cat $D/logs/install_all.log
bash run.sh pred --config h100 --mode fast -i $IN -o $D/out/all -s 0,1,2,3 -c 4 -p 200 -e 5 --use_msa true --use_template true $EXTRA --dtype bf16 -n $MODEL
EOT
rm -rf $D/out/all
L=$D/logs/all.log
echo "RUN $KIT all-n host=$(hostname) date=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" > $L
CU130_EXTRA_LD=$K/libstdcxx:$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib $K/cu130run.sh bash $D/tmp/inner_all.sh >> $L 2>&1
RC=$?
echo "CONTAINER_RC=$RC" >> $L
: > $D/logs/all.results
for n in 1 2 4 8 12 16; do
  FWD=$(grep -a "PHASE item=5i28_x$n " $L | awk '{v=""; for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="fwd_s") v=a[2]; else if(a[1]=="total_s" && v=="") t=a[2]} print (v!="")?v:t}' | tr "\n" " ")
  WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | grep . | sort -g | awk '{a[NR]=$1} END{if(NR) print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
  TR=$(grep -a "TEMPLATE event=census item=5i28_x$n " $L | head -1 | grep -oE 'real=[0-9]+ dummy=[0-9]+|real=[0-9]+/[0-9]+' | tr '\n' ';')
  NF=$(echo $FWD | wc -w); S=ok; [ "$NF" -lt 4 ] && { if grep -aqiE "out of memory|OutOfMemory|CUDA error: out of memory" $L; then S=oom; else S=failed; fi; }
  echo "KIT $KIT n=$n exit=$([ $S = ok ] && echo 0 || echo $S) fwd=[$FWD] warm_median=${WARM:-NA} templates=[$TR] active=$(grep -c 'ACTIVE mode=fast' $L) process_rc=$RC" | tee -a $D/logs/all.results
  echo "job=${SLURM_JOB_ID:-} host=$(hostname) single-process KIT $KIT n=$n status=$S fwd=[$FWD] warm_median=${WARM:-NA}" > $DONE/kit_${KIT}_x$n
done
