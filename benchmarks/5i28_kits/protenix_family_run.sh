#!/bin/bash
# usage: protenix_family_run.sh KIT N   (KIT = protenix_v1 | opendde | protenix_v2), the 5I28 benchmark (benchmarks/5i28.md): real MSA + 20-hit
# template a3m (the kit keeps 5syd_A, 6kol_A, 5yt7_A, 6tfd_A), fast mode, 4 recycles, 200 steps, 5 samples, bf16, seeds 0-3 (warm = 1-3).
# Runs inside the kit container (cu130run.sh). Prints one KIT summary line.
KIT=$1; N=$2
K=$KITS_DIR; B=$BENCH_DIR; D=$B/$KIT
IN=$B/in/5i28_x$N.json
case $KIT in
  protenix_v1) VENV=$K/protenix_v1/venv130; KDIR=$K/protenix_v1/kit/protenix_v1; ROOTVAR=PROTENIX_ROOT_DIR; MODEL=protenix_base_default_v1.0.0; EXTRA="--kalign_binary_path $FF_SCRATCH/kalign_build/install/bin/kalign";;
  opendde)     VENV=$K/opendde/venv;        KDIR=$K/opendde/kit/opendde;         ROOTVAR=OPENDDE_ROOT_DIR;  MODEL=opendde_v1; EXTRA="--template_mmcif_dir $D/root/mmcif";;
  protenix_v2) VENV=$K/protenix_v2/venv;      KDIR=$K/protenix_v2/kit/protenix_v2;  ROOTVAR=PROTENIX_ROOT_DIR; MODEL=protenix-v2; EXTRA="--kalign_binary_path $FF_SCRATCH/kalign_build/install/bin/kalign";;
esac
mkdir -p $D/logs $D/tmp $D/out
cat > $D/tmp/inner_x$N.sh <<EOT
export XDG_CACHE_HOME=$D/cache UV_CACHE_DIR=$D/cache/uv PIP_CACHE_DIR=$D/cache/pip HF_HOME=$D/hf TMPDIR=$D/tmp
export MODEL_OPT_JIT_ROOT=$D/jit TRITON_CACHE_DIR=$D/jit/triton TORCH_EXTENSIONS_DIR=$D/jit/torch_ext TORCHINDUCTOR_CACHE_DIR=$D/jit/inductor
export PYTHONHASHSEED=0 CUDA_HOME=$K/cuda13home CC=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/gcc CXX=$FF_SCRATCH/cuda13tk/.pixi/envs/default/bin/g++
export PATH=$VENV/bin:$FF_SCRATCH/kalign_build/install/bin:$K/cuda13home/bin:/usr/bin:/bin:/usr/local/bin $ROOTVAR=$D/root
. $VENV/bin/activate
cd $KDIR
bash run.sh install > $D/logs/install_x$N.log 2>&1 || cat $D/logs/install_x$N.log
bash run.sh pred --config h100 --mode fast -i $IN -o $D/out/x$N -s 0,1,2,3 -c 4 -p 200 -e 5 --use_msa true --use_template true $EXTRA --dtype bf16 -n $MODEL
EOT
rm -rf $D/out/x$N
CU130_EXTRA_LD=$K/libstdcxx:$FF_SCRATCH/cuda13tk/.pixi/envs/default/lib $K/cu130run.sh bash $D/tmp/inner_x$N.sh > $D/logs/x$N.log 2>&1
RC=$?
L=$D/logs/x$N.log
FWD=$(grep -E "PHASE item=" $L | awk '{v=""; for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="fwd_s") v=a[2]; else if(a[1]=="total_s" && v=="") t=a[2]} print (v!="")?v:t}' | tr "\n" " ")
WARM=$(echo $FWD | tr ' ' '\n' | tail -n +2 | sort -g | awk '{a[NR]=$1} END{if(NR) print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')
TR=$(grep -oE 'real=[0-9]+/[0-9]+|real=[0-9]+ dummy=[0-9]+|Found [0-9]+ templates' $L | sort | uniq -c | tr -s ' ' | tr '\n' ';')
MSA=$(grep -oE 'N_msa [0-9]+' $L | sort -u | tr '\n' ';')
echo "KIT $KIT n=$N exit=$RC fwd=[$FWD] warm_median=$WARM templates=[$TR] msa=[$MSA] active=$(grep -c 'ACTIVE mode=fast' $L)"
