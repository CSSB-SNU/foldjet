# Shared defaults of the paper experiments (source it: `. common/env.sh`). Override any of these in the environment.
#   FF_SCRATCH   scratch root (default ~/foldforge-scratch)
#   PAPER_OUT    run outputs, NOT in the repository: $PAPER_OUT/<ID>/<card>/<run name>   (default $FF_SCRATCH/paper)
#   FB_SPECS     FoldJet specs of the 1,522 FoldBench targets (benchmarks/foldbench/build_specs.py output)
#   FFENV        the FoldJet pixi env (never `pixi run` on a shared env)
#   ENGINE_OVERLAY  an extracted engine wheel in front of the env's engine (release-candidate runs); empty = the env's own engine
export FF_SCRATCH=${FF_SCRATCH:-$HOME/foldforge-scratch}
export PAPER_OUT=${PAPER_OUT:-$FF_SCRATCH/paper}
export FB_SPECS=${FB_SPECS:-$FF_SCRATCH/foldbench/specs_all}
export FF_ALLOW_SIZE_MISMATCH=${FF_ALLOW_SIZE_MISMATCH:-1}   # local checkpoint blobs of another converter version (see benchmarks/foldbench/README.md)
