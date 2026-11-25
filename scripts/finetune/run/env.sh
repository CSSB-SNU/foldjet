# source this: python of the FoldJet pixi env (no `pixi run`), this checkout's src, the MiniWorld source tree, the MiniWorld data adapter.
# Override FFENV / MINIWORLD_SRC / FT_EXTRA_SITE for another machine.
FT_RUN_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export FF_REPO=$(cd "$FT_RUN_DIR/../../.." && pwd)
export FT_ADAPTER=$FF_REPO/scripts/finetune/miniworld_adapter
export FFENV=${FFENV:-$HOME/FoldForge/.pixi/envs/default}
export MINIWORLD_SRC=${MINIWORLD_SRC:-$HOME/MiniWorld/src}
export PATH=$FFENV/bin:$PATH
export LD_LIBRARY_PATH=$FFENV/lib:${LD_LIBRARY_PATH:-}
# FT_EXTRA_SITE: packages MiniWorld needs that the FoldJet env lacks (on cssb: pyarrow symlinked from MiniWorld's env)
export PYTHONPATH=$FT_ADAPTER:$FF_REPO/src:$MINIWORLD_SRC${FT_EXTRA_SITE:+:$FT_EXTRA_SITE}${PYTHONPATH:+:$PYTHONPATH}
export PYTHONFAULTHANDLER=1
