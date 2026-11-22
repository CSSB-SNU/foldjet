#!/bin/bash
# Remove the output of one task (a contended run is not a result). b200_purge.sh <exp:model:arg>
IFS=: read -r EXP MODEL ARG <<<"$1"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/common/cards/b200.env"
case $EXP in
  f3) rm -rf "$PAPER_OUT/F3/$CARD/engine_rep$ARG/$MODEL" ;;
  f4) rm -f "$PAPER_OUT/F4/$CARD/rows/${ARG}__$MODEL.jsonl"; rm -rf "$PAPER_OUT/F4/$CARD/pred/$ARG/$MODEL" ;;
  f9) rm -rf "$PAPER_OUT/F9/$CARD/$ARG/$MODEL" ;;
  f6) O=$PAPER_OUT/F6_training_speed/$CARD; rm -f "$O/${MODEL}_${ARG}_a"*"_r"*.done "$O/${MODEL}_${ARG}_a"*"_r"*.log; rm -rf "$O/runs/${MODEL}_${ARG}_a"* ;;
esac
echo "purged $1"
