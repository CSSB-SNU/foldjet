#!/bin/bash
# OpenFold3 0.4.1 query JSON for the 5I28 benchmark: n identical chains, the real MSA (inputs/msa/A) and the 4 distinct real templates
# 5syd_A, 6kol_A, 5yt7_A, 6tfd_A as a template alignment file (0.4.1 has no CIF-direct template fields): inputs/templates_4.a3m = the query
# line + those 4 hits of benchmarks/5i28_data/5i28_templates_hits20.a3m; structures from inputs/template_cifs (runner.yml structure_directory).
# usage: make_inputs.sh N COPIES OUT
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
SEQ=$(awk '!/^>/ {print; exit}' "$D/inputs/msa/A/colabfold_main.a3m")
n=$1 c=$2 ids=""
for ((k = 0; k < n; k++)); do ids+="${ids:+, }\"$(printf "\\x$(printf %x $((65 + k)))")\""; done
q="{\"chains\": [{\"molecule_type\": \"protein\", \"chain_ids\": [$ids], \"sequence\": \"$SEQ\", \"main_msa_file_paths\": \"$D/inputs/msa/A\", \"template_alignment_file_path\": \"$D/inputs/templates_4.a3m\", \"template_entry_chain_ids\": [\"5syd_A\", \"6kol_A\", \"5yt7_A\", \"6tfd_A\"]}]}"
{ printf '{"seeds": [0], "queries": {'; for ((k = 0; k < c; k++)); do printf '%s"q%d": %s' "$([ $k -gt 0 ] && echo ', ')" $k "$q"; done; printf '}}\n'; } > "$3"
