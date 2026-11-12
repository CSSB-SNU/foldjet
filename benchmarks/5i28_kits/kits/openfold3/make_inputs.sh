#!/bin/bash
# OpenFold3 query JSON for the 5I28 benchmark: n identical chains (one entry, chain_ids A..), the real MSA (inputs/msa/A/colabfold_main.a3m,
# 6789 a3m rows incl. the query) and the 4 distinct real templates in CIF-direct mode (template_cif_paths + template_cif_chain_ids):
# 5syd_A, 6kol_A, 5yt7_A, 6tfd_A (gunzipped from ~/data/cif_raw). OF3 keys its template slots <entry>_<chain>, so 4 distinct entries = 4 slots.
# The JSON holds COPIES identical queries q0..q<COPIES-1>: one kit process, COPIES forwards (the first cold).
# usage: make_inputs.sh                  -> inputs/x<N>.json for N = 1 2 4 8 12 16 (COPIES = 4 at n=1, else 3)
#        make_inputs.sh N COPIES OUT     -> one file
set -euo pipefail
D=$BENCH_DIR/openfold3
SEQ=$(awk '!/^>/ {print; exit}' "$D/inputs/msa/A/colabfold_main.a3m")
[ ${#SEQ} -eq 128 ] || { echo "query length ${#SEQ} != 128" >&2; exit 1; }
TPL=(5syd:A 6kol:A 5yt7:A 6tfd:A)
one() {  # N COPIES
  local n=$1 c=$2 ids="" k paths="" chains=""
  for ((k = 0; k < n; k++)); do ids+="${ids:+, }\"$(printf "\\x$(printf %x $((65 + k)))")\""; done
  for t in "${TPL[@]}"; do paths+="${paths:+, }\"$D/inputs/template_cifs/${t%%:*}.cif\""; chains+="${chains:+, }\"${t##*:}\""; done
  local q="{\"chains\": [{\"molecule_type\": \"protein\", \"chain_ids\": [$ids], \"sequence\": \"$SEQ\", \"main_msa_file_paths\": \"$D/inputs/msa/A\", \"template_cif_paths\": [$paths], \"template_cif_chain_ids\": [$chains]}]}"
  printf '{"seeds": [0], "queries": {'
  for ((k = 0; k < c; k++)); do printf '%s"q%d": %s' "$([ $k -gt 0 ] && echo ', ')" $k "$q"; done
  printf '}}\n'
}
if [ $# -eq 3 ]; then one "$1" "$2" > "$3"; exit 0; fi
for n in 1 2 4 8 12 16; do one $n $([ $n = 1 ] && echo 4 || echo 3) > "$D/inputs/x$n.json"; echo "wrote $D/inputs/x$n.json"; done
