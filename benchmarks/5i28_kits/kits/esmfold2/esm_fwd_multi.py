"""ESMFold2 kit timing driver, ALL sizes in ONE process (derived from the kit harness esm_fwd_kit_msa.py: same MSA, builder.fold call, synchronized
timer span; model + MSA loaded once). SIZES (default 1 2 4 8 12 16) ascending; FWD_CALLS forwards per size (first = cold for that shape).
An out-of-memory in one size is recorded (`FWDOOM <n>`) and the caches are emptied; the process goes on with the next size."""
import os, sys, time, json
import torch
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
from esm.models.esmfold2 import ESMFold2InputBuilder
from esm.utils.msa import MSA
from esm.utils.structure.input_builder import ProteinInput, StructurePredictionInput
SEQ = "AECSVDIQGNDQMQFNTNAITVDKSCKQFCVNLSHPGNLPKNVMGHNWVLSTAADMQGVVTDGMASGLDKDYLKPDDSRVIAHTKLIGSGEKDSVTFDVSKLKEGEQYMFFCTFPGHSALMKGTLTLK"
A3M = os.environ.get("A3M") or os.environ["A3M_5I28"]; seed = 0
sizes = [int(x) for x in os.environ.get("SIZES", "1 2 4 8 12 16").split()]
msa = MSA.from_a3m(A3M)
print("[harness] MSA loaded from", A3M, "depth(rows)", msa.depth, "query_ok", msa.sequences[0] == SEQ, flush=True)
t0 = time.perf_counter()
model = ESMFold2Model.from_pretrained("biohub/ESMFold2").to("cuda").eval()
print(f"LOADTIME {time.perf_counter()-t0:.1f}", flush=True)
builder = ESMFold2InputBuilder()
for n in sizes:
    ids = [chr(ord("A") + i) for i in range(n)]
    spi = StructurePredictionInput(sequences=[ProteinInput(id=i, sequence=SEQ, msa=msa) for i in ids])
    print(f"[harness] SIZE n={n} start {time.strftime('%H:%M:%S')}", flush=True)
    for k in range(int(os.environ.get("FWD_CALLS", "4"))):
        try:
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                results = builder.fold(model, spi, num_loops=3, num_sampling_steps=200, num_diffusion_samples=5, seed=seed)
            torch.cuda.synchronize(); print(f"FWDTIME n={n} {k} {time.perf_counter() - t:.3f}", flush=True)
        except torch.OutOfMemoryError as e:
            print(f"FWDOOM {n} {k} {str(e)[:120]!r}", flush=True)
            results = None; import gc; gc.collect(); torch.cuda.empty_cache(); break
    print(f"[harness] SIZE n={n} done peak_mem_GB {torch.cuda.max_memory_allocated()/2**30:.1f} {time.strftime('%H:%M:%S')}", flush=True)
    torch.cuda.reset_peak_memory_stats()
