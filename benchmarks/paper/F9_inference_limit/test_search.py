"""Logic test of the F9 search with a fake probe (CPU only, no GPU): `python test_search.py`. Edge at n = 36 (37+ out of memory), and a non-monotonic variant."""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "common"))
import limit_search as ls  # noqa: E402


def fake(ok):
    def probe(a, n):
        return {"status": "ok" if ok(n) else "oom", "model_peak_alloc_gib": 1.0 * n, "process_peak_alloc_gib": 1.1 * n, "error": "" if ok(n) else "CUDA out of memory"}
    return probe


def run(ok, steps=None):
    ls.ff_probe = fake(ok)
    out = Path(tempfile.mkdtemp())
    sys.argv = ["limit_search.py", "--model", "af3", "--arm", "engine", "--out", str(out)] + (["--steps", *map(str, steps)] if steps else [])
    ls.main()
    return json.loads((out / "limit.json").read_text()), [json.loads(x)["n"] for x in (out / "sizes.jsonl").read_text().splitlines()]


s, tried = run(lambda n: n <= 36)
assert s["largest_ok_contiguous"] == 36 and s["first_not_ok"] == 37 and s["tokens_ok_contiguous"] == 36 * 128, s
assert tried[:5] == [16, 20, 24, 32, 40] and 36 in tried and 37 in tried and 38 in tried, tried  # steps, bisection (36, 38, 37 ...), probe-after
assert not s["non_monotonic"]
s, _ = run(lambda n: n <= 18 or 23 <= n <= 24)  # the 5090 pattern: 19-22 OOM, 23-24 ok
assert s["largest_ok_contiguous"] == 18 and s["non_monotonic"] and s["largest_ok_any"] == 24, s
s, _ = run(lambda n: False)
assert s["largest_ok_contiguous"] is None and s["first_not_ok"] == 1, s  # 16 fails -> bisect down to n = 1: even one copy does not complete
s, tried = run(lambda n: True, steps=[1, 2])
assert s["largest_ok_contiguous"] == 2 and s["first_not_ok"] is None, s
print("F9 search logic: OK")
