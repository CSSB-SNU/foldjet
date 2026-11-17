"""`import foldforge` sets expandable segments unless the user set an allocator conf."""

import json
import os
import subprocess
import sys

import pytest

from foldforge.allocator import DEFAULT_ALLOC_CONF

VARIABLES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF")
REPORT = (
    "import json, os; print(json.dumps({k: os.environ.get(k) for k in "
    "('PYTORCH_CUDA_ALLOC_CONF', 'PYTORCH_ALLOC_CONF')}))"
)


def _after_import(preamble: str, **set_vars: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in VARIABLES}
    env.update(set_vars)
    result = subprocess.run(  # noqa: S603 - fixed trusted Python code
        [sys.executable, "-c", f"{preamble}; {REPORT}"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("preamble", ["import foldforge", "import torch, foldforge"])
def test_default_applied_when_absent(preamble):
    seen = _after_import(preamble)
    assert seen == {
        "PYTORCH_CUDA_ALLOC_CONF": DEFAULT_ALLOC_CONF,
        "PYTORCH_ALLOC_CONF": None,
    }


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64"),
        ("PYTORCH_CUDA_ALLOC_CONF", ""),  # present but empty: torch's own defaults
        ("PYTORCH_ALLOC_CONF", "expandable_segments:False"),
    ],
)
def test_user_value_kept(name, value):
    seen = _after_import("import foldforge", **{name: value})
    expected = dict.fromkeys(VARIABLES)
    expected[name] = value
    assert seen == expected
