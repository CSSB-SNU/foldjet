"""The pre-rename FOLDFORGE_* environment names map to FOLDJET_*."""

from foldjet import _envcompat


def test_old_names_fill_unset_new_names():
    env = {"FOLDFORGE_HOME": "/a", "FOLDFORGE_RUNS": "/r", "FOLDJET_RUNS": "/new"}
    assert sorted(_envcompat.sync(env)) == ["FOLDFORGE_HOME"]
    assert env["FOLDJET_HOME"] == "/a"
    assert env["FOLDJET_RUNS"] == "/new"
