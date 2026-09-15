"""Backend API tests that do not require a CUDA device."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eneuro.base.cuda import dispatch


def test_backend_default_and_switch():
    old = dispatch.get_backend()
    dispatch.set_backend("cupy")
    assert dispatch.get_backend() == "cupy"
    dispatch.set_backend(old if old in {"cupy", "rawmodule", "auto", "extension"} else "cupy")


def test_invalid_backend():
    try:
        dispatch.set_backend("bad")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid backend did not raise")
