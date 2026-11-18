"""Small environment/dispatch probe for the CUDA C fast route.

The full numerical comparison is deliberately separate from this probe so it
can be run in an EnNeuro environment after the optional test dependencies are
installed.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", nargs="+", default=["cupy", "rawmodule"])
    ap.add_argument("--out", default="artifacts/cuda_stage1")
    args = ap.parse_args()
    report = {"schema_version": 1, "python": sys.version,
              "command": " ".join(sys.argv), "backends": {}}
    try:
        import cupy as cp
        report["cupy"] = cp.__version__
        report["device_count"] = int(cp.cuda.runtime.getDeviceCount())
        if report["device_count"]:
            report["compute_capability"] = cp.cuda.Device().compute_capability
    except Exception as exc:
        report["status"] = "no_cupy"
        report["error"] = repr(exc)
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from eneuro.base.cuda import dispatch
        for backend in args.backends:
            dispatch.set_backend(backend)
            report["backends"][backend] = {
                "available": bool(dispatch.is_available(backend)),
                "diagnostics": dispatch.diagnostics(reset=True),
            }
        report["status"] = "probe_complete"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cuda_lenet_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
