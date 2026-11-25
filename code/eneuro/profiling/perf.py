"""Collect integer counters from Linux perf stat."""

import re
import shutil
import subprocess


def perf_stat(command, events):
    """Run a command with perf stat and return counted events and its stdout."""
    perf = shutil.which("perf")
    if perf is None:
        raise FileNotFoundError("perf not found in PATH")

    argv = [perf, "stat", "-x,", "-e", ",".join(events), *command]
    result = subprocess.run(argv, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "perf stat failed")

    counters = {}
    for line in result.stderr.splitlines():
        fields = line.split(",", maxsplit=3)
        if len(fields) >= 3:
            count, _, event = (field.strip() for field in fields[:3])
            if re.fullmatch(r"-?\d+", count):
                counters[event] = int(count)
    return counters, result.stdout
