"""Per-process JSONL timing, kept separate to avoid concurrent file writes."""

import json
import time
from pathlib import Path


class Monitor:
    def __init__(self, path, role, rank=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.rank = rank
        # Each role owns its file; a repeated benchmark run must not count old steps.
        self.path.write_text('', encoding='utf-8')

    def log(self, event, start_ns, *, step=None, bytes_sent=0, bytes_received=0, **details):
        end_ns = time.perf_counter_ns()
        record = {'role': self.role, 'rank': self.rank, 'event': event, 'step': step,
                  'duration_ms': (end_ns - start_ns) / 1e6,
                  'start_perf_ns': start_ns, 'end_perf_ns': end_ns,
                  'bytes_sent': bytes_sent, 'bytes_received': bytes_received,
                  'timestamp_ns': time.time_ns(), **details}
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
