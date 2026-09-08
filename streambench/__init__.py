"""streambench: a benchmark for vector indexes whose data changes.

ann-benchmarks measures a static index -- build once, query forever. Every
mature system is tuned for it, which is why it cannot separate them on the axis
that matters once documents get edited, users exercise deletion rights, or an
agent writes memories continuously.

streambench measures the other axis: sustained turnover. One operation trace is
replayed against every engine, exact ground truth is recomputed against the
*current* live set each epoch, and the report covers write throughput, recall
drift, latency tails, space growth and reclamation.

A system passes by being boring. Flat recall, flat latency, flat space, across
many complete turnovers.

Two workloads are available. The replace trace holds the live set constant and
measures sustained turnover. The ops trace times insert, update, delete and
replace separately, because they are not the same operation and an engine can
be good at one and bad at another -- update in place especially, which most
engines have to emulate as delete plus insert.

It also records what an engine *cannot* do. An index with no delete has to
rebuild to express a turnover, and that shows up as a cost rather than as a
missing row.
"""

from .engines import (ENGINES, Engine, available_engines,
                       capability_matrix, format_capability_matrix)
from .runner import (run_batching_benchmark, run_benchmark,
                     run_concurrent_benchmark, run_dimension_sweep,
                     run_equal_recall_benchmark, run_filtered_benchmark,
                     run_ops_benchmark, run_snapshot_benchmark)
from .trace import (OpsTrace, Trace, build_mixed_trace, build_ops_trace,
                    build_trace)

__all__ = ["ENGINES", "Engine", "OpsTrace", "Trace", "available_engines",
           "capability_matrix", "format_capability_matrix",
           "build_mixed_trace", "build_ops_trace", "build_trace",
           "run_batching_benchmark",
           "run_benchmark", "run_concurrent_benchmark",
           "run_dimension_sweep", "run_equal_recall_benchmark",
           "run_filtered_benchmark",
           "run_ops_benchmark", "run_snapshot_benchmark"]
__version__ = "0.1.0"
