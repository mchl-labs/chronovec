"""The examples must run. A README demo that has drifted is worse than none."""

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
# lats_benchmark.py is covered separately below: it takes argparse CLI flags
# (this suite's own pytest argv would otherwise be parsed as its arguments)
# and its default workload is a deliberately slow sweep, not a quick demo.
_SEPARATELY_COVERED = {"lats_benchmark.py"}
EXAMPLES = sorted(
    path
    for path in EXAMPLES_DIR.glob("*.py")
    # A leading underscore marks shared support code for other examples
    # (e.g. _lats_engine.py), not a standalone runnable demo of its own.
    if not path.name.startswith("_") and path.name not in _SEPARATELY_COVERED
)


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_runs_without_error(script, capsys):
    runpy.run_path(str(script), run_name="__main__")
    printed = capsys.readouterr().out
    assert printed.strip(), f"{script.name} produced no output"


def test_agent_memory_example_demonstrates_rewind_and_retry():
    # Guards the substance, not just the exit code: the retry must stay
    # isolated until merge, and the pre-interaction snapshot must survive it.
    script = Path(__file__).resolve().parent.parent / "examples" / "agent_memory.py"
    namespace = runpy.run_path(str(script))
    AgentMemory, embed, width = (namespace["AgentMemory"], namespace["embed"], namespace["WIDTH"])
    memory = AgentMemory(dimensions=width, metric="cosine", nprobe=64)
    memory.add("x", embed("the user prefers dark mode interfaces"), text="dark")
    before = memory.snapshot()
    memory.add("x", embed("the user prefers light mode interfaces"), text="light mistake")
    retry = memory.branch("retry", snapshot=before)
    retry.add("x", embed("the user prefers dark mode interfaces"), text="dark retry")
    assert memory.search(embed("mode preference"), k=1)[0][1].payload["text"] == "light mistake"
    assert retry.search(embed("mode preference"), k=1)[0][1].payload["text"] == "dark retry"
    retry.merge()
    now = memory.search(embed("mode preference"), k=1)[0][1].payload["text"]
    then = memory.search(embed("mode preference"), k=1, as_of=before)[0][1].payload["text"]
    assert now == "dark retry" and then == "dark"
    memory.close()


def test_lats_benchmark_runs_with_a_small_workload():
    # A real subprocess, not runpy: the script's own argparse must see only
    # these flags, not pytest's argv, and must actually exit 0.
    script = EXAMPLES_DIR / "lats_benchmark.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--depth",
            "2",
            "--branching-factor",
            "3",
            "--iterations",
            "10",
            "--corpus-size",
            "3",
            "--concurrency",
            "1",
            "2",
            "--rollout-latency-ms",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "found the true best path" in result.stdout
    assert "delta-branch node creation is" in result.stdout
    assert "async concurrency sweep" in result.stdout
