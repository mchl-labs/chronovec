"""The gate must both pass a clean tree and fail a regressed one.

A gate that never fires is worse than none, because it is trusted. These check
the judging logic against synthetic baselines rather than by rebuilding the
engine, so they run in milliseconds and still cover the cases that matter.
"""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "regression_gate", Path(__file__).resolve().parent.parent / "benchmarks" / "regression_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def baseline():
    return {
        "build_us_per_row": 1.5,
        "insert_single_us": 7.0,
        "delete_single_us": 0.9,
        "vacuum_s": 0.003,
        "qps_at_recall_0.90": 30000.0,
        "qps_at_recall_0.95": 20000.0,
        "qps_at_recall_0.99": 15000.0,
        "best_recall": 1.0,
        "amplification": 1.5,
        "bytes_per_live_vector": 970.0,
        "amplification_after_churn": 1.55,
        "recall_after_churn": 0.99,
    }


def test_an_unchanged_tree_passes(capsys):
    assert gate.judge(dict(baseline()), baseline()) == 0
    assert "No regression" in capsys.readouterr().out


def test_a_throughput_drop_beyond_tolerance_fails(capsys):
    current = baseline()
    current["qps_at_recall_0.95"] *= 0.80  # 20% slower
    assert gate.judge(current, baseline()) == 1
    assert "qps_at_recall_0.95" in capsys.readouterr().out


def test_a_drop_within_tolerance_passes():
    current = baseline()
    current["qps_at_recall_0.95"] *= 0.95  # 5%, inside the 12% band
    assert gate.judge(current, baseline()) == 0


def test_space_growth_fails_on_a_tighter_band():
    # Space is a property of the algorithm, not the machine, so it is judged
    # more strictly than throughput.
    current = baseline()
    current["amplification"] *= 1.07  # 7%, inside throughput bands
    assert gate.judge(current, baseline()) == 1


def test_recall_loss_fails():
    current = baseline()
    current["best_recall"] = 0.97  # 3% below
    assert gate.judge(current, baseline()) == 1


def test_an_improvement_never_fails():
    current = baseline()
    current["qps_at_recall_0.95"] *= 1.5
    current["amplification"] *= 0.5
    assert gate.judge(current, baseline()) == 0


def test_a_trade_is_reported_as_a_trade(capsys):
    current = baseline()
    current["qps_at_recall_0.95"] *= 0.80  # worse
    current["amplification"] *= 0.70  # better
    assert gate.judge(current, baseline()) == 1
    printed = capsys.readouterr().out
    assert "Improved at the same time" in printed
    assert "amplification" in printed
    assert "trade, not a failure" in printed


def test_a_metric_that_barely_worsened_is_not_called_an_improvement(capsys):
    # The bug this guards: testing only the sign counted a metric 1.9% worse
    # as having improved.
    current = baseline()
    current["qps_at_recall_0.95"] *= 0.80  # forces a failure report
    current["delete_single_us"] *= 1.019  # marginally worse
    current["amplification"] *= 0.70  # a genuine improvement
    gate.judge(current, baseline())
    printed = capsys.readouterr().out
    # The line is only printed when something did improve, so the genuine one
    # above guarantees it exists and the marginal regression must be absent
    # from it.
    assert "Improved at the same time:" in printed
    improved = printed.split("Improved at the same time:")[1].splitlines()[0]
    assert "amplification" in improved
    assert "delete_single_us" not in improved


def test_noisy_repeats_abstain_rather_than_accuse(capsys):
    current = baseline()
    current["qps_at_recall_0.95"] *= 0.80
    current["qps_at_recall_0.95__spread"] = 0.5  # repeats disagreed hugely
    assert gate.judge(current, baseline()) == 0
    assert "NOISY" in capsys.readouterr().out


def test_a_target_previously_reached_but_now_missing_fails(capsys):
    current = baseline()
    current["qps_at_recall_0.99"] = None
    assert gate.judge(current, baseline()) == 1
    assert "WORSE (not reached)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "key,direction,tolerance",
    [(key, direction, tolerance) for key, (tolerance, direction) in gate.TOLERANCE.items()],
)
def test_every_guarded_metric_fails_beyond_its_tolerance(key, direction, tolerance, capsys):
    """A listed metric is only a guard when its failure path is tested."""
    reference = {name: 100.0 for name in gate.TOLERANCE}
    current = dict(reference)
    multiplier = 1.0 - tolerance * 1.1 if direction == "higher" else 1.0 + tolerance * 1.1
    current[key] *= multiplier
    assert gate.judge(current, reference) == 1
    assert key in capsys.readouterr().out


def test_every_recorded_metric_has_a_tolerance():
    # A metric measured but not judged is a silent gap in the gate. Read from
    # the recorded baseline rather than a list repeated here: a hardcoded list
    # only catches the omission until someone updates the list instead of the
    # tolerances, which is the easier of the two to do by accident.
    import json

    recorded = {
        key
        for key in json.loads(gate.BASELINE.read_text())
        if not key.startswith("_") and not key.endswith("__spread") and key != "resident_mb"
    }
    missing = recorded - set(gate.TOLERANCE)
    assert not missing, f"measured but never judged: {sorted(missing)}"


def test_the_committed_baseline_is_readable_and_complete():
    import json

    data = json.loads(gate.BASELINE.read_text())
    for key in gate.TOLERANCE:
        assert key in data, f"baseline is missing {key}"
    assert data.get("_commit") and data.get("_machine")


def test_a_drifted_machine_withholds_timing_verdicts(capsys):
    """Sustained load must not produce a confident regression report.

    The repeat-spread rule cannot catch this: under steady load every repeat
    is slow together, so the spread stays small and the gate calls it WORSE.
    That happened twice in one day, each time reporting five regressions that
    were not there. The calibration is measured independently of anything
    under test, so when it moves the machine moved.
    """
    base = baseline()
    base["_calibration_ms"] = 100.0
    current = dict(base)
    current["_calibration_ms"] = 140.0
    for key in gate.TIMING_METRICS:
        if key in current and current[key]:
            direction = gate.TOLERANCE[key][1]
            current[key] = current[key] * 0.6 if direction == "higher" else current[key] * 1.4
    code = gate.judge(current, base)
    out = capsys.readouterr().out
    assert "untrusted" in out
    assert "calibration has moved" in out
    assert code == 0, "a drifted machine must not fail the gate on timings"


def test_a_drifted_machine_still_judges_deterministic_metrics(capsys):
    """Recall and space do not care how busy the machine is."""
    base = baseline()
    base["_calibration_ms"] = 100.0
    current = dict(base)
    current["_calibration_ms"] = 140.0
    current["recall_after_churn"] = base["recall_after_churn"] - 0.2
    code = gate.judge(current, base)
    out = capsys.readouterr().out
    assert "recall_after_churn" in out
    assert code != 0, "a recall drop must fail even on a busy machine"


def test_a_different_host_withholds_timing_verdicts_even_with_no_calibration_drift(capsys):
    """A cross-machine comparison must not report false regressions.

    Reproduces a real CI failure: a baseline recorded on a developer's Mac,
    judged against a GitHub Actions Linux runner, reported 40-580% "WORSE" on
    every timing metric with recall/fill/space all exactly unchanged. The
    generic calibration probe (plain arithmetic) read as "close enough"
    between the two machines even though the actual compiled engine's timing
    differed by an order of magnitude -- calibration drift alone cannot catch
    this, so the host identity itself must gate timing verdicts too.
    """
    base = baseline()
    base["_host_id"] = "aaaa1111"
    base["_calibration_ms"] = 100.0
    base["_machine"] = "Darwin"
    current = dict(base)
    current["_host_id"] = "bbbb2222"
    current["_calibration_ms"] = 100.0  # no drift on the generic probe
    current["_machine"] = "Linux"
    for key in gate.TIMING_METRICS:
        if key in current and current[key]:
            direction = gate.TOLERANCE[key][1]
            current[key] = current[key] * 0.1 if direction == "higher" else current[key] * 5.0
    code = gate.judge(current, base)
    out = capsys.readouterr().out
    assert "untrusted" in out
    assert "different machine" in out
    assert "WORSE" not in out
    assert code == 0, "a different host must not fail the gate on timings"


def test_a_legacy_baseline_without_host_id_still_detects_a_different_machine(capsys):
    """Before _host_id existed, _machine (OS family) is the only signal -- it
    must still withhold timing verdicts, not just print an FYI note."""
    base = baseline()
    base["_calibration_ms"] = 100.0
    base["_machine"] = "Darwin"
    # No _host_id at all, matching a baseline predating that field.
    current = dict(base)
    current["_calibration_ms"] = 100.0
    current["_machine"] = "Linux"
    for key in gate.TIMING_METRICS:
        if key in current and current[key]:
            direction = gate.TOLERANCE[key][1]
            current[key] = current[key] * 0.1 if direction == "higher" else current[key] * 5.0
    code = gate.judge(current, base)
    out = capsys.readouterr().out
    assert "untrusted" in out
    assert "different machine" in out
    assert code == 0, "a legacy cross-OS-family comparison must not fail the gate on timings"


def test_same_host_still_judges_timings_normally(capsys):
    """The new host check must not swallow real regressions on the same machine."""
    base = baseline()
    base["_host_id"] = "aaaa1111"
    base["_calibration_ms"] = 100.0
    current = dict(base)
    current["_host_id"] = "aaaa1111"
    current["_calibration_ms"] = 100.0
    current["insert_single_us"] = base["insert_single_us"] * 3.0
    code = gate.judge(current, base)
    out = capsys.readouterr().out
    assert "insert_single_us" in out
    assert "untrusted" not in out
    assert code != 0, "a same-host regression must still fail the gate"
