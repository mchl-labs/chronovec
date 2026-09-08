"""Every CLI mode, end to end.

The CLI had no coverage at all, and it is where a whole mode went missing:
`--mode mixed` was documented and listed in the reproduction instructions while
the argument parser did not accept it. Nothing in the library tests could see
that, because the trace builder it called existed and worked.

These run the real entry point against a small generated dataset, so a mode
that is advertised and not wired fails here.
"""

import json

import numpy as np
import pytest

pytest.importorskip("h5py")

from streambench.cli import main  # noqa: E402

MODES = ["bulk", "interleaved", "ops", "mixed", "concurrent", "snapshot", "equal-recall"]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    import h5py

    path = tmp_path_factory.mktemp("data") / "tiny.hdf5"
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(16, 8)).astype(np.float32) * 3.0
    train = (centres[rng.integers(0, 16, 4000)] + rng.normal(scale=1.0, size=(4000, 8))).astype(
        np.float32
    )
    test = (centres[rng.integers(0, 16, 40)] + rng.normal(scale=1.0, size=(40, 8))).astype(
        np.float32
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=train)
        handle.create_dataset("test", data=test)
    return str(path)


@pytest.mark.parametrize("mode", MODES)
def test_every_advertised_mode_runs_and_reports(mode, dataset, tmp_path, capsys):
    output = tmp_path / f"{mode}.json"
    main(
        [
            "--dataset",
            dataset,
            "--mode",
            mode,
            "--live",
            "800",
            "--batch",
            "100",
            "--epochs",
            "1",
            "--queries",
            "20",
            "--engines",
            "chronovec",
            "--output",
            str(output),
        ]
    )
    printed = capsys.readouterr().out
    assert "chronovec" in printed
    assert "skipped" not in printed, printed
    report = json.loads(output.read_text())
    assert report["engines"]["chronovec"]


def test_help_lists_every_mode_the_tests_cover(capsys):
    # Pins the parser to the list above, so a mode that is documented but not
    # wired -- which is exactly what happened to `mixed` -- fails here rather
    # than at a user's command line.
    with pytest.raises(SystemExit):
        main(["--help"])
    printed = capsys.readouterr().out
    for mode in MODES:
        assert mode in printed, f"--help does not offer {mode}"


def test_an_unknown_mode_is_rejected(dataset):
    with pytest.raises(SystemExit):
        main(["--dataset", dataset, "--mode", "nonsense"])


def test_list_engines_reports_availability(dataset, capsys):
    main(["--dataset", dataset, "--list-engines"])
    printed = capsys.readouterr().out
    assert "chronovec" in printed
    assert "available" in printed or "not installed" in printed
