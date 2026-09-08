"""The ABI gate must catch a breaking change and let an additive one through.

Tested against the judging logic directly, not by editing the real header --
same reasoning as test_regression_gate.py: fast, and still covers what
matters.
"""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "abi_gate", Path(__file__).resolve().parent.parent / "tools" / "abi_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def snapshot(**overrides):
    base = {
        "version": {"MAJOR": 0, "MINOR": 1, "PATCH": 0},
        "functions": {
            "cv_insert": "int cv_insert(cv_index *index, int64_t id)",
            "cv_delete": "int cv_delete(cv_index *index, int64_t id)",
        },
        "structs": {
            "cv_stats": [["uint64_t", "clock"], ["uint64_t", "pages"]],
        },
        "enum_values": {"CV_COSINE": 0, "CV_L2": 1},
    }
    base.update(overrides)
    return base


def test_no_change_passes(capsys):
    assert gate.judge(snapshot(), snapshot()) == 0
    assert "No ABI change" in capsys.readouterr().out


def test_removed_function_without_major_bump_fails(capsys):
    current = snapshot(functions={"cv_insert": snapshot()["functions"]["cv_insert"]})
    assert gate.judge(current, snapshot()) == 1
    assert "function removed: cv_delete" in capsys.readouterr().out


def test_changed_signature_without_major_bump_fails():
    current = snapshot(
        functions={
            "cv_insert": "int cv_insert(cv_index *index, int32_t id)",  # narrowed
            "cv_delete": snapshot()["functions"]["cv_delete"],
        }
    )
    assert gate.judge(current, snapshot()) == 1


def test_struct_layout_change_without_major_bump_fails(capsys):
    current = snapshot(structs={"cv_stats": [["uint32_t", "clock"], ["uint64_t", "pages"]]})
    assert gate.judge(current, snapshot()) == 1
    assert "struct layout changed: cv_stats" in capsys.readouterr().out


def test_enum_value_change_without_major_bump_fails(capsys):
    current = snapshot(enum_values={"CV_COSINE": 1, "CV_L2": 0})  # swapped
    assert gate.judge(current, snapshot()) == 1
    assert "enum value changed" in capsys.readouterr().out


def test_breaking_change_with_major_bump_passes(capsys):
    current = snapshot(
        version={"MAJOR": 1, "MINOR": 0, "PATCH": 0},
        functions={"cv_insert": snapshot()["functions"]["cv_insert"]},
    )
    assert gate.judge(current, snapshot()) == 0
    assert "acknowledged" in capsys.readouterr().out


def test_additive_function_passes_without_major_bump(capsys):
    fns = dict(snapshot()["functions"])
    fns["cv_new_thing"] = "int cv_new_thing(cv_index *index)"
    current = snapshot(functions=fns)
    assert gate.judge(current, snapshot()) == 0
    assert "function added: cv_new_thing" in capsys.readouterr().out


def test_additive_without_minor_bump_notes_but_does_not_fail(capsys):
    fns = dict(snapshot()["functions"])
    fns["cv_new_thing"] = "int cv_new_thing(cv_index *index)"
    current = snapshot(functions=fns)  # version unchanged from baseline
    code = gate.judge(current, snapshot())
    assert code == 0
    assert "not bumped" in capsys.readouterr().out


def test_parse_header_ignores_comments():
    text = """
    /* CV_API void this_is_in_a_comment(void); */
    // CV_API void also_a_comment(void);
    CV_API int cv_real(cv_index *index);
    """
    parsed = gate.parse_header(text)
    assert "cv_real" in parsed["functions"]
    assert "this_is_in_a_comment" not in parsed["functions"]
    assert "also_a_comment" not in parsed["functions"]


def test_parse_header_reads_current_header_without_error():
    # The real file must parse cleanly and find the known, stable functions.
    text = gate.HEADER.read_text()
    parsed = gate.parse_header(text)
    assert "cv_insert" in parsed["functions"]
    assert "cv_search" in parsed["functions"]
    assert "cv_stats" in parsed["structs"]
    assert parsed["version"]["MAJOR"] is not None


def test_committed_baseline_matches_current_header():
    # Guards against a header edit that was committed without --update.
    import json

    if not gate.BASELINE.exists():
        return
    baseline = json.loads(gate.BASELINE.read_text())
    current = gate.parse_header(gate.HEADER.read_text())
    breaking, additive = gate.diff(current, baseline)
    assert not breaking and not additive, (
        "bindings/rust/chronovec-sys/native/include/chronovec.h changed without updating "
        "native/abi_baseline.json -- run `python tools/abi_gate.py --update`"
    )
