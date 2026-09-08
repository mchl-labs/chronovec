# Code Intelligence: CodeGraph

This project is indexed by [CodeGraph](https://github.com/colbymchenry/codegraph) (see
`.claude/CLAUDE.md` for the auto-managed usage block, kept there rather than duplicated
here since CodeGraph owns and regenerates it). In short: reach for `codegraph_explore` (MCP)
or `codegraph explore "<query>"` (shell) before grep/find when you need to understand or
locate code: it returns verbatim source plus call paths and a blast-radius section in one
call. The index auto-syncs on file changes; there is no manual re-index step.

Migrated from GitNexus this session: GitNexus required a manual `analyze` re-index after
every commit and corrupted its own index three times in one session doing so (an internal
FTS-index bug, recovered each time via `clean` + rebuild). CodeGraph's auto-sync removes
that failure mode entirely, which is worth calling out since Step 8 of the old procedure
below no longer exists for exactly this reason.

---

# Pre-Commit Procedure (MANDATORY: follow every time)

Every change, no matter how small, goes through this checklist in order before any commit is made. Each step is non-optional unless explicitly noted.

## Step 0: Repo code conventions

Follow these while writing the change, not after:

- **Formatter/linter:** `ruff format` (double quotes, 100-col) and `ruff check` with `E, F, I, UP, B, SIM`. Don't fight the configured ignores in `pyproject.toml`; they're deliberate (e.g. `B008` because default-arg calls are idiomatic here).
- **C++:** `clang-format` with the project's `.clang-format`.
- **Docstring/comment voice:** match the existing modules, using a short module-level docstring stating *why the module exists and what tradeoff it encodes*, not what it contains. Inline comments explain a non-obvious constraint or the reason a workaround exists, not what the next line does. See `chronovec/collection.py` and `benchmarks/regression_gate.py` for the calibrated voice.
- **No speculative abstraction:** don't add config knobs, feature flags, or generality for a use case that doesn't exist yet. This codebase's history (`experiments/`, the "Rejected experiments" section of `CHANGELOG.md`) is full of things that were tried and reverted because they didn't earn their complexity; new code should default to the same bar.
- **Type annotations:** add them; `mypy` is non-blocking today but annotations are expected on new public surface.
- **`CHANGELOG.md`:** add an entry under `## Upcoming` in the matching section (`Performance`, `Compression`, `Architecture`, `Integrations & APIs`, `Testing & Quality`) for anything a user or contributor would want to know about: new capability, behavior change, or fixed inconsistency. Skip only for pure internal refactors with no observable effect.

## Step 1: Before editing, impact analysis

Run `codegraph_explore` (MCP) or `codegraph explore "<symbolName>"` (shell) for **every**
symbol you intend to modify. Its output includes a "Blast radius" section (callers,
callee-side effects, whether tests cover it). Report that to the user before editing. If a
symbol has many callers across modules, or no covering tests, treat that as the HIGH/CRITICAL
signal the old GitNexus-based wording called out explicitly, and confirm with the user before
proceeding.

## Step 2: After editing, correctness

```bash
python -m pytest tests/ -x -q --tb=short
```

All tests must pass. Fix failures before continuing. The `-x` flag stops on the first failure; do not suppress it.

## Step 3: After editing, style

```bash
python -m ruff check chronovec/ tests/
python -m ruff format --check chronovec/ tests/
```

Fix any lint or format errors (`ruff check --fix` / `ruff format`). Do not commit with lint failures.

## Step 4: After editing, performance gate

Run only when the change touches Python under `chronovec/`, C++ source, or any benchmark/test file. Skip for pure doc/SKILL/workflow edits.

```bash
python benchmarks/regression_gate.py
```

Exit code 0 = pass. Exit code 1 = a regression beyond tolerance was found: **do not commit; investigate and fix first**. The gate has machine-drift detection: if the runner is noisy it withholds timing verdicts (exit 0) but still fires on algorithmic regressions (recall, space amplification).

If a change intentionally changes a metric (e.g. a deliberate recall/speed trade), update the baseline:

```bash
python benchmarks/regression_gate.py --update
```

Then commit the updated `benchmarks/regression_baseline.json` alongside the code change.

### ABI gate: only when `bindings/rust/chronovec-sys/native/include/chronovec.h` changes

```bash
python tools/abi_gate.py
```

Every language binding (Python, Rust, Go, Node) links against this header. Exit code 1
means a breaking change (removed/changed function, struct field, or enum value) shipped
without bumping `CV_ABI_VERSION_MAJOR`; fix the bump or the change wasn't meant to be
breaking. On a deliberate, version-bumped break: `python tools/abi_gate.py --update` and
commit the updated `native/abi_baseline.json`.

## Step 5: After editing, library-wide DX alignment

A feature or API change is not done when it works on the class you touched. This library
exposes the same capability through several hand-maintained surfaces; none of them are
generated, so none of them update themselves. When you add or change something on one
surface, walk this list and port it to every other surface it applies to, for the best
consistent DX. "Not applicable" is a fine answer for a row; silence is not.

| Surface | Port the change when... | Where |
|---|---|---|
| `AsyncCollection` | You touched `Collection`. This wrapper mirrors nearly the entire `Collection` surface by hand via `asyncio.to_thread`, so a new method or constructor param on `Collection` needs the matching async method (or a documented reason it stays sync, like `snapshot()`/`count()`). | `chronovec/async_collection.py` |
| `AgentMemory` | The change is about ids, metadata, or persistence semantics shared with `Collection` (e.g. the `str \| int` id fix). Branching-specific features don't need to flow the other way. | `chronovec/memory.py` |
| LangChain / LlamaIndex adapters | The change adds a capability agent workflows would want surfaced through `VectorStore`/node-storage conventions (e.g. persistence, snapshot reads). Not every `Collection` method needs an adapter method; only what fits those frameworks' idioms. | `chronovec/integrations/langchain.py`, `chronovec/integrations/llamaindex.py`, `chronovec/integrations/core.py` |
| C ABI header | The change is in the native layer (C++) and needs a new/changed symbol exposed across languages. Bump `CV_ABI_VERSION_MAJOR`/`MINOR` per the ABI gate step above. | `bindings/rust/chronovec-sys/native/include/chronovec.h` |
| Raw-tier bindings (Rust `Index`, Go, Node) | The C ABI gained or changed a raw-tier operation (`insert`/`delete`/`search`/`vacuum`/`save`/`load`/`clock`/...). All three should offer the same operation under each language's own naming convention: Rust `snake_case`, Go `PascalCase`, JS `camelCase`, not just one of them. | `bindings/rust/chronovec/src/lib.rs`, `bindings/go/chronovec/chronovec.go`, `bindings/node/src/lib.rs` |
| Ergonomic `Collection` (Rust, Go, Node) | You touched Python's `Collection` in a way that's about ids, metadata, filtering, or snapshots (not something Python-specific like `embedding_function`). All four languages now have this tier, sharing the same vocabulary as Python (`add`/`query`/`snapshot`/`vacuum`), ported idiomatically per language: Node's and Go's `where=`/`Where` are a plain object/map matching Python's shape, Rust's is a `Filter` enum. `add()`/`Add()` carries forward the previous metadata/document when omitted (`None`/`nil`/`undefined`), matching Python; an explicit empty value (`Some(HashMap::new())`, `map[string]any{}`, `{}`) replaces instead. None of the three can explicitly clear an existing document back to "no document" while keeping metadata, matching a real limitation Python's own `Collection.add()` already has. | `bindings/rust/chronovec/src/collection.rs`, `bindings/go/chronovec/collection.go`, `bindings/node/src/collection.rs` |
| `__init__.py` | Any new public class or top-level function. Must be exported and added to `__all__`. | `chronovec/__init__.py` |
| API reference | Constructor signature, method signatures, return types, new methods. | `docs/api-reference.md` |
| Agent skill | Usage examples, footguns, persistence section, anything an agent using the skill would need to know to use the new surface correctly. | `.claude/skills/chronovec/chronovec/SKILL.md` |
| Getting-started | Only if the change affects the 3-minute tour path. | `docs/getting-started.md` |
| Examples | Run any affected example and confirm it still produces correct output. | `examples/` |
| `llms.txt` | A new top-level class or pattern was added. | `docs/public/llms.txt` |
| Binding integration docs | The change touches a raw-tier operation, or a language's own binding doc has drifted from its actual API (check while you're in there; `rust.md` had a stale constructor signature and a false ABI-stability claim found this way). | `docs/integrations/rust.md`, `docs/integrations/go.md`, `docs/integrations/node.md` |
| README integration table | A binding's scope changed (e.g. gained a new tier). | `README.md` |
| `CHANGELOG.md` | Already covered in Step 0; confirm it wasn't skipped. | `CHANGELOG.md` |

Do not leave a method documented with the wrong signature, mirrored on one wrapper but not
another, or missing an entry in the API reference. Wrong or inconsistent surfaces are worse
than an honest gap; they mislead or surprise rather than merely omit.

## Step 6: Pre-commit scope verification

CodeGraph doesn't have a git-diff-aware "what changed" tool the way GitNexus's
`detect_changes` did; its design philosophy is check-before-you-edit (Step 1's blast
radius), not diff-after-you're-done. Substitute: `git diff` (or `git diff --cached` once
staged) against what you intended to touch, and for any symbol whose blast radius from
Step 1 was non-trivial, re-run `codegraph explore "<symbolName>"` once more before
committing. The index is always current (auto-sync), so this reflects the real
post-edit state, not a stale snapshot. If unexpected files or symbols show up in the
diff, investigate before committing.

## Step 7: Commit

Stage only the files that belong to this change. Review `git status` after staging to catch accidental inclusions (env files, temp outputs, benchmark data not intended to update). Write a commit message that explains *why*, not just *what*.

No re-index step after this: CodeGraph auto-syncs on file changes (a `UserPromptSubmit`
hook plus an optional background daemon; see `codegraph status`), which is the entire
reason this project moved off GitNexus's manual `analyze` step.
