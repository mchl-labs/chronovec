# Contributing

Bug reports, reproducible benchmarks, and design discussions are welcome.

## Setting up the development environment

```bash
git clone https://github.com/mchl-labs/chronovec
cd chronovec
make dev           # builds native library + pip install -e ".[dev]"
make test          # runs the full test suite
make lint          # ruff check + ruff format --check
make fmt           # ruff format (auto-fix)
make typecheck     # mypy
```

## Running tests

```bash
# Full suite
pytest tests/ -v

# Specific module
pytest tests/test_mvcc_and_space.py -v

# Fast subset (no large builds)
pytest tests/ -v -k "not benchmark"

# With coverage
pytest tests/ --cov=chronovec --cov-report=term-missing
```

## Building the documentation site

The VitePress site has a committed npm lockfile, so validate it with the same
clean install used by docs CI:

```bash
npm ci
npm run docs:build
```

The build fails on dead links instead of silently publishing broken navigation.

## Code conventions

- **Formatter:** `ruff format` (configured in `pyproject.toml`)
- **Linter:** `ruff check` with `E, F, I, UP, B, SIM` rules
- **Type checker:** `mypy` (the primary package is checked on Python 3.11 in CI)
- **C++:** clang-format with the project's `.clang-format` style

Run `make fmt` before committing to auto-fix formatting issues.

## Test requirements

Any change that:

- Modifies a scoring path, index structure, or reclamation logic must pass the **equal-recall regression gate**: before and after QPS at equal recall, replacement throughput, and capacity amplification must not regress beyond the declared tolerance.
- Adds a new feature must include tests for that feature, including edge cases.
- Claims improved performance must include interleaved A/B benchmark pairs with Student-t 95% confidence intervals.

See [`docs/performance.md`](performance.md) for the benchmark methodology.

## C++ changes

Build with sanitizers before submitting:

```bash
cmake -B build-asan -DCHRONOVEC_SANITIZE_ADDRESS=ON
cmake --build build-asan -j
cd build-asan && ctest

cmake -B build-tsan -DCHRONOVEC_SANITIZE_THREAD=ON
cmake --build build-tsan -j
cd build-tsan && ctest
```

## Pull request checklist

- [ ] `make lint` passes
- [ ] `make test` passes
- [ ] New tests added for new behaviour
- [ ] Benchmark gate passes if performance-sensitive code changed
- [ ] `make abi-gate` passes if `bindings/rust/chronovec-sys/native/include/chronovec.h` changed, since every language binding (Python, Rust, Go, Node) links against this header, so a breaking change here must bump `CV_ABI_VERSION_MAJOR`
- [ ] Sanitizer targets pass (ASan/UBSan and TSan) if C++ was modified
- [ ] `CHANGELOG.md` updated under `## Upcoming`

## What makes a good contribution

- **Bug fixes** with a failing test that becomes passing.
- **New use-case examples** under `examples/` (must run standalone, no API keys).
- **Benchmark improvements** with reproducible numbers (interleaved A/B, CI reported).
- **Documentation improvements** with accurate code examples.
- **New language bindings** that wrap the stable C ABI.

## What is out of scope

- Changes that improve initial QPS but damage post-churn recall, replacement throughput, or capacity amplification.
- Features that require modifying the MVCC model in ways that break snapshot isolation.
- Contributions that copy code from incompatible sources.

## Releasing

`.github/workflows/publish.yml` publishes new package versions. It triggers only on push to
`main` (never other branches or PRs) and is idempotent: every run checks whether the
version declared in the manifest is already on the registry and skips the publish if so, so
pushing to `main` without a version bump is always a safe no-op.

| Package | Registry | Version source |
|---|---|---|
| `chronovec` | PyPI | `pyproject.toml`'s `[project].version` |
| `@chronovec/native` + 4 platform packages | npm | `bindings/node/package.json`'s `version` |
| `chronovec-sys`, `chronovec` | crates.io | each crate's `Cargo.toml` `version` |

To release: bump the version in the relevant manifest, merge to `main`. Nothing else:
the platform package versions and the root package's `optionalDependencies` are synced
automatically by `bindings/node/scripts/sync-versions.js` during the publish job, not by
hand.

**Required repository configuration** (one-time setup, not done by this repo's CI):

- **PyPI**: configure [trusted publishing](https://docs.pypi.org/trusted-publishers/) at
  `pypi.org/manage/project/chronovec/settings/publishing/`, pointing at this repo, the
  `publish.yml` workflow, and a `pypi` GitHub Actions environment. No stored token: the
  workflow authenticates via OIDC.
- **npm**: add an `NPM_TOKEN` repository secret (an npm automation token with publish
  access to the `@chronovec` scope).
- **crates.io**: add a `CARGO_REGISTRY_TOKEN` repository secret. The workflow publishes
  `chronovec-sys` first, waits for index propagation, then publishes `chronovec`.

**Python ships prebuilt wheels** for CPython 3.11-3.13 on Linux x86-64, macOS arm64,
and Windows x86-64, plus an sdist fallback for unsupported platforms. Unsupported
platforms still need a C++20 compiler and CMake because `pip` builds the native core
from source.

**Rust crates (`chronovec-sys`, `chronovec`) can be published to crates.io.**
`chronovec-sys`'s `build.rs` links against a prebuilt `libchronovec` via `CHRONOVEC_LIB_DIR`
rather than requiring a separately prebuilt library. `chronovec-sys` now bundles the C++
core and builds it with CMake during Cargo installation. Repository CI may still set
`CHRONOVEC_LIB_DIR` to exercise the shared library built by top-level CMake.

**Node ships prebuilt binaries for 4 platforms**, built on GitHub-hosted native runners
only (no cross-compilation of the C++ core): macOS arm64, Linux x64, Linux arm64, Windows
x64. **No Intel Mac (darwin-x64) build**: GitHub no longer offers a hosted Intel macOS
runner. Apple's toolchain can cross-compile from Apple Silicon, but this pipeline
deliberately doesn't attempt it yet; revisit if Intel Mac support turns out to matter.

## Reporting security issues

Do not report security vulnerabilities in public issues. See
[`SECURITY.md`](https://github.com/mchl-labs/chronovec/blob/main/SECURITY.md) for
the private reporting contact: [mchl-labs@outlook.com](mailto:mchl-labs@outlook.com).
