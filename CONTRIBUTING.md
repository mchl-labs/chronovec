# Contributing

Bug reports, reproducible benchmarks, and design discussions are welcome.

## Quick start

```bash
git clone https://github.com/mchl-labs/chronovec
cd chronovec
make dev    # builds native library + pip install -e ".[dev]"
make test   # run the test suite
make lint   # check formatting and lint
npm ci && npm run docs:build  # validate the published docs site
```

## Full contributing guide

See [docs/contributing.md](docs/contributing.md) for:
- Development environment setup
- Code conventions (ruff, mypy, clang-format)
- Test requirements and the equal-recall regression gate
- Pull request checklist
- C++ sanitizer targets

## Code of conduct

Be respectful. Discussions should focus on the technical merits of ideas, not on the people proposing them.
