# Contributing

Thanks for your interest in the Qwen Scope Lab Bench.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

You do **not** need a GPU or any credentials to develop or run the test suite. The bench ships a dev backend (a tiny in-memory CPU model) that exercises the real activation/steering/detection code paths:

```bash
python serve_web.py --dev      # http://127.0.0.1:7870
pytest                          # full suite, GPU-free
```

See `README.md` for the full tour and `docs/USER_GUIDE.md` for a click-along walkthrough.

## Conventions

- **Keep it testable GPU-free.** New analysis/steering/detection logic should run under the dev backend so it stays covered by CI. The model-touching part lives behind `SteeringService`; the pure logic operates on the activation dicts it returns.
- **Honest verdicts.** The bench scores every result against controls and reports an honest `validation_decision` (`VALIDATED` only if it beats the prompt baseline and all controls; otherwise `BENCHMARKED`). A clean negative is a finding — report it, don't bury it. Build the control in from the start.
- **Add tests.** Unit-test new logic on synthetic inputs; add an API round-trip where relevant. The existing suite must stay green.
- **Update docs.** Touch `README.md` and the relevant doc under `docs/` when you add a capability or endpoint.

## Pull requests

1. Fork and branch from `main`.
2. Make your change with tests and docs.
3. Ensure `pytest` is green.
4. Open a PR describing the change and its motivation. Include any honest caveats (small samples, untested on real model, etc.).

By contributing, you agree that your contributions are licensed under the project's [Apache License 2.0](LICENSE).
