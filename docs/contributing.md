# Contributing to SpaceMining

Thanks for helping improve SpaceMining! This is the minimal workflow to contribute.

## Setup

```bash
git clone https://github.com/reveurmichael/space_mining.git
cd space_mining
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e '.[dev]'
```

## Develop

- Run tests: `pytest`
- Format: `black . && isort .`
- Type hints encouraged; keep code simple and readable.

## Commit & PR

1. Branch: `git checkout -b feat/short-title`
2. Commit small, focused edits (e.g., `fix: collision threshold`, `docs: quickstart tweak`).
3. Ensure tests pass: `pytest`
4. Push & open PR against `main` with a clear description and links to any related issues.

## Issues

- Use GitHub Issues for bugs and feature requests. Include steps to reproduce, expected vs actual behavior, and environment details.

That’s it—thank you for contributing!
