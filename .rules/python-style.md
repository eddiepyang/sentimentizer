## Brief overview

Python style for this repo, based on **PEP 8** (the canonical Python style
guide) and **PEP 257** (docstrings), reconciled with the project's tooling.
Mechanical formatting is enforced by `ruff` (`E/W/F/I/UP/B/SIM/ANN`) and
`ruff format`; this file covers the conventions tools don't fully decide. Run
`make ci` (format + lint + pyright + pytest) before done.

## Tooling is the source of truth

- **Line length is 100** (set in `pyproject.toml`, `[tool.ruff] line-length`),
  which overrides PEP 8's default of 79 — defer to the config, don't hand-wrap to 79.
- `target-version = "py312"`; `UP` (pyupgrade) keeps syntax modern — use `X | Y`
  unions, `list[str]` not `List[str]`, `dict`/`tuple` builtins as generics.
- Don't fight the formatter. If `ruff format` reflows your code, that's correct.
- **`make lint` does not check formatting.** It runs only `ruff check`. CI runs
  `ruff format --check .` as a separate gate. Use `make ci` for the full
  verification (format + lint + typecheck + test).
- **Do not run `black`.** The project formats with `ruff format`; running
  another formatter produces churn that CI rejects.

## Naming (PEP 8)

- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes and `Protocol`/type aliases of class-like things.
- `UPPER_SNAKE_CASE` for module-level constants (`LABEL_NAMES`, `NUM_CLASSES`).
- `_leading_underscore` for non-public module/class internals; a single leading
  underscore is the convention — avoid `__dunder` name-mangling unless needed.
- No single-char names except short-lived loop indices; no `l`/`O`/`I` as names.
- Don't prefix with type (`str_name`); don't restate the module
  (`config.config_model`).

## Layout & imports

- Imports grouped and ordered: **stdlib → third-party → first-party
  (`sentimentizer`)**, one blank line between groups; `ruff` (`I`/isort)
  enforces this. One import per line; no wildcard (`from x import *`) imports.
- Absolute imports for first-party code (`from sentimentizer.config import ...`),
  not relative, except short intra-package where it reads clearly.
- Two blank lines around top-level defs, one between methods.
- Module-level dunders (`__all__`) after the docstring, before imports.

## Type hints (required here)

- **Every function signature is fully annotated** (`ruff ANN`, `pyright`);
  `tests/**` are exempt. Annotate return types including `-> None`.
- `Any` is allowed only where genuinely needed (`ANN401` is off) — prefer a
  precise type or a `Protocol`. Use modern syntax (`X | None`, not
  `Optional[X]`).
- Prefer `Protocol` for structural seams (see [solid](solid.md)).
- Annotate class attributes too.

## Docstrings & comments (PEP 257)

- Public modules, classes, and functions get a docstring: one summary line in
  the imperative mood, a blank line, then detail if needed. Triple double-quotes.
- Comments explain **why**, not what the code already says. Keep them current —
  a stale comment is worse than none. Complete sentences, `# ` with one space.
- No commented-out code; delete it (git remembers).

## Idioms & correctness

- Truthiness: `if seq:` / `if not seq:` for emptiness; `is`/`is not` only for
  `None` and singletons. Compare to `None` with `is None`.
- Prefer comprehensions and `enumerate`/`zip` over index bookkeeping; `SIM`
  flags collapsible branches and redundant constructs — heed it.
- EAFP over LBYL where it reads cleaner (`try/except`), but **catch narrow
  exceptions** — never a bare `except:`; `except Exception` only at deliberate
  boundaries, and re-raise with `raise ... from err` to preserve context.
- Use context managers (`with`) for files/sockets/locks; don't leak resources.
- Default arguments are never mutable (`def f(x: list | None = None)` then
  `x = x or []`); `B` (bugbear) catches this.
- f-strings for interpolation, not `%`/`.format()`; no f-string without a
  placeholder.

## Functions & modules

- Small, single-purpose functions (see [solid](solid.md)); keep cyclomatic
  complexity low — extract rather than nest deeply. Return early to avoid
  arrow-shaped code.
- Keep public surface minimal; `_private` what callers don't need.
