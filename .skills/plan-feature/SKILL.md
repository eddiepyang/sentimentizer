---
name: plan-feature
description: Use ONLY when the user asks to plan a new feature, refactor, or architectural change. Guides a structured research-and-review cycle: discover codebase, identify risks, iterate on the plan, then produce a final implementation-ordered plan. Do NOT use for bug fixes, small edits, or single-file changes.
---

# Plan a Feature

You are planning a non-trivial feature, architectural change, or refactor. Follow this structured process to produce a reviewed, risk-adjusted implementation plan before writing any code.

## When to Use This Skill

- The user describes a multi-step change spanning multiple files or modules.
- The change involves new dependencies, optional feature flags, or cross-cutting concerns.
- The user explicitly asks to "plan" or "review" before implementing.

Do NOT use this for one-off edits, bug fixes, or changes confined to a single file.

## Phase 1: Discover

1. **Read the AGENTS.md / CONTRIBUTING.md** at the project root for conventions, naming patterns, and known pitfalls.
2. **Map the affected surface area**: Use `glob`, `grep`, and `read` to find every file that will be touched. Look for:
   - Import sites of any module you're adding or changing.
   - Config files, dataclasses, or YAML that define structural constants.
   - Test files that cover the affected area.
   - Optional dependency groups in `pyproject.toml` (or equivalent).
3. **Understand the existing pattern**: Read 2–3 representative implementations of whatever you're adding (e.g., if adding a new predictor class, read all existing predictor classes end-to-end).
4. **Check for precedent**: Search for similar features that were added recently (git log, recent file modifications). Mimic their structure.
5. **Verify third-party library capabilities**: If the plan depends on a library (e.g., mflux, mlx), check its actual API surface and supported models — don't assume features exist based on the library's marketing or category. Read the source or docs to confirm.

## Phase 2: Draft the Plan

Write a plan that includes:

### Required sections

1. **Goal**: One-sentence summary of what the feature does.
2. **Scope**: What's in and what's explicitly out of scope.
3. **Architecture**: How new code fits into existing structure. Show the file tree with NEW markers.
4. **File Changes**: Every file to create or modify, with a summary of the change per file.
5. **Optional Dependency Design** (if applicable):
   - Import guard pattern (try/except ImportError → AVAILABILITY_FLAG)
   - What happens when the dep is missing (graceful fallback? ImportError on explicit use?)
   - How the factory/registry resolves backends at runtime.
6. **Config Changes**: New fields, env var overrides, YAML entries. Show exact additions.
7. **Testing Strategy**: Which test files to add or modify. Mark tests that require the optional dep with `pytest.importorskip` or `@pytest.mark.skipif`.
8. **Convention Updates**: Any new entries needed in AGENTS.md, style guides, or similar docs.

### For each file change, specify:
- Exact import additions
- New classes/functions with signatures
- Modified functions with before/after signatures
- Config field additions with types and defaults

## Phase 3: Risk Review

Before finalizing, explicitly check for these common risks:

- **Import-time side effects**: Does the new optional dep get imported at module level? Guard with try/except.
- **Base class coupling**: If the new subclass overrides `__init__`, does it skip parent initialization that sets required state? Document which parent methods are called/skipped and why.
- **Dependency conflicts**: Does the new dep conflict with existing deps (e.g., same library, different versions)? Check `pyproject.toml` and the new package's dependencies.
- **Semantic mismatches**: Do config fields from the existing system map cleanly to the new backend, or do some fields become meaningless? Document which fields are ignored and what happens (warn? error?).
- **Test coverage gaps**: Are there code paths that only execute when the optional dep IS installed? These need separate test files or conditional skips.
- **Coexistence risk**: Can both backends run in the same process? Do they compete for GPU/memory? Log a warning if simultaneous use is risky.
- **API divergence**: Does the new library have the same API surface as the existing one? Document exact mismatches (e.g., seed handling, return types, unsupported features like negative prompts or reference images).
- **Registry completeness**: For every model/key in the system, is there a defined set of available backends? Is there a clear error if someone requests a backend that doesn't exist for that model?
- **Library capability verification**: Did you confirm the third-party library actually supports every model/feature you planned to use? List which models are supported and which are not. Never assume a library supports X because it's in the same category.

## Phase 4: Iterate

Present the plan to the user. Ask specifically:

1. "Does the scope match what you intended?"
2. "Are there risks I should dig deeper into?"
3. "Should we adjust the optional dependency boundary?"

Incorporate feedback and re-review. Do not proceed to implementation until the user approves.

## Phase 5: Implementation Order

After approval, produce a numbered implementation checklist ordered by dependency (things that must exist before other things can reference them). Each item should be a single commit-sized unit:

1. Import guard / compat module (no other code depends on it yet)
2. Config changes (new fields, registry, env vars)
3. New implementation classes (behind the import guard)
4. Factory/wiring (routes config → implementation)
5. Serve layer / integration points
6. Optional dep group in pyproject.toml
7. Tests (unit tests for factory, config; integration tests behind skipif)
8. Convention docs (AGENTS.md, etc.)

Each step should be independently testable (lint + existing tests pass after each step).

## Tips

- **Prefer the existing pattern**: If the codebase has an import-guard + factory pattern (like `TRANSFORMERS_AVAILABLE` + `new_trainer()`), copy it exactly. Consistency matters.
- **Immutable config dataclasses**: If the existing config uses `frozen=True`, your new fields must have defaults or use the same pattern for construction.
- **YAML config precedence**: New env vars need corresponding entries in both the `_ENV_OVERRIDES` dict and the YAML file. Don't forget the `_FIELD_TYPES` dict for non-string types.
- **Return type consistency**: If the existing interface returns `(PIL.Image, int)`, the new backend must return the same tuple type. Wrap the library's return type if it differs.
- **Seed handling divergence**: Many ML libraries use different seed mechanisms. If the base class uses `torch.Generator`, the new backend must override seed resolution to avoid importing `torch` unnecessarily.
- **Web-verify library claims**: Before planning around a library's supported models or features, check its actual source code, model directory, or API docs. Marketing READMEs often overstate coverage. The plan's risk review is the right place to flag "supported models" vs "assumed models."