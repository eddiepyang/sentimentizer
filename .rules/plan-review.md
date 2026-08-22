## Brief overview
Rule for mandatory plan review before implementation. Applies to any task that
requires architectural decisions, multi-file changes, or dependency migrations.

## Plan-then-review workflow
- After writing the initial plan, always perform an explicit review pass before
  presenting it or starting implementation
- The review pass must check for: risks, gaps, edge cases, and tradeoffs
- After identifying issues, revise the plan and review again (at least one
  revision cycle)
- Do not skip the review step even when the plan seems straightforward —
  surface-level plans often hide dependency conflicts, backward-compatibility
  breaks, or test gaps

## Review checklist
- **Risks**: What can break? Are there implicit assumptions? What happens if a
  dependency version doesn't resolve?
- **Gaps**: Are there files or import paths that were missed? Are there callers
  that reference the old API that weren't updated?
- **Tradeoffs**: What is being sacrificed? Is there a simpler alternative? Is
  backward compatibility maintained or intentionally broken?
- **Edge cases**: What happens with empty inputs, single-item classes, missing
  files, or pre-existing data in old formats?
- **Test coverage**: Do existing tests cover the changed behavior? Are new
  tests needed for the new code paths?

## Multiple review cycles
- If the first review reveals significant issues (e.g., dependency conflicts,
  broken backward compatibility), revise the plan and run a second review
- Stop reviewing when the plan has no unresolved risks or gaps that could cause
  implementation failure

## Plan lifecycle status (keep plans honest about what's built)
- Every plan doc in `docs/` must carry a **`**Status:**`** line directly under
  the title. Use one of: `📋 Not started`, `⏳ Partially implemented`,
  `✅ Completed`, and always include the date (`YYYY-MM` or full).
- When a plan (or a self-contained phase of one) is implemented, **update its
  Status in the same change that lands the code** — a merged plan and stale
  "Approved/Draft" status is a lie that causes re-implementation.
- Mark `✅ Completed` **only when the plan's in-scope work is actually
  shipped**, verified against the code (files/tests exist), not against intent.
  If phases remain, use `⏳ Partially implemented` and keep per-phase `STATUS:`
  markers accurate; do not stamp the whole plan complete.
- Preserve the prior status text (e.g. as an italic `_Original status: …_`
  line) rather than deleting it, so the plan's history stays legible.
