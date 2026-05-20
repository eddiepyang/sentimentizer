## Brief overview
Rule for mandatory plan review before implementation. Applies to any task that requires architectural decisions, multi-file changes, or dependency migrations.

## Plan-then-review workflow
- After writing the initial plan, always perform an explicit review pass before presenting it or starting implementation
- The review pass must check for: risks, gaps, edge cases, and tradeoffs
- After identifying issues, revise the plan and review again (at least one revision cycle)
- Do not skip the review step even when the plan seems straightforward — surface-level plans often hide dependency conflicts, backward-compatibility breaks, or test gaps

## Review checklist
- **Risks**: What can break? Are there implicit assumptions? What happens if a dependency version doesn't resolve?
- **Gaps**: Are there files or import paths that were missed? Are there callers that reference the old API that weren't updated?
- **Tradeoffs**: What is being sacrificed? Is there a simpler alternative? Is backward compatibility maintained or intentionally broken?
- **Edge cases**: What happens with empty inputs, single-item classes, missing files, or pre-existing data in old formats?
- **Test coverage**: Do existing tests cover the changed behavior? Are new tests needed for the new code paths?

## Multiple review cycles
- If the first review reveals significant issues (e.g., dependency conflicts, broken backward compatibility), revise the plan and run a second review
- Stop reviewing when the plan has no unresolved risks or gaps that could cause implementation failure