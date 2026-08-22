## Brief overview
Rule for mandatory linting and testing during the implementation of plans or
features. This ensures code quality and prevents compounding errors as the
implementation progresses.

## Lint and Test Workflow
- **Continuous Validation**: Whenever you complete a logical section, milestone,
  or component of a larger implementation plan, you MUST run the project's
  verification suite before proceeding to the next section.
- **Do Not Defer**: Do not wait until the entire plan is implemented to run the
  checks. Catching errors early makes debugging significantly easier and
  prevents cascading failures.
- **Tools**: Use the project's checking commands (`make check` for the fast
  inner loop — format + lint only; `make ci` for the full gate — format + lint
  + pyright + pytest).
- **Fix Before Moving On**: If the linter or tests fail, you MUST fix the
  issues before beginning work on the next logical component or step in the
  plan.
- **Reporting**: When you communicate with the user that a step is completed,
  mention that you have successfully ran the checks and that the codebase
  remains in a healthy state.
