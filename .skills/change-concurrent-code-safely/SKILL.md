---
name: change-concurrent-code-safely
description: Use when implementing or reviewing Python changes involving shared mutable state, threading, Ray worker processes, Serve replica lifecycle, lazy resource construction, shutdown or cleanup, filesystem state shared by multiple objects, process-wide registries, metrics file writes, or training state coordination. Guides ownership and invariant analysis, process boundary reasoning, failure/teardown races, deterministic interleaving tests, and verification.
---

# Change Concurrent Code Safely

Prevent concurrency fixes from moving a race elsewhere. Make the protected
resource, ownership boundary, and unsafe interleaving explicit before editing,
then prove the behavior with deterministic tests.

Do not run `mdformat` on this or any `.skills/**/SKILL.md`; it corrupts the YAML
frontmatter.

## 1. Establish the contract

1. Inspect the branch and worktree. Preserve unrelated changes.
1. Read `AGENTS.md` and the applicable `.rules/` files. Treat the documented
   invariants as the current contract; verify them against the code before
   relying on them.
1. Identify every participant:
   - calling threads and worker threads (training driver, Ray Train workers);
   - Ray actor instances (Serve replicas, Tune trials);
   - object instances that can share the same backing resource (model
     checkpoints, metrics files, `JobStore`);
   - process boundaries (driver vs workers, exporter process vs training);
   - owners responsible for construction and shutdown.
1. State these four facts in working notes before changing code:
   - **Resource:** the state or lifecycle being protected.
   - **Invariant:** what must never become observably inconsistent.
   - **Sharing scope:** per replica, per worker, per model type, per process,
     or host-wide.
   - **Linearization point:** the operation that makes the change visible.

Do not key a lock by an incidental wrapper object when several wrappers can
reach the same protected resource. For example, multiple sentiment prediction
replicas loading the same checkpoint file need checkpoint-path-scoped
coordination, not one lock per replica.

## 2. Enumerate unsafe interleavings

Write a compact race table before choosing a primitive:

| Operation A | Operation B | Unsafe interleaving | Required outcome |
|---|---|---|---|
| model load | model load | two replicas race to load the same checkpoint | one loads, other gets cached |
| metric write | exporter read | exporter reads a torn write | reader gets a consistent snapshot |
| checkpoint save | training resume | resume reads a partially-written checkpoint | resume fails loudly or gets the last complete checkpoint |
| router lazy-load | classify call | concurrent first-call races router load | one call wins, other waits |
| Serve replica shutdown | in-flight request | request dropped mid-prediction | graceful drain before shutdown |
| driver checkpoint write | worker metric report | checkpoint and metric race for file lock | both land, or checkpoint wins |
| Ray session cleanup | training start | stale session files kill new run | cleanup runs before init |

Add task-specific rows. Include ordinary success, exception paths,
cancellation/timeouts where supported, cleanup, and repeated/idempotent
shutdown.

## 3. Choose the synchronization boundary

- Protect the invariant's full check-then-act or check-then-publish sequence.
  Locking only the individual reads and writes does not make the sequence
  atomic.
- Use the narrowest sharing scope that still covers every participant.
- Record lock ordering when more than one lock can be acquired. Avoid acquiring
  a process-wide registry lock while holding a resource lock or vice versa.
- Keep slow work outside critical sections unless the invariant requires it and
  the documented design accepts the serialization. Never hold a lock while
  sleeping or waiting on a network call.
- For lazy construction (router model, model checkpoint loading), check state
  before loading, publish under the state boundary, and check shutdown again
  afterward. Close any constructed object that loses the publication race.
- State whether coordination is thread-only, cross-process, or cross-machine.
  A Python lock cannot protect multiple Ray worker processes; use atomic
  filesystem operations, Ray's distributed state, or file locking when
  cross-process coordination is in scope.
- For metrics files in `/tmp/sentimentizer_metrics/`: writes are short and
  atomic (write-then-rename or single write call). If concurrent training runs
  of the same model type are possible, use the per-model-type file split; if
  two runs of the same type race, last-writer-wins is acceptable but should be
  documented.

## 4. Build deterministic race tests

Tests must force the interleaving, not hope the scheduler produces it.

- Use `threading.Event`, `Barrier`, monkeypatched pause points, and nonblocking
  lock acquisition. For Ray actors, use `.options(max_concurrency=1)` and
  controlled ordering of remote calls.
- Do not use `sleep()` as proof that another thread/process is blocked. A
  delayed thread can make broken code pass.
- Pause after the first operation has observed the state but before it mutates
  or publishes. Start the competing operation at that boundary.
- Test separate Ray actor handles or Serve replicas when the backing resource
  is shared (a checkpoint path, a metrics file, a `JobStore`).
- Capture exceptions from worker threads, release pause events in cleanup
  paths, join every thread with a timeout, and assert that none remain alive.
  For Ray, use `.shutdown()` in cleanup and assert actors reach their final
  state.
- Cover success, owner failure, constructor failure, cleanup failure, shutdown
  during construction, and repeated shutdown when those paths exist.
- Run a component test file alone when import order or global state could hide
  the bug.

Repeating a deterministic test is a useful supplement, but repetition never
substitutes for controlling the interleaving.

## 5. Review the implementation

Before declaring the change complete, check:

- Does every access to the protected invariant use the same boundary?
- Can two instances (replicas, workers, threads) reach the same resource with
  different locks?
- Can an early return or exception skip publication, notification, or cleanup?
- Can shutdown race construction, enqueueing, replacement, or retry?
- Is any lock held across GPU work, filesystem traversal, network/LLM calls,
  or `Future.result()`? If so, is that serialization required and documented?
- Does corruption or partial state become a safe miss/failure, or can it
  poison every later request?
- Do comments describe the actual lock scope and ownership?

## 6. Verify and report

For Python changes, run `make ci`. Run the focused deterministic concurrency
tests separately so a broader environment failure cannot hide their result.

Report:

- the invariant and sharing scope;
- the unsafe interleaving covered by each new test;
- focused and full-suite results;
- any environment-dependent failure without bypassing or hiding it.
