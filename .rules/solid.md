## Brief overview

SOLID — five design principles for keeping modules cohesive and loosely coupled.
Adapted to idiomatic Python (protocols and duck typing, not Java-style
interfaces). These govern how the model/training/serving pipeline and its
backends are structured.

## S — Single Responsibility

- A module/class has **one reason to change**. The pipeline already splits this
  way: `loader.py` loads data, `trainer.py` trains, `exporter.py` exports,
  `predictor.py` predicts, `serve/app.py` serves. Don't let a training change
  force a serving edit.
- No god-modules and no generic `utils.py`. If a helper has no obvious home, the
  abstraction is probably wrong.

## O — Open/Closed

- Open for extension, closed for modification. Adding a new model type must not
  require editing existing model classes — follow the five-step checklist in
  AGENTS.md ("Adding a new model type").
- New serving endpoints mount on the `create_fastapi_app()` factory; don't edit
  the app class.

## L — Liskov Substitution

- Any model subclass of `BaseSentimentModel` must be substitutable: same
  `forward()` shape `(B, num_classes)`, same `predict()` returning softmax
  probabilities, same `predict_text()` contract. A ModernBERT model and an RNN
  model are interchangeable from the caller's view.
- A subtype must not strengthen preconditions or weaken postconditions (e.g.
  don't return `(B, N)` where `num_classes != 3` when the contract promises 3).

## I — Interface Segregation

- Prefer narrow `Protocol`s over one fat interface. A model that only trains
  shouldn't have to implement ONNX export methods. `BaseSentimentModel` is the
  common interface; `HFTransformerModel` extends it for HF-backed models.
- Callers depend only on the methods they use.

## D — Dependency Inversion

- High-level pipeline code depends on **abstractions**, not concrete
  implementations. `Trainer` talks to `BaseSentimentModel`; the concrete model
  is injected via the factory (`new_trainer()`), never constructed deep inside
  training code.
- This is what keeps heavy/optional deps behind import guards — the abstraction
  lets the concrete impl be absent. See AGENTS.md's `SUPPORTS_ONNX` pattern.

## Python specifics

- Use `typing.Protocol` (structural typing) for seams — no abstract base class
  ceremony unless you need shared implementation. `pyright` checks conformance.
- Inject collaborators as constructor args or function parameters; avoid module
  globals and singletons for anything you'd want to swap in a test.
- Keep these in balance with [dry](dry.md): deduplicate genuine shared
  knowledge, but don't couple two responsibilities just because they share a
  few lines.
