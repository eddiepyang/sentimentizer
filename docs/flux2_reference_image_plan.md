# FLUX.2 Klein — Reference Image Support Plan

Add reference image conditioning to `/v1/images/generate` (and the async
`/v1/images/jobs` path) for FLUX.2 Klein. The pipeline already accepts an
`image=` argument that VAE-encodes one or more reference images and
concatenates their tokens with the text tokens — this is true reference
conditioning, not init-noise img2img (no `strength` parameter).

Phase 1 covers FLUX.2 Klein only. SDXL / SD 3.5 reference support is a
separate change because those models require a different pipeline class
(`StableDiffusionXLImg2ImgPipeline` / `StableDiffusion3Img2ImgPipeline`)
with different semantics (init-noise + `strength`).

**Phase 2 prerequisite note**: SDXL img2img requires
`enable_model_cpu_offload()` for 11 GB VRAM and may need resolution ≤
768×768. It uses init-noise + `strength` semantics (not true reference
conditioning). The `StableDiffusionXLImg2ImgPipeline` does not have the
VAE device bug since it doesn't use bitsandbytes.

---

## Design choices

| Question | Decision | Rationale |
|---|---|---|
| Endpoint | Reuse `/v1/images/generate` and `/v1/images/jobs`, add optional field | One endpoint, callers just add a field. No new auth / rate-limit / idempotency wiring. |
| Transport | JSON + base64 list | Symmetric with response (`image_b64`). Avoids multipart machinery. |
| Field name | `reference_images: list[str] \| None` | "Image" is overloaded with the response; "reference" matches FLUX terminology. |
| Model gating | HTTP 400 if model ≠ `flux2_klein` | Schema is unified but semantics stay explicit. Gating happens **after** model name resolution (i.e., when `model=None` defaults to a non-Klein model, reference_images is still rejected). |
| Body size | Increase to 4 MiB for `/v1/images/` routes | Current `_RequestBodySizeLimitMiddleware` rejects >1 MiB with 413. A single 1024² WebP reference (~270 KB b64) plus prompt JSON fits in 1 MiB, but multi-reference payloads exceed it. Increase the limit for `/v1/images/` routes only; keep 1 MiB for all other routes. |
| Async jobs | Mirror the sync validation path | Same dispatcher helper used in both `generate_image` and `create_job`. |
| Multi-reference cap | 2 entries | With `ref_max_pixels = 512²`, each reference adds ~256 attention tokens. 2 references + 1024² generation = ~5120 total tokens (~4864 after text). 4 references would push to ~5632 tokens, which OOMs on 11 GB NF4 cards. Increase cap later if 24+ GB becomes the deployment baseline. |
| Per-image pixel cap (generation) | Reuse `cfg.max_pixels` (1 MP) | Generation output at 1024² = 1 MP. |
| Per-image pixel cap (reference) | `ref_max_pixels = 262_144` (512²) | Reference images are VAE-encoded and injected as extra attention tokens. Each 1024² reference adds ~1024 tokens, and attention memory is O(n²). On consumer GPUs (11 GB with NF4), a 1024² reference + 1024² generation OOMs. 512² references add only ~256 tokens and provide strong conditioning at 4× less activation memory. |
| Over-cap behavior | Reject with HTTP 400 | Predictable for callers; client-side downscale is trivial. Silent downscale would surprise users. |
| Empty list handling | Coerce `[]` → `None` silently | An empty list is semantically "no reference" — treat it the same as omitting the field. |
| Reference dims → output dims | Document that API always sends explicit `height`/`width` | `GenerateRequest` has `height: int = 1024` and `width: int = 1024` as Pydantic defaults — they are always present in a validated request, never `None`. The pipeline auto-infers only when `height is None`, which can't happen via the API. No validation code needed; just document the behavior. |
| Reference image preprocessing | Pipeline crops/resizes to reference's native dimensions | The pipeline preprocesses references with `resize_mode="crop"` to the reference's own `height`/`width` (line 778). References don't need to match generation resolution. The VAE encodes them at native resolution, and the resulting tokens are concatenated at that size. Callers should understand that non-square references may be cropped. |

---

## Implementation steps

### 1. Request schema

[sentimentizer/serve/diffusion_models.py](../sentimentizer/serve/diffusion_models.py)
— extend `GenerateRequest`:

- `reference_images: list[str] | None = None`
- Validator:
  - When present, must contain 1 – 2 entries (empty list `[]` is coerced to `None` via `@field_validator`)
  - Each entry must be a non-empty string
- Docstring: "Base64-encoded reference images (raw base64 or
  `data:image/<fmt>;base64,…`). FLUX.2 Klein only. Up to 2 images,
  each ≤ 512×512 after decoding (262,144 pixels)."

### 2. Decoding helper

New private util in
[sentimentizer/diffusion/predictor.py](../sentimentizer/diffusion/predictor.py)
alongside `_encode_pil`:

```python
_REF_MAX_PIXELS = 512 * 512  # 262,144 — reference images capped smaller than generation

def _decode_b64_image(b64: str, max_pixels: int = _REF_MAX_PIXELS) -> PIL.Image.Image:
    """Decode a base64 image string (raw or data URL) to PIL RGB.

    Raises ValueError on malformed input or images exceeding max_pixels.
    """
```

- Strips `data:image/...;base64,` prefix if present
- Decodes via `base64.b64decode`, opens with `PIL.Image.open`, converts to RGB
- Raises `ValueError` on malformed input (caught by the dispatcher and
  surfaced as HTTP 400)
- Validates decoded `width * height` against `max_pixels`
- **Strips alpha channel**: `Image.open(...).convert("RGB")` — RGBA/PNG with
  transparency is common for reference images; the pipeline expects 3-channel input

### 3. Predictor signature

- `DiffusionPredictor.generate()` (abstract): add
  `reference_images: list[PIL.Image.Image] | None = None`
- `Flux2KleinPredictor.generate()`: pass through to the pipeline as
  `image=` — always pass as a list (the pipeline wraps single images
  internally at line 760-761, so passing a list is consistent and avoids
  a special case).
- `SDXLPredictor.generate()` / `SD35Predictor.generate()`: accept the
  kwarg, raise `NotImplementedError` when non-empty. The dispatcher
  converts this to HTTP 400 (defense in depth — gating also happens at
  the dispatcher layer).

### 3a. VAE device fix for quantized models

When NF4/INT8 quantization is active, `bitsandbytes` auto-places the
transformer and text_encoder on GPU, but the VAE stays on CPU. The
reference-image code path calls `self.vae.encode()` which needs CUDA.
Fix `Flux2KleinPredictor.warmup()`:

```python
# After pipeline creation, when quantized:
if quant is not None:
    self._pipeline.vae.to(self._device)
```

This was debugged live — without it, `prepare_image_latents()` crashes with
`RuntimeError: Expected all tensors to be on the same device` because the
VAE encoder weights are on CPU while the input image tensor is on CUDA.

### 3b. GPU cleanup after every generation

Intermediate tensors (VAE latents, attention maps, scheduler state)
accumulate across requests. PyTorch's CUDA allocator doesn't release
these automatically. Add cleanup in `Flux2KleinPredictor.generate()`
after **every** generation, not just reference-image requests:

```python
gc.collect()
torch.cuda.empty_cache()
```

This was debugged live — consecutive requests OOM without cleanup, even
for text-only generation. The overhead is ~50 ms per call, negligible
compared to generation time.

### 4. Dispatcher wiring

[sentimentizer/serve/diffusion_app.py](../sentimentizer/serve/diffusion_app.py)
— in both `generate_image` and `create_job`:

- After `_get_handle(model_name)`: if `body.reference_images` is present,
  - If `model_name != "flux2_klein"`: return HTTP 400
    `{"code": "reference_images_unsupported"}`
  - Else: decode each entry via `_decode_b64_image(_REF_MAX_PIXELS)`, catching
    `ValueError` and returning HTTP 400 with the underlying message
- Append `reference_images=<list[PIL.Image]>` to the kwargs forwarded to
  `handle.generate.remote(...)`
- `PIL.Image` is cloudpickle-serializable, so Ray actor calls work
  without extra encoding.

### 4a. Body size limit increase

Increase the request body size limit from 1 MiB to 4 MiB for routes
under `/v1/images/`. The current `_RequestBodySizeLimitMiddleware` is
Starlette middleware that processes requests before routing, so it
cannot selectively apply limits by route without inspecting the path.

**Implementation**: Update `_RequestBodySizeLimitMiddleware` to accept a
`path_limits` mapping. When `request.url.path` matches a key in
`path_limits`, use that limit instead of the default. This follows the
existing middleware pattern and is the most targeted approach.

```python
# Example:
app.add_middleware(
    _RequestBodySizeLimitMiddleware,
    default_max_bytes=1 * 1024 * 1024,       # 1 MiB
    path_limits={"/v1/images/": 4 * 1024 * 1024},  # 4 MiB
)
```

Alternatives considered:
- **Global 4 MiB limit**: Simple but increases the limit for all routes.
- **Sub-app middleware**: The diffusion app registers its own body-size
  middleware at 4 MiB. Works but duplicates middleware logic.

### 4b. CUDA allocator configuration

Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the serve
process environment. This reduces VRAM fragmentation by allowing the
allocator to expand existing segments rather than creating new ones.
This is a defense-in-depth measure alongside the GPU cleanup steps.
The CUDA OOM error messages explicitly recommend this flag.

Add to the serve startup (alongside the existing `RAY_ENABLE_UV_RUN_RUNTIME_ENV`
setting) and document for notebook users.

### 5. Idempotency hash

No code change needed. `_body_hash` already serializes the full request
via `model_dump(mode="python")`, so identical reference payloads hash
identically. The hash computation includes the base64 strings (potentially
megabytes), but SHA-256 handles arbitrary input sizes — the overhead is
negligible (~5 ms for 4 MiB). Confirm with a test.

### 6. Logging

Add `reference_images_count` (an integer, not the content) to the
existing `logger.info("image generated", ...)` and
`logger.info("job created", ...)` calls. **Never log image content** —
base64 strings can be megabytes.

### 7. Tests

- Unit (decode helper):
  - Raw base64 input → valid PIL.Image
  - `data:image/png;base64,…` input → valid PIL.Image
  - RGBA/PNG with transparency → converted to RGB (no alpha channel)
  - Malformed base64 → `ValueError`
  - Decoded image exceeds `max_pixels` → `ValueError`
  - Decoded image at exactly `max_pixels` → accepted (boundary)
  - Non-square reference (e.g., 768×512 where total ≤ 262144) → accepted
- Unit (request schema):
  - `reference_images=[]` coerced to `None` (treated as omitted)
  - `reference_images=[3 entries]` rejected (≤ 2 cap)
  - `reference_images` with non-string entries → validation error
- Unit (predictor):
  - `Flux2KleinPredictor.generate(reference_images=[img])` passes `image=[img]` to pipeline
  - `Flux2KleinPredictor.generate(reference_images=None)` passes `image=None` to pipeline
  - `SDXLPredictor.generate(reference_images=[img])` raises `NotImplementedError`
  - `SD35Predictor.generate(reference_images=[img])` raises `NotImplementedError`
  - GPU cleanup (`gc.collect()` + `empty_cache()`) runs after every generation, not just with references
- Unit (VAE device):
  - After `warmup()` with NF4 quantization, `pipeline.vae` is on the correct device
- Unit (body size middleware):
  - Request to `/v1/images/generate` with 2 MiB body → accepted
  - Request to `/v1/sentiment/predict` with 2 MiB body → rejected with 413
- Integration:
  - Mock `Flux2KleinPredictor`, POST with reference images, assert
    `predictor.generate(...)` received `reference_images=[PIL.Image]`
  - POST reference images targeting `sd35` → HTTP 400 with
    `reference_images_unsupported`
  - POST reference images with `model=None` (defaults to non-Klein) → HTTP 400
  - POST malformed base64 → HTTP 400
  - POST with >2 reference images → HTTP 400
  - Consecutive POST requests with reference images succeed without OOM
    (verify `gc.collect()` + `empty_cache()` runs between requests)
  - POST with `reference_images=[]` → treated same as omitted (no error)

### 8. Docs

- Module docstring in
  [sentimentizer/serve/app.py](../sentimentizer/serve/app.py): document
  the `reference_images` field on the generate endpoints, including the
  512×512 pixel cap
- Update any curl examples in the README / serving docs
- Add a "Reference images" section to `docs/serving.md` explaining:
  - FLUX.2 Klein only (other models return 400)
  - Max 2 reference images per request
  - Max 512×512 pixels per reference image (auto-rejected if exceeded)
  - Reference images don't need to match generation resolution — the pipeline
    encodes them at native dimensions and concatenates tokens
  - Non-square references may be cropped by the pipeline's `resize_mode="crop"`
  - Body size limit is 4 MiB for `/v1/images/` routes (1 MiB for others)
  - Recommended: use WebP/JPEG format for reference images to stay under limits
  - VRAM implications: reference images increase attention memory quadratically
  - Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation

---

## Files touched

1. [sentimentizer/serve/diffusion_models.py](../sentimentizer/serve/diffusion_models.py)
   — `GenerateRequest.reference_images` field + validator
2. [sentimentizer/diffusion/predictor.py](../sentimentizer/diffusion/predictor.py)
   — `_decode_b64_image` helper, `_REF_MAX_PIXELS` constant, Klein `generate()` passthrough
   (always-list), VAE device fix in `warmup()`, GPU cleanup after every generation,
   SDXL/SD35 `NotImplementedError` stubs
3. [sentimentizer/serve/diffusion_app.py](../sentimentizer/serve/diffusion_app.py)
   — dispatcher decode + forward + error gating in `generate_image` and `create_job`
4. [sentimentizer/serve/middleware.py](../sentimentizer/serve/middleware.py)
   — `_RequestBodySizeLimitMiddleware` path-aware `path_limits` support
5. `tests/` — new test cases (placement matches existing serve test files)
6. [sentimentizer/serve/app.py](../sentimentizer/serve/app.py) (optional)
   — module-level docstring update
7. [docs/serving.md](../docs/serving.md) — reference images documentation

---

## Sequence

1. Schema + decode helper + VAE device fix + GPU cleanup (independent, all testable in isolation)
2. Predictor signature changes (depends on schema, blocks dispatcher)
3. Dispatcher wiring + body size limit + CUDA alloc config (depends on 1 + 2)
4. Tests (alongside each step)
5. Docs (last, once shape is final)