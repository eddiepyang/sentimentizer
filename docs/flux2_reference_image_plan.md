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

---

## Design choices

| Question | Decision | Rationale |
|---|---|---|
| Endpoint | Reuse `/v1/images/generate` and `/v1/images/jobs`, add optional field | One endpoint, callers just add a field. No new auth / rate-limit / idempotency wiring. |
| Transport | JSON + base64 list | Symmetric with response (`image_b64`). Avoids multipart machinery. |
| Field name | `reference_images: list[str] \| None` | "Image" is overloaded with the response; "reference" matches FLUX terminology. |
| Model gating | HTTP 400 if model ≠ `flux2_klein` | Schema is unified but semantics stay explicit. Gating happens **after** model name resolution (i.e., when `model=None` defaults to a non-Klein model, reference_images is still rejected). |
| Body size | Increase to 4 MiB for diffusion endpoints | Current `_RequestBodySizeLimitMiddleware` rejects >1 MiB with 413. A single 1024² WebP reference (~270 KB b64) plus prompt JSON fits in 1 MiB, but multi-reference payloads exceed it. Increase the limit for `/v1/images/` routes only; keep 1 MiB for all other routes. |
| Async jobs | Mirror the sync validation path | Same dispatcher helper used in both `generate_image` and `create_job`. |
| Multi-reference cap | 4 entries | Each 1024² PNG ≈ 1.5 MB base64; 4 only fits with WebP/JPEG under the 4 MiB body limit. |
| Per-image pixel cap (generation) | Reuse `cfg.max_pixels` (1 MP) | Generation output at 1024² = 1 MP. |
| Per-image pixel cap (reference) | `ref_max_pixels = 262_144` (512²) | Reference images are VAE-encoded and injected as extra attention tokens. Each 1024² reference adds ~1024 tokens, and attention memory is O(n²). On consumer GPUs (11 GB with NF4), a 1024² reference + 1024² generation OOMs. 512² references add only ~256 tokens and provide strong conditioning at 4× less activation memory. |
| Over-cap behavior | Reject with HTTP 400 | Predictable for callers; client-side downscale is trivial. Silent downscale would surprise users. |
| Empty list handling | Coerce `[]` → `None` silently | An empty list is semantically "no reference" — treat it the same as omitting the field. |
| Reference dims → output dims | Always require explicit `height`/`width` when `reference_images` is present | The pipeline auto-infers `height`/`width` from the reference image dimensions (pipeline line 780-781), which silently changes output size from the default 1024² to the reference's dimensions. Require explicit dimensions to avoid surprises. |

---

## Implementation steps

### 1. Request schema

[sentimentizer/serve/diffusion_models.py](../sentimentizer/serve/diffusion_models.py)
— extend `GenerateRequest`:

- `reference_images: list[str] | None = None`
- Validator:
  - When present, must contain 1 – 4 entries (empty list `[]` is coerced to `None` via `@field_validator`)
  - Each entry must be a non-empty string
- Docstring: "Base64-encoded reference images (raw base64 or
  `data:image/<fmt>;base64,…`). FLUX.2 Klein only. Up to 4 images,
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

### 3b. GPU cleanup after reference-image generation

Reference images create large intermediate tensors (VAE latents, attention
maps for the concatenated sequence). PyTorch's CUDA allocator doesn't
release these between requests. Add cleanup in
`Flux2KleinPredictor.generate()`:

```python
if reference_images is not None:
    gc.collect()
    torch.cuda.empty_cache()
```

Without this, the second request with reference images OOMs on consumer
GPUs. This is the same pattern used for diffusion inference in other
high-VRam-usage paths.

### 3c. Explicit height/width validation

When `reference_images` is provided, the dispatcher must validate that
both `height` and `width` are explicitly set in the request (i.e., the
caller didn't rely on defaults). The pipeline auto-infers output dimensions
from the reference image, which would silently change output size from
1024² to 512². The dispatcher should reject with HTTP 400 if
`reference_images` is present and either `height` or `width` is `None`
(or not present).

**Note**: Since `GenerateRequest` has `height: int = 1024` and `width: int = 1024`
with defaults, the "explicitly set" check needs a sentinel pattern. Use
`height: int | None = None` when `reference_images` is present, or validate
in a model validator that `height` and `width` are within reasonable bounds
(they already have `ge=256, le=2048` constraints).

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

Increase `MAX_REQUEST_BODY_BYTES` from 1 MiB to 4 MiB for routes under
`/v1/images/`. This is needed because a single 1024² WebP reference is
~270 KB base64, and multi-reference payloads (up to 4 images) can exceed
1 MiB. The existing `_RequestBodySizeLimitMiddleware` should be updated to
accept a per-route override, or the diffusion app can register its own
middleware with a higher limit.

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
- Unit (request schema):
  - `reference_images=[]` coerced to `None` (treated as omitted)
  - `reference_images=[5 entries]` rejected (≤ 4 cap)
  - `reference_images` with non-string entries → validation error
- Unit (predictor):
  - `Flux2KleinPredictor.generate(reference_images=[img])` passes `image=[img]` to pipeline
  - `Flux2KleinPredictor.generate(reference_images=None)` passes `image=None` to pipeline
  - `SDXLPredictor.generate(reference_images=[img])` raises `NotImplementedError`
  - `SD35Predictor.generate(reference_images=[img])` raises `NotImplementedError`
- Unit (VAE device):
  - After `warmup()` with NF4 quantization, `pipeline.vae` is on the correct device
- Integration:
  - Mock `Flux2KleinPredictor`, POST with reference images, assert
    `predictor.generate(...)` received `reference_images=[PIL.Image]`
  - POST reference images targeting `sd35` → HTTP 400 with
    `reference_images_unsupported`
  - POST reference images with `model=None` (defaults to non-Klein) → HTTP 400
  - POST malformed base64 → HTTP 400
  - POST reference images without explicit `height`/`width` → HTTP 400
  - Consecutive POST requests with reference images succeed without OOM
    (verify `gc.collect()` + `empty_cache()` runs between requests)
  - POST with `reference_images=[]` → treated same as omitted (no error)

### 8. Docs

- Module docstring in
  [sentimentizer/serve/app.py](../sentimentizer/serve/app.py): document
  the `reference_images` field on the generate endpoints, including the
  512×512 pixel cap and explicit `height`/`width` requirement
- Update any curl examples in the README / serving docs
- Add a "Reference images" section to `docs/serving.md` explaining:
  - FLUX.2 Klein only (other models return 400)
  - Max 4 reference images per request
  - Max 512×512 pixels per reference image (auto-rejected if exceeded)
  - Must specify `height` and `width` explicitly when using reference images
  - Body size limit is 4 MiB for `/v1/images/` routes (1 MiB for others)
  - Recommended: use WebP/JPEG format for reference images to stay under limits
  - VRAM implications: reference images increase attention memory quadratically

---

## Files touched

1. [sentimentizer/serve/diffusion_models.py](../sentimentizer/serve/diffusion_models.py)
   — `GenerateRequest.reference_images` field + validator + explicit height/width requirement
2. [sentimentizer/diffusion/predictor.py](../sentimentizer/diffusion/predictor.py)
   — `_decode_b64_image` helper, `_REF_MAX_PIXELS` constant, Klein `generate()` passthrough
   (always-list), VAE device fix in `warmup()`, GPU cleanup after reference generation,
   SDXL/SD35 `NotImplementedError` stubs
3. [sentimentizer/serve/diffusion_app.py](../sentimentizer/serve/diffusion_app.py)
   — dispatcher decode + forward + error gating in `generate_image` and `create_job`,
   body size limit increase for `/v1/images/` routes
4. `tests/` — new test cases (placement matches existing serve test files)
5. [sentimentizer/serve/app.py](../sentimentizer/serve/app.py) (optional)
   — module-level docstring update
6. [docs/serving.md](../docs/serving.md) — reference images documentation

---

## Sequence

1. Schema + decode helper + VAE device fix (independent, all testable in isolation)
2. Predictor signature changes + GPU cleanup (depends on schema, blocks dispatcher)
3. Dispatcher wiring + body size limit (depends on 1 + 2)
4. Tests (alongside each step)
5. Docs (last, once shape is final)