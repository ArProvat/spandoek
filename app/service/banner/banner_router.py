import asyncio
import logging
from typing import Optional, AsyncGenerator, List
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from app.service.banner.banner_schema import GenerateData
from app.service.banner.banner import (
     build_variant_prompts,
     uploads_to_bytes_list,
     _stream_variant,
     sse,
     sse_comment,
     SSE_HEADERS,
)

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
     "/generate",
     summary="Generate 4 personalised banner variants",
     tags=["Generate"],
)
async def generate_banners(
     data: str = Form(
          ...,
          description='JSON banner parameters. E.g. {"occasion":"birthday","style":"cartoon","headline":"Happy 30th!"}',
     ),
     ref_image_1: Optional[UploadFile] = File(default=None, description="Reference image 1 (PNG/JPEG/WEBP, max 20 MB)."),
     ref_image_2: Optional[UploadFile] = File(default=None, description="Reference image 2."),
     ref_image_3: Optional[UploadFile] = File(default=None, description="Reference image 3."),
     ref_image_4: Optional[UploadFile] = File(default=None, description="Reference image 4."),
):
     """
     ## Generate 4 banner variants — streaming SSE

     ### SSE events

     | `event` | Key fields |
     |---------|-----------|
     | `files_processed` | `message` |
     | `prompts_ready` | `prompts[]` |
     | `partial` | `variant`, `partial_index`, **`url`**, `image_b64` |
     | `final` | `variant`, **`url`**, `image_b64`, `revised_prompt` |
     | `variant_done` | `variant` |
     | `all_done` | — |
     | `error` | `variant?`, `message` |
     """
     try:
          parsed = GenerateData.model_validate_json(data)
     except (ValidationError, ValueError) as exc:
          raise HTTPException(status_code=422, detail=f"Invalid `data` JSON: {exc}")

     valid_uploads = [
          f for f in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
          if f is not None and f.filename and f.filename.strip()
     ]

     async def stream() -> AsyncGenerator[str, None]:
          yield sse_comment("connected")

          ref_bytes_list: List[bytes] = []
          if valid_uploads:
               try:
                    ref_bytes_list = await uploads_to_bytes_list(valid_uploads)
                    yield sse({"event": "files_processed",
                              "message": f"{len(ref_bytes_list)} reference image(s) processed."})
               except HTTPException as exc:
                    yield sse({"event": "error", "message": exc.detail})
                    return

          try:
               prompts = build_variant_prompts(parsed, ref_count=len(ref_bytes_list))
          except Exception as exc:
               log.exception("Prompt build failed")
               yield sse({"event": "error", "message": f"Prompt build error: {exc}"})
               return

          #yield sse({"event": "prompts_ready", "prompts": prompts})

          queue: asyncio.Queue = asyncio.Queue()
          total, done_count = len(prompts), 0

          tasks = [
               asyncio.create_task(
                    _stream_variant(
                         variant_idx=i,
                         prompt=prompts[i],
                         size=parsed.size,
                         quality=parsed.quality,
                         partial_images=parsed.partial_images,
                         ref_bytes_list=ref_bytes_list,
                         queue=queue,
                    )
               )
               for i in range(total)
          ]

          while done_count < total:
               try:
                    evt = await asyncio.wait_for(queue.get(), timeout=300.0)
               except asyncio.TimeoutError:
                    yield sse({"event": "error", "message": "Generation timed out after 5 minutes."})
                    break

               if evt["event"] == "_done":
                    done_count += 1
                    yield sse({"event": "variant_done", "variant": evt["variant"]})
               else:
                    yield sse(evt)

          for task in tasks:
               task.cancel()

          yield sse({"event": "all_done"})
          log.info("All variants complete | %s / %s", parsed.occasion.value, parsed.style.value)

     return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


@router.post(
     "/regenerate",
     summary="Refine a selected banner",
     tags=["Regenerate"],
)
async def regenerate_banner(
     banner_image:   UploadFile = File(..., description="Banner to refine (PNG/JPEG/WEBP, max 20 MB)."),
     prompt:         str        = Form(..., min_length=3, max_length=2000,
                                        description="Edit instruction, e.g. 'Starry night background, keep all text'."),
     size:           str        = Form(default="1536x1024"),
     quality:        str        = Form(default="medium"),
     partial_images: int        = Form(default=2, ge=0, le=3),
):
     """
     ## Refine a selected banner — SSE stream

     SSE events: `file_processed` → `partial` (url) → `final` (url) → `done`

     ### cURL
     ```bash
     curl -N -X POST http://localhost:8800/regenerate \\
          -F 'banner_image=@selected.png' \\
          -F 'prompt=Change background to starry night sky, keep all text unchanged'
     ```
     """
     if size not in ("1024x1024", "1536x1024", "1024x1536"):
          raise HTTPException(status_code=422, detail="Invalid size.")
     if quality not in ("low", "medium", "high", "auto"):
          raise HTTPException(status_code=422, detail="Invalid quality.")

     async def stream() -> AsyncGenerator[str, None]:
          yield sse_comment("connected")

          try:
               raw = await upload_to_bytes(banner_image)
               yield sse({"event": "file_processed", "message": f"'{banner_image.filename}' loaded."})
          except HTTPException as exc:
               yield sse({"event": "error", "message": exc.detail})
               return

          try:
               image_file     = make_image_file(raw, filename=banner_image.filename or "banner.png")
               final_b64:      Optional[str] = None
               revised_prompt: Optional[str] = None

               # Correct pattern: await → AsyncStream → async for
               edit_stream = await client.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    stream=True,
                    partial_images=partial_images,
               )
               event_count = 0
               async for event in edit_stream:
                    event_count += 1
                    try:
                         event_data = event.model_dump() if hasattr(event, "model_dump") else \
                                   {k: v for k, v in event.__dict__.items()
                                   if not k.startswith("_") and not callable(v)}
                    except Exception:
                         event_data = {}

                    event_type = getattr(event, "type", type(event).__name__)
                    log.info("regenerate event[%d] type=%r keys=%s",
                              event_count, event_type,
                              list(event_data.keys()) if isinstance(event_data, dict) else "?")

                    b64 = (
                         getattr(event, "b64_json", None) or
                         getattr(event, "b64",      None) or
                         (event_data.get("b64_json") if isinstance(event_data, dict) else None)
                    )
                    if not b64:
                         data_list = getattr(event, "data", None)
                         if data_list and isinstance(data_list, list) and data_list:
                              b64 = getattr(data_list[0], "b64_json", None)

                    if event_type in ("image_edit.partial_image", "image_generation.partial_image"):
                         idx = getattr(event, "partial_image_index", 0)
                         if b64:
                              final_b64 = b64
                         _, url = save_b64_image(b64, prefix=f"regen_p{idx}")
                         log.info("regenerate — partial[%d] → %s", idx, url)
                         yield sse({"event": "partial", "partial_index": idx, "url": url, "image_b64": b64})
                    elif event_type in ("image_edit.completed", "image_generation.completed", "response.completed"):
                         revised_prompt = getattr(event, "revised_prompt", None)
                         if b64:
                              final_b64 = b64
                    else:
                         if b64:
                              log.warning("regenerate — unknown event type %r, capturing b64", event_type)
                              final_b64 = b64

               log.info("regenerate — stream closed (%d events)", event_count)

               if final_b64:
                    _, final_url = save_b64_image(final_b64, prefix="regen_final")
                    yield sse({
                         "event":          "final",
                         "url":            final_url,
                         #"image_b64":      final_b64,
                         #"revised_prompt": revised_prompt,
                    })
               else:
                    yield sse({"event": "error", "message": "Model returned no image."})

          except openai.BadRequestError as exc:
               yield sse({"event": "error", "message": getattr(exc, "message", str(exc))})
          except Exception as exc:
               log.exception("regenerate error")
               yield sse({"event": "error", "message": str(exc)})
          finally:
               yield sse({"event": "done"})

     return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)

@router.get("/options", tags=["Utils"], summary="Valid enum values for frontend dropdowns")
async def options():
     return {
          "occasions": [e.value for e in BannerOccasion],
          "styles": {
               "new":     ["3d_illustration", "pixel_art", "minimalistic", "cartoon",
                         "realistic", "surreal", "2d", "flat_design"],
               "classic": ["elegant", "playful", "bold_modern", "vintage_retro",
                         "watercolor", "neon_glow", "rustic_natural", "luxury_gold", "dark_dramatic"],
          },
          "sizes":    ["1024x1024", "1536x1024", "1024x1536"],
          "qualities": ["low", "medium", "high", "auto"],
          "layout_archetypes": [a["name"] for a in _VARIANT_ARCHETYPES],
          "max_reference_images": 4,
          "max_image_size_mb": MAX_IMAGE_BYTES // (1024 * 1024),
          "image_url_base": f"{BASE_URL}/images/",
          "image_ttl_hours": IMAGE_TTL_HOURS,
     }
