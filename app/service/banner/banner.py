import json
import logging
import textwrap
from typing import AsyncGenerator, List, Optional

import openai

from app.config import IMAGE_MODEL
from app.service.banner.banner_schema import GenerateData, PersonalInfo
from app.service.banner.banner_utilits import (
     client,
     save_b64_image,
)

log = logging.getLogger(__name__)


def sse(payload: dict) -> str:
     return f"data: {json.dumps(payload)}\n\n"

def sse_comment(msg: str = "keep-alive") -> str:
     return f": {msg}\n\n"


async def _stream_variant(
     variant_idx:    int,
     prompt:         str,
     size:           str,
     quality:        str,
     partial_images: int,
     ref_bytes_list: List[bytes],
     queue:          asyncio.Queue,
) -> None:
     """
     Generate one banner variant using gpt-image-1.5 streaming.

     Image delivery
     ──────────────
     Every image (partial preview + final) is:
          1. Decoded from base64
          2. Written to IMAGES_DIR as a PNG
          3. Returned in the SSE event as:
               url       → http://host/images/xxxx.png   ← use as <img src="">
               image_b64 → raw base64                    ← also available if needed

     Streaming events from gpt-image-1.5
     ─────────────────────────────────────
     event.type == "image_generation.partial_image"
          • event.partial_image_index  — 0-based frame index
          • event.b64_json             — PNG data
     The last event in the stream is the final completed image.
     """
     log.info("variant %d — start (refs=%d)", variant_idx, len(ref_bytes_list))
     try:
          # Build named BytesIO objects for reference images
          image_files = [
               make_image_file(raw, filename=f"ref_{i}.png")
               for i, raw in enumerate(ref_bytes_list)
          ]

          # ── Choose generate vs edit ────────────────────────────────────────────
          if ref_bytes_list:
               # images.edit() accepts a list of file-like objects directly.
               # No extra_body, no base64 string, no 'input' parameter.
               stream_coro = client.images.edit(
                    model=IMAGE_MODEL,
                    image=image_files,          # list[BytesIO] ✓
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    stream=True,
                    partial_images=partial_images,
               )
          else:
               stream_coro = client.images.generate(
                    model=IMAGE_MODEL,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    stream=True,
                    partial_images=partial_images,
               )

          final_b64:      Optional[str] = None
          revised_prompt: Optional[str] = None
          event_count = 0

          # ── Correct async streaming pattern for openai AsyncClient ────────────
          # `await stream_coro` resolves the coroutine → returns an AsyncStream.
          # Then iterate with `async for` directly — no `async with` needed.
          stream = await stream_coro
          async for event in stream:
               event_count += 1

               # ── Dump full event for diagnosis ─────────────────────────────────
               # Log every field on the event object so we can see real names.
               try:
                    if hasattr(event, "model_dump"):
                         event_data = event.model_dump()
                    elif hasattr(event, "__dict__"):
                         event_data = {k: v for k, v in event.__dict__.items()
                                   if not k.startswith("_") and not callable(v)}
                    else:
                         event_data = str(event)
               except Exception:
                    event_data = repr(event)

               event_type = getattr(event, "type", type(event).__name__)
               log.info("variant %d event[%d] type=%r", variant_idx, event_count, event_type)

               # ── Extract b64 — check every plausible field name ─────────────
               b64 = (
                    getattr(event, "b64_json",    None) or
                    getattr(event, "b64",         None) or
                    getattr(event, "image",       None) or
                    (event_data.get("b64_json")   if isinstance(event_data, dict) else None) or
                    (event_data.get("b64")        if isinstance(event_data, dict) else None) or
                    (event_data.get("image")      if isinstance(event_data, dict) else None)
               )
               # Some SDK versions nest it under .data[0].b64_json
               if not b64:
                    data_list = getattr(event, "data", None)
                    if data_list and isinstance(data_list, list) and data_list:
                         b64 = getattr(data_list[0], "b64_json", None)

               # ── Handle event types ────────────────────────────────────────────
               # images.edit     → "image_edit.partial_image"  / "image_edit.completed"
               # images.generate → "image_generation.partial_image" / "image_generation.completed"
               if event_type in ("image_edit.partial_image", "image_generation.partial_image"):
                    idx = getattr(event, "partial_image_index", 0)
                    if b64:
                         final_b64 = b64
                         _, url = save_b64_image(b64, prefix=f"v{variant_idx}_p{idx}")
                         log.info("variant %d — partial[%d] → %s", variant_idx, idx, url)
                         await queue.put({
                         "event":         "partial",
                         "variant":       variant_idx,
                         "partial_index": idx,
                         "url":           url,
                         # "image_b64":     b64,
                         })

               elif event_type in ("image_edit.completed", "image_generation.completed", "response.completed"):
                    revised_prompt = getattr(event, "revised_prompt", None)
                    if b64:
                         final_b64 = b64

               else:
                    if b64:
                         log.warning("variant %d — unknown event type %r, capturing b64", variant_idx, event_type)
                         final_b64 = b64

          log.info("variant %d — stream closed (%d events)", variant_idx, event_count)

          # ── Emit final ─────────────────────────────────────────────────────────
          if final_b64:
               _, final_url = save_b64_image(final_b64, prefix=f"v{variant_idx}_final")
               await queue.put({
                    "event":          "final",
                    "variant":        variant_idx,
                    "url":            final_url,     # ← use directly as <img src="">
                    # "image_b64":      final_b64,
                    # "revised_prompt": revised_prompt,
               })
               log.info("variant %d — done  url=%s", variant_idx, final_url)
          else:
               log.warning("variant %d — no image in %d events", variant_idx, event_count)
               await queue.put({
                    "event":   "error",
                    "variant": variant_idx,
                    "message": f"No image returned (received {event_count} stream events). "
                              f"Ensure IMAGE_MODEL={IMAGE_MODEL} supports streaming.",
               })

     except openai.BadRequestError as exc:
          log.error("variant %d — BadRequest: %s", variant_idx, exc)
          await queue.put({
               "event":   "error",
               "variant": variant_idx,
               "message": getattr(exc, "message", str(exc)),
          })
     except Exception as exc:
          log.exception("variant %d — unexpected error", variant_idx)
          await queue.put({"event": "error", "variant": variant_idx, "message": str(exc)})
     finally:
          await queue.put({"event": "_done", "variant": variant_idx})

SSE_HEADERS = {
     "Cache-Control":     "no-cache",
     "Connection":        "keep-alive",
     "X-Accel-Buffering": "no",
     "Content-Type":      "text/event-stream",
}

# ─── Prompt builder ───────────────────────────────────────────────────────────

_OCCASION_META: dict[str, dict] = {
     "birthday":       {"motifs":    "birthday cake with candles, balloons, confetti, streamers, gift boxes, stars",
                         "palette":   "bright cheerful colours, gold accents",
                         "atmosphere":"joyful, celebratory, warm"},
     "wedding":        {"motifs":    "floral arch, roses, peonies, white doves, rings, lace, romantic foliage",
                         "palette":   "ivory, blush pink, champagne gold, sage green",
                         "atmosphere":"romantic, luxurious, timeless"},
     "anniversary":    {"motifs":    "roses, champagne glasses, golden number, hearts, soft candles, starry night",
                         "palette":   "gold, deep red, pearl white",
                         "atmosphere":"romantic, nostalgic, elegant"},
     "baby_shower":    {"motifs":    "baby onesie, stork, baby animals, stars, clouds, pastel toys",
                         "palette":   "soft pastels — mint, lavender, blush, sky blue",
                         "atmosphere":"sweet, gentle, whimsical"},
     "graduation":     {"motifs":    "graduation cap and tassel, diploma scroll, laurel wreath, open book, confetti",
                         "palette":   "navy blue, gold, white",
                         "atmosphere":"proud, accomplished, bright future"},
     "party":          {"motifs":    "disco ball, music notes, cocktails, neon lights, dance floor, confetti",
                         "palette":   "vibrant, high-contrast, electric",
                         "atmosphere":"energetic, fun, exciting"},
     "corporate":      {"motifs":    "clean geometric shapes, subtle grid lines, brand-friendly abstract forms",
                         "palette":   "professional — navy, slate, silver, white",
                         "atmosphere":"polished, authoritative, modern"},
     "product_launch": {"motifs":    "spotlight beam, reveal curtain, starburst, abstract tech shapes, launch rocket",
                         "palette":   "bold contrast, brand colours, metallic sheen",
                         "atmosphere":"exciting, innovative, anticipatory"},
     "sports":         {"motifs":    "dynamic motion blur, stadium lights, sport-specific equipment, trophy",
                         "palette":   "bold team colours, high-energy",
                         "atmosphere":"powerful, intense, victorious"},
     "holiday":        {"motifs":    "seasonal decorations, snowflakes or sunshine, festive ornaments, wreaths",
                         "palette":   "red, green, gold (winter) or bright warm tones (summer)",
                         "atmosphere":"festive, warm, family-oriented"},
     "farewell":       {"motifs":    "open road, sunset horizon, paper planes, waving hands, flowers",
                         "palette":   "warm sunset tones, soft gold",
                         "atmosphere":"nostalgic, hopeful, bittersweet"},
     "welcome":        {"motifs":    "open door, sunrise, blooming flowers, outstretched hands, bright paths",
                         "palette":   "fresh greens, sky blue, warm yellow",
                         "atmosphere":"inviting, warm, optimistic"},
     "custom":         {"motifs":    "relevant objects and symbols for the specific event",
                         "palette":   "appropriate colours for the occasion",
                         "atmosphere":"fitting the mood of the event"},
}

_STYLE_DIRECTIVE: dict[str, str] = {
     "3d_illustration": (
          "Full 3-D rendered illustration style. Soft volumetric lighting, subsurface scattering on "
          "rounded objects, gentle drop-shadows, depth-of-field blur on background elements. "
          "Characters and props have a polished clay or smooth-plastic look (think Pixar / Blender render)."
     ),
     "pixel_art": (
          "Retro pixel-art style. Crisp, hard-edged pixels — no anti-aliasing or blur. "
          "Limited colour palette (16–64 colours). 8-bit or 16-bit game aesthetic."
     ),
     "minimalistic": (
          "Pure minimalist design. At most 2–3 colours. One dominant visual element, no clutter. "
          "Ultra-thin lines or simple geometric shapes only."
     ),
     "cartoon": (
          "Bold cartoon illustration style. Strong black outlines (2–4 px), flat or cel-shaded fills, "
          "exaggerated proportions, expressive faces. Bright, punchy colours. "
          "Think Saturday-morning cartoon or comic-strip aesthetic."
     ),
     "realistic": (
          "Photorealistic style. Studio-quality lighting, accurate shadows and reflections, "
          "hyper-detailed textures. Virtual 50 mm prime lens — shallow depth of field."
     ),
     "surreal": (
          "Dreamlike surrealist style. Impossible physics: objects float, melt, morph. "
          "Juxtapose unrelated items in unexpected scales. Hyper-detailed rendering."
     ),
     "2d": (
          "Classic 2-D hand-drawn animation style. Clean ink lines. "
          "Colour fills with minimal shading. Think 1990s Disney or Studio Ghibli."
     ),
     "flat_design": (
          "Modern flat design. Zero gradients, zero shadows, zero textures. "
          "Solid colour blocks only. Icons built from geometric primitives."
     ),
     "elegant":        "Refined and sophisticated. Smooth gradients, serif or script typography, generous whitespace.",
     "playful":        "Fun and whimsical. Rounded shapes, bright colours, hand-drawn feel, cheerful lettering.",
     "bold_modern":    "Strong geometric shapes, high contrast, bold sans-serif type, dynamic asymmetric composition.",
     "vintage_retro":  "Worn texture overlays, muted palette, retro badge shapes, distressed lettering.",
     "watercolor":     "Soft watercolour washes, organic bleed edges, painterly background, delicate brush strokes.",
     "neon_glow":      "Dark background, vibrant neon light effects, glowing colour halos, cyberpunk energy.",
     "rustic_natural": "Wood-grain texture, earthy tones, hand-lettered feel, botanical illustration accents.",
     "luxury_gold":    "Deep rich backgrounds (black/navy/emerald), lavish gold foil textures, premium typography.",
     "dark_dramatic":  "Dark moody palette, cinematic lighting, dramatic shadows, intense theatrical atmosphere.",
}

_VARIANT_ARCHETYPES = [
     {
          "name":       "Centred Hero",
          "layout":     "Perfectly symmetrical centre composition. Large hero graphic dominates the middle. "
                         "Headline sits directly below the hero in a clear, readable band.",
          "mood_tweak": "Warm and inviting colour temperature.",
     },
     {
          "name":       "Bold Left Split",
          "layout":     "Left–right split layout. Left half: bold solid colour block with headline and subtext. "
                         "Right half: main illustration or graphic element fills the space.",
          "mood_tweak": "Cooler, more modern tone with sharp contrasts.",
     },
     {
          "name":       "Diagonal Dynamic",
          "layout":     "Diagonal slash divides the banner into two zones. Decorative elements flow along "
                         "the diagonal for a sense of energy and motion. Typography on one clean side.",
          "mood_tweak": "Energetic and vibrant — slightly bolder saturation.",
     },
     {
          "name":       "Framed Elegant",
          "layout":     "Decorative border surrounds the entire canvas. "
                         "Central content area with balanced, well-spaced typography. "
                         "Corner ornaments echo the occasion theme.",
          "mood_tweak": "Soft and refined — slightly desaturated palette for sophistication.",
     },
]


def _personal_block(info: Optional[PersonalInfo]) -> str:
     if not info:
          return "No specific personal details provided."
     parts: list[str] = []
     if info.name:       parts.append(f"The banner is for {info.name}.")
     if info.age:        parts.append(f"They are turning {info.age} years old.")
     if info.profession: parts.append(f"Their profession is {info.profession}.")
     if info.hobbies:    parts.append("Hobbies: " + ", ".join(info.hobbies[:8]) + ". Subtly incorporate matching motifs.")
     if info.message:    parts.append(f'Include this message on the banner: "{info.message}".')
     return " ".join(parts) if parts else "No specific personal details provided."


def _text_block(data: GenerateData) -> str:
     parts: list[str] = []
     if data.headline: parts.append(f'Primary headline (render prominently): "{data.headline}".')
     if data.subtext:  parts.append(f'Secondary text (smaller, supporting): "{data.subtext}".')
     return " ".join(parts) if parts else "No specific text required — keep text minimal."


def build_variant_prompts(data: GenerateData, ref_count: int) -> list[str]:
     label = (
          data.custom_occasion
          if data.occasion == BannerOccasion.custom and data.custom_occasion
          else data.occasion.value.replace("_", " ")
     )
     meta      = _OCCASION_META.get(data.occasion.value, _OCCASION_META["custom"])
     style_dir = _STYLE_DIRECTIVE[data.style.value]
     personal  = _personal_block(data.personal_info)
     text      = _text_block(data)
     extra     = data.description or "None."
     roles     = data.reference_roles or []

     if ref_count and not roles:
          ref_note = f"{ref_count} reference image(s) provided. Use them to inform colour palette, style, and composition."
     elif roles:
          ref_note = "Reference images — roles: " + "; ".join(
               f"Image {i+1}: {r}" for i, r in enumerate(roles[:ref_count])
          ) + "."
     else:
          ref_note = "No reference images provided."

     prompts: list[str] = []
     for arch in _VARIANT_ARCHETYPES:
          prompts.append(textwrap.dedent(f"""
               Create a high-quality, print-ready {label} banner image.

               === OCCASION & ATMOSPHERE ===
               Event     : {label}
               Motifs    : {meta['motifs']}
               Palette   : {meta['palette']}
               Atmosphere: {meta['atmosphere']}

               === VISUAL STYLE — {data.style.value.upper().replace('_', ' ')} ===
               {style_dir}

               === LAYOUT VARIANT — {arch['name'].upper()} ===
               {arch['layout']}
               Mood for this variant: {arch['mood_tweak']}

               === TEXT TO RENDER ===
               {text}

               === PERSONAL DETAILS ===
               {personal}

               === EXTRA CREATIVE DIRECTION ===
               {extra}

               === REFERENCE IMAGE GUIDANCE ===
               {ref_note}

               === TECHNICAL REQUIREMENTS ===
               - Horizontal banner format, full-bleed background, no white margins.
               - All text must be legible, correctly spelled, and anti-aliased.
               - Clear visual hierarchy: headline is always the largest typographic element.
               - Professional quality suitable for both print and web use.
               - Do NOT include watermarks, stock-photo stamps, or lorem ipsum placeholder text.
          """).strip())

     log.info("Built 4 prompts | %s / %s | refs=%d", data.occasion.value, data.style.value, ref_count)
     return prompts

#-------------------------------------upload image HELPER-----------------------------------------------
async def upload_to_bytes(upload: UploadFile) -> bytes:
     ct = (upload.content_type or "").lower()
     if ct not in ALLOWED_IMAGE_TYPES:
          raise HTTPException(
               status_code=400,
               detail=f"Unsupported type '{ct}'. Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}",
          )
     chunks, total = [], 0
     while True:
          chunk = await upload.read(65_536)
          if not chunk:
               break
          total += len(chunk)
          if total > MAX_IMAGE_BYTES:
               raise HTTPException(status_code=413, detail=f"File exceeds {MAX_IMAGE_BYTES // (1024*1024)} MB.")
          chunks.append(chunk)
     raw = b"".join(chunks)
     log.info("Loaded '%s' (%d bytes)", upload.filename, len(raw))
     return raw


async def uploads_to_bytes_list(uploads: List[UploadFile]) -> List[bytes]:
     return list(await asyncio.gather(*(upload_to_bytes(u) for u in uploads)))


def make_image_file(raw: bytes, filename: str = "image.png") -> io.BytesIO:
     """Named BytesIO — what the OpenAI SDK image param expects."""
     buf = io.BytesIO(raw)
     buf.name = filename
     return buf