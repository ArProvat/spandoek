"""
Banner Maker API  v5  — FastAPI + OpenAI gpt-image-1.5
=======================================================

Changes in v5
─────────────
  • Model     : gpt-image-1.5  (supports streaming + partial_images)
  • Images    : saved to IMAGES_DIR, served as static files at /images/{filename}
  • SSE events: carry `url` (http://host/images/xxx.png) instead of raw base64
                base64 still present in events as `image_b64` for convenience
  • Streaming : async with await client.images.generate/edit(..., stream=True)
  • Cleanup   : background task deletes images older than IMAGE_TTL_HOURS

Endpoints
─────────
  POST /generate      → 4 concurrent SSE banner streams
  POST /regenerate    → SSE stream of 1 edited banner
  GET  /images/{fn}  → Serve generated PNG files  (mounted StaticFiles)
  GET  /options       → Enum values for frontend dropdowns
  GET  /health        → Health check
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import textwrap
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional

import openai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("banner_api")

# ─── Config ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# gpt-image-1.5 is required for streaming + partial_images support.
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1.5")

# Base URL used when building image URLs returned in SSE events.
# Override with BASE_URL=https://yourdomain.com in production.
BASE_URL = os.getenv("BASE_URL", "http://localhost:8800").rstrip("/")

# Directory where generated PNG files are saved and served from.
IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "/tmp/banner_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# How many hours to keep generated images before auto-deleting.
IMAGE_TTL_HOURS = int(os.getenv("IMAGE_TTL_HOURS", "24"))

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB per file

# ─── Image storage helpers ────────────────────────────────────────────────────

def save_b64_image(b64: str, prefix: str = "img") -> tuple[str, str]:
    """
    Decode base64 PNG, write to IMAGES_DIR.
    Returns (filename, full_url).
    Frontend can use url directly as <img src="url">.
    """
    filename = f"{prefix}_{uuid.uuid4().hex}.png"
    filepath = IMAGES_DIR / filename
    filepath.write_bytes(base64.b64decode(b64))
    url = f"{BASE_URL}/images/{filename}"
    log.info("Saved  %s  →  %s", filepath.name, url)
    return filename, url


def cleanup_old_images() -> None:
    """Delete images older than IMAGE_TTL_HOURS hours."""
    cutoff  = time.time() - IMAGE_TTL_HOURS * 3600
    deleted = 0
    for f in IMAGES_DIR.glob("*.png"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
                deleted += 1
        except OSError:
            pass
    if deleted:
        log.info("Cleanup: removed %d old image(s)", deleted)


async def _periodic_cleanup(interval_seconds: int = 3600) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        cleanup_old_images()


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Banner Maker API",
    version="5.0.0",
    description=(
        "AI-powered banner generation with **gpt-image-1.5** streaming.\n\n"
        "SSE events carry a `url` field — put it straight into `<img src='...'>`.\n\n"
        "**Flow:** `POST /generate` → partial preview URLs stream in → final URL arrives → done."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve saved PNGs at /images/<filename>
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


@app.on_event("startup")
async def startup_event() -> None:
    cleanup_old_images()
    asyncio.create_task(_periodic_cleanup())
    log.info(
        "v5 started | model=%s | base_url=%s | images_dir=%s",
        IMAGE_MODEL, BASE_URL, IMAGES_DIR,
    )


# ─── Enums ────────────────────────────────────────────────────────────────────

class BannerOccasion(str, Enum):
    birthday       = "birthday"
    wedding        = "wedding"
    anniversary    = "anniversary"
    baby_shower    = "baby_shower"
    graduation     = "graduation"
    party          = "party"
    corporate      = "corporate"
    product_launch = "product_launch"
    sports         = "sports"
    holiday        = "holiday"
    farewell       = "farewell"
    welcome        = "welcome"
    custom         = "custom"


class VisualStyle(str, Enum):
    three_d_illustration = "3d_illustration"
    pixel_art            = "pixel_art"
    minimalistic         = "minimalistic"
    cartoon              = "cartoon"
    realistic            = "realistic"
    surreal              = "surreal"
    two_d                = "2d"
    flat_design          = "flat_design"
    elegant              = "elegant"
    playful              = "playful"
    bold_modern          = "bold_modern"
    vintage_retro        = "vintage_retro"
    watercolor           = "watercolor"
    neon_glow            = "neon_glow"
    rustic_natural       = "rustic_natural"
    luxury_gold          = "luxury_gold"
    dark_dramatic        = "dark_dramatic"


# ─── Pydantic models ──────────────────────────────────────────────────────────

class PersonalInfo(BaseModel):
    name:       Optional[str]       = Field(None, max_length=100)
    age:        Optional[int]       = Field(None, ge=1, le=150)
    hobbies:    Optional[List[str]] = None
    profession: Optional[str]       = Field(None, max_length=100)
    message:    Optional[str]       = Field(None, max_length=300)


class GenerateData(BaseModel):
    occasion:        BannerOccasion         = Field(...)
    style:           VisualStyle            = Field(VisualStyle.elegant)
    custom_occasion: Optional[str]          = Field(None, max_length=100)
    personal_info:   Optional[PersonalInfo] = None
    headline:        Optional[str]          = Field(None, max_length=120)
    subtext:         Optional[str]          = Field(None, max_length=200)
    description:     Optional[str]          = Field(None, max_length=1000)
    reference_roles: Optional[List[str]]    = None
    size:            Literal["1024x1024", "1536x1024", "1024x1536"] = "1536x1024"
    quality:         Literal["low", "medium", "high", "auto"]       = "medium"
    partial_images:  int                    = Field(2, ge=0, le=3)


# ─── Upload helpers ───────────────────────────────────────────────────────────

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


# ─── SSE helpers ─────────────────────────────────────────────────────────────

def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def sse_comment(msg: str = "keep-alive") -> str:
    return f": {msg}\n\n"

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


# ─── Core streaming worker ────────────────────────────────────────────────────

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


# ─── POST /generate ───────────────────────────────────────────────────────────

@app.post(
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

    **`url`** is a fully-qualified PNG URL. Use it directly:
    ```js
    if (evt.event === 'partial') img.src = evt.url;
    if (evt.event === 'final')   img.src = evt.url;
    ```

    ### cURL
    ```bash
    curl -N -X POST http://localhost:8800/generate \\
      -F 'data={"occasion":"birthday","style":"cartoon","headline":"Happy 30th!"}' \\
      -F 'ref_image_1=@photo.jpg'
    ```

    ### JavaScript
    ```js
    const form = new FormData();
    form.append('data', JSON.stringify({
      occasion: 'birthday', style: 'cartoon', headline: 'Happy 30th!'
    }));
    form.append('ref_image_1', fileInput.files[0]);  // optional

    const res = await fetch('/generate', { method: 'POST', body: form });
    const reader = res.body.getReader();
    const dec    = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const blocks = buf.split('\\n\\n');
      buf = blocks.pop();
      for (const block of blocks) {
        if (!block.startsWith('data:')) continue;
        const evt = JSON.parse(block.slice(5).trim());
        if (evt.event === 'partial') previewImgs[evt.variant].src = evt.url;
        if (evt.event === 'final')   finalImgs[evt.variant].src   = evt.url;
        if (evt.event === 'all_done') console.log('All 4 banners ready!');
      }
    }
    ```
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

        yield sse({"event": "prompts_ready", "prompts": prompts})

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


# ─── POST /regenerate ─────────────────────────────────────────────────────────

@app.post(
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


# ─── Utility endpoints ────────────────────────────────────────────────────────

@app.get("/options", tags=["Utils"], summary="Valid enum values for frontend dropdowns")
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


@app.get("/health", tags=["Utils"], summary="Health check")
async def health():
    return {
        "status":         "ok",
        "service":        "Banner Maker API",
        "image_model":    IMAGE_MODEL,
        "version":        "5.0.0",
        "base_url":       BASE_URL,
        "images_dir":     str(IMAGES_DIR),
        "images_on_disk": len(list(IMAGES_DIR.glob("*.png"))),
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8800)),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )