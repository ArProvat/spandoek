"""
Banner Maker API  v4  — FastAPI + OpenAI GPT Image
===================================================

Fully corrected to match the official OpenAI Image Generation API:

  • Reference images → client.images.edit(image=[BytesIO, ...])   — no extra_body hacks
  • Streaming        → stream=True + partial_images param
                       events: image_generation.partial_image
  • No-reference     → client.images.generate(stream=True, ...)

Endpoints
─────────
  POST /generate      → Rich form inputs → 4 concurrent SSE banner streams
  POST /regenerate    → Upload selected banner + prompt → SSE stream of edit
  GET  /options       → List every valid enum value  (for frontend dropdowns)
  GET  /health        → Health check
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import textwrap
from enum import Enum
from pathlib import Path
from typing import Annotated, AsyncGenerator, List, Literal, Optional

import openai
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("banner_api")

# ─── OpenAI client ────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# Supported GPT Image models: gpt-image-1.5 | gpt-image-1 | gpt-image-1-mini
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

# ─── Accepted image MIME types ────────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
    "image/gif",
}

MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB per file

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Banner Maker API",
    version="4.0.0",
    description=(
        "AI-powered personalised banner generation — **multipart/form-data** interface.\n\n"
        "**Workflow:**\n"
        "1. `POST /generate` — Submit banner details + optional reference image files → "
        "streams 4 variants.\n"
        "2. User selects a favourite.\n"
        "3. `POST /regenerate` — Upload the selected banner file + prompt → streams refined result.\n\n"
        "> All image files are held in memory as BytesIO — nothing is written to disk."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# ─── Upload → in-memory bytes ─────────────────────────────────────────────────

async def upload_to_bytes(upload: UploadFile) -> bytes:
    """
    Read an UploadFile into memory, validate type/size.
    Returns raw bytes — no disk I/O at all.
    """
    ct = (upload.content_type or "").lower()
    if ct not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File '{upload.filename}' has unsupported type '{ct}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
            ),
        )

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(65_536)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    f"File '{upload.filename}' exceeds the "
                    f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."
                ),
            )
        chunks.append(chunk)

    raw = b"".join(chunks)
    log.info("Loaded upload '%s' into memory (%d bytes)", upload.filename, len(raw))
    return raw


def bytes_to_image_file(raw: bytes, filename: str = "image.png") -> io.BytesIO:
    """Wrap raw bytes in a named BytesIO — what the OpenAI SDK expects."""
    buf = io.BytesIO(raw)
    buf.name = filename
    return buf


async def uploads_to_bytes_list(uploads: List[UploadFile]) -> List[bytes]:
    """Convert multiple UploadFiles to raw bytes concurrently."""
    return list(await asyncio.gather(*(upload_to_bytes(u) for u in uploads)))


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
    "birthday":       {"motifs": "birthday cake with candles, balloons, confetti, streamers, gift boxes, stars",
                       "palette": "bright cheerful colours, gold accents",
                       "atmosphere": "joyful, celebratory, warm"},
    "wedding":        {"motifs": "floral arch, roses, peonies, white doves, rings, lace, romantic foliage",
                       "palette": "ivory, blush pink, champagne gold, sage green",
                       "atmosphere": "romantic, luxurious, timeless"},
    "anniversary":    {"motifs": "roses, champagne glasses, golden number, hearts, soft candles, starry night",
                       "palette": "gold, deep red, pearl white",
                       "atmosphere": "romantic, nostalgic, elegant"},
    "baby_shower":    {"motifs": "baby onesie, stork, baby animals, stars, clouds, pastel toys",
                       "palette": "soft pastels — mint, lavender, blush, sky blue",
                       "atmosphere": "sweet, gentle, whimsical"},
    "graduation":     {"motifs": "graduation cap and tassel, diploma scroll, laurel wreath, open book, confetti",
                       "palette": "navy blue, gold, white",
                       "atmosphere": "proud, accomplished, bright future"},
    "party":          {"motifs": "disco ball, music notes, cocktails, neon lights, dance floor, confetti",
                       "palette": "vibrant, high-contrast, electric",
                       "atmosphere": "energetic, fun, exciting"},
    "corporate":      {"motifs": "clean geometric shapes, subtle grid lines, brand-friendly abstract forms",
                       "palette": "professional — navy, slate, silver, white",
                       "atmosphere": "polished, authoritative, modern"},
    "product_launch": {"motifs": "spotlight beam, reveal curtain, starburst, abstract tech shapes, launch rocket",
                       "palette": "bold contrast, brand colours, metallic sheen",
                       "atmosphere": "exciting, innovative, anticipatory"},
    "sports":         {"motifs": "dynamic motion blur, stadium lights, sport-specific equipment, trophy",
                       "palette": "bold team colours, high-energy",
                       "atmosphere": "powerful, intense, victorious"},
    "holiday":        {"motifs": "seasonal decorations, snowflakes or sunshine, festive ornaments, wreaths",
                       "palette": "red, green, gold (winter) or bright warm tones (summer)",
                       "atmosphere": "festive, warm, family-oriented"},
    "farewell":       {"motifs": "open road, sunset horizon, paper planes, waving hands, flowers",
                       "palette": "warm sunset tones, soft gold",
                       "atmosphere": "nostalgic, hopeful, bittersweet"},
    "welcome":        {"motifs": "open door, sunrise, blooming flowers, outstretched hands, bright paths",
                       "palette": "fresh greens, sky blue, warm yellow",
                       "atmosphere": "inviting, warm, optimistic"},
    "custom":         {"motifs": "relevant objects and symbols for the specific event",
                       "palette": "appropriate colours for the occasion",
                       "atmosphere": "fitting the mood of the event"},
}

_STYLE_DIRECTIVE: dict[str, str] = {
    "3d_illustration": (
        "Full 3-D rendered illustration style. Soft volumetric lighting, subsurface scattering on "
        "rounded objects, gentle drop-shadows, depth-of-field blur on background elements. "
        "Characters and props have a polished clay or smooth-plastic look (think Pixar / Blender render). "
        "Rich depth with foreground, midground and background layers clearly separated."
    ),
    "pixel_art": (
        "Retro pixel-art style. Crisp, hard-edged pixels — no anti-aliasing or blur. "
        "Limited colour palette (16–64 colours), dithering for gradients where needed. "
        "Characters and objects built from visible square pixels. 8-bit or 16-bit game aesthetic."
    ),
    "minimalistic": (
        "Pure minimalist design. Extreme whitespace or single solid-colour background. "
        "At most 2–3 colours total. One dominant visual element, no clutter. "
        "Ultra-thin lines or simple geometric shapes only."
    ),
    "cartoon": (
        "Bold cartoon illustration style. Strong black outlines (2–4 px), flat or cel-shaded fills, "
        "exaggerated proportions, expressive faces. Bright, punchy colours. "
        "Think Saturday-morning cartoon or comic-strip aesthetic."
    ),
    "realistic": (
        "Photorealistic style. Studio-quality lighting, accurate shadows and reflections, "
        "hyper-detailed textures. Shot with a virtual 50 mm prime lens — shallow depth of field. "
        "Colour grading: natural, slightly warm, high dynamic range."
    ),
    "surreal": (
        "Dreamlike surrealist style inspired by Dalí and Magritte. "
        "Impossible physics: objects float, melt, morph or appear inside each other. "
        "Hyper-detailed rendering of bizarre scenes."
    ),
    "2d": (
        "Classic 2-D hand-drawn animation style. Clean ink lines with slight variation in stroke weight. "
        "Colour fills stay inside lines with minimal shading. "
        "Think 1990s Disney or Studio Ghibli background art."
    ),
    "flat_design": (
        "Modern flat design / material design aesthetic. Zero gradients, zero shadows, zero textures. "
        "Solid colour blocks only. Icons and illustrations built from geometric primitives. "
        "Typography: heavy geometric sans-serif."
    ),
    "elegant":        "Refined and sophisticated. Smooth gradients, serif or script typography, generous whitespace.",
    "playful":        "Fun and whimsical. Rounded shapes, bright colours, hand-drawn feel, cheerful lettering.",
    "bold_modern":    "Strong geometric shapes, high contrast, bold sans-serif type, dynamic asymmetric composition.",
    "vintage_retro":  "Worn texture overlays, muted palette, retro badge shapes, distressed lettering.",
    "watercolor":     "Soft watercolour washes, organic bleed edges, painterly background, delicate brush strokes.",
    "neon_glow":      "Dark background, vibrant neon light effects, glowing colour halos, cyberpunk energy.",
    "rustic_natural": "Wood-grain texture, earthy tones, hand-lettered feel, botanical illustration accents.",
    "luxury_gold":    "Deep rich backgrounds (black / navy / emerald), lavish gold foil textures, premium typography.",
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
        "layout":     "Left–right split layout. Left half: bold solid colour block with headline and subtext stacked. "
                      "Right half: main illustration or graphic element fills the space.",
        "mood_tweak": "Cooler, more modern tone with sharp contrasts.",
    },
    {
        "name":       "Diagonal Dynamic",
        "layout":     "Diagonal slash divides the banner into two zones. Decorative elements flow along "
                      "the diagonal for a sense of energy and motion. Typography placed on one clean side.",
        "mood_tweak": "Energetic and vibrant — slightly bolder saturation.",
    },
    {
        "name":       "Framed Elegant",
        "layout":     "Decorative border or frame surrounds the entire canvas. "
                      "Central content area with balanced, well-spaced typography. "
                      "Corner ornaments echo the occasion theme.",
        "mood_tweak": "Soft and refined — slightly desaturated palette for sophistication.",
    },
]


def _personal_block(info: Optional[PersonalInfo]) -> str:
    if not info:
        return "No specific personal details provided."
    parts: list[str] = []
    if info.name:        parts.append(f"The banner is for {info.name}.")
    if info.age:         parts.append(f"They are turning {info.age} years old.")
    if info.profession:  parts.append(f"Their profession is {info.profession}.")
    if info.hobbies:
        parts.append("Hobbies/interests: " + ", ".join(info.hobbies[:8]) + ". Subtly incorporate relevant motifs.")
    if info.message:     parts.append(f'Include this personal message: "{info.message}".')
    return " ".join(parts) if parts else "No specific personal details provided."


def _text_block(data: GenerateData) -> str:
    parts: list[str] = []
    if data.headline: parts.append(f'Primary headline (render prominently): "{data.headline}".')
    if data.subtext:  parts.append(f'Secondary text (smaller, supporting): "{data.subtext}".')
    return " ".join(parts) if parts else "No specific text required — keep text minimal."


def _occasion_label(data: GenerateData) -> str:
    if data.occasion == BannerOccasion.custom and data.custom_occasion:
        return data.custom_occasion
    return data.occasion.value.replace("_", " ")


def build_variant_prompts(
    data: GenerateData,
    ref_count: int,
) -> list[str]:
    """Build 4 distinct image-generation prompts."""
    label     = _occasion_label(data)
    meta      = _OCCASION_META.get(data.occasion.value, _OCCASION_META["custom"])
    style_dir = _STYLE_DIRECTIVE[data.style.value]
    personal  = _personal_block(data.personal_info)
    text      = _text_block(data)
    extra     = data.description or "None."

    roles = data.reference_roles or []
    if ref_count and not roles:
        ref_note = (
            f"{ref_count} reference image(s) provided. "
            "Use them to inform colour palette, style, and compositional choices."
        )
    elif roles:
        paired = [f"Image {i+1}: {role}" for i, role in enumerate(roles[:ref_count])]
        ref_note = "Reference images provided — honour these roles: " + "; ".join(paired) + "."
    else:
        ref_note = "No reference images provided."

    prompts: list[str] = []
    for arch in _VARIANT_ARCHETYPES:
        p = textwrap.dedent(f"""
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
        """).strip()
        prompts.append(p)

    log.info(
        "Built 4 prompts | occasion=%s | style=%s | refs=%d",
        data.occasion.value, data.style.value, ref_count,
    )
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
    Generate one banner variant via the OpenAI Images API.

    Strategy
    ────────
    • PRIMARY  — always do a standard (non-streaming) call first.
      This is guaranteed to work for all GPT Image models (gpt-image-1,
      gpt-image-1-mini, gpt-image-1.5) and returns response.data[0].b64_json.

    • STREAMING — partial previews via stream=True + partial_images are only
      supported on gpt-image-1.5.  When IMAGE_MODEL ends with "1.5" we use
      the streaming path.  All streaming event types received are logged so
      you can debug event names if the model changes.

    Reference images
    ────────────────
    Passed directly as BytesIO objects to client.images.edit(image=[...]).
    No extra_body, no base64 string encoding, no 'input' parameter.
    """
    log.info("variant %d — starting (refs=%d, model=%s)", variant_idx, len(ref_bytes_list), IMAGE_MODEL)

    # Streaming partial previews only work reliably on gpt-image-1.5
    use_streaming = partial_images > 0 and IMAGE_MODEL.endswith("1.5")

    try:
        image_files = [
            bytes_to_image_file(raw, filename=f"ref_{i}.png")
            for i, raw in enumerate(ref_bytes_list)
        ]

        # ── STREAMING PATH (gpt-image-1.5 only) ──────────────────────────────
        if use_streaming:
            log.info("variant %d — using streaming path (partial_images=%d)", variant_idx, partial_images)

            if ref_bytes_list:
                # Use context-manager form for async streaming — avoids the
                # "awaiting a stream directly then iterating" pitfall.
                stream_cm = client.images.edit(
                    model=IMAGE_MODEL,
                    image=image_files,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    stream=True,
                    partial_images=partial_images,
                )
            else:
                stream_cm = client.images.generate(
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

            async with stream_cm as stream:
                async for event in stream:
                    # Log every event type so we can identify the correct names
                    event_type = getattr(event, "type", type(event).__name__)
                    log.debug("variant %d — stream event: %s", variant_idx, event_type)

                    b64 = getattr(event, "b64_json", None)
                    if b64:
                        final_b64 = b64  # always capture; last value = final image

                    if event_type == "image_generation.partial_image":
                        idx = getattr(event, "partial_image_index", 0)
                        if idx < partial_images:
                            await queue.put({
                                "event":         "partial",
                                "variant":       variant_idx,
                                "partial_index": idx,
                                "image_b64":     b64,
                            })
                    elif event_type in ("image_generation.completed", "response.completed"):
                        revised_prompt = getattr(event, "revised_prompt", None)
                        if b64:
                            final_b64 = b64

            if final_b64:
                await queue.put({
                    "event":          "final",
                    "variant":        variant_idx,
                    "image_b64":      final_b64,
                    "revised_prompt": revised_prompt,
                })
                log.info("variant %d — done (streaming)", variant_idx)
                return

            # If streaming yielded nothing fall through to non-streaming below
            log.warning("variant %d — streaming yielded no image, falling back to non-streaming", variant_idx)

        # ── NON-STREAMING PATH (default; all models) ──────────────────────────
        log.info("variant %d — using non-streaming path", variant_idx)

        if ref_bytes_list:
            # Re-create BytesIO objects (they may have been consumed above)
            image_files = [
                bytes_to_image_file(raw, filename=f"ref_{i}.png")
                for i, raw in enumerate(ref_bytes_list)
            ]
            response = await client.images.edit(
                model=IMAGE_MODEL,
                image=image_files,      # list of BytesIO objects ✓
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )
        else:
            response = await client.images.generate(
                model=IMAGE_MODEL,
                prompt=prompt,
                size=size,
                quality=quality,
                n=1,
            )

        final_b64      = response.data[0].b64_json if response.data else None
        revised_prompt = getattr(response.data[0], "revised_prompt", None) if response.data else None

        if final_b64:
            await queue.put({
                "event":          "final",
                "variant":        variant_idx,
                "image_b64":      final_b64,
                "revised_prompt": revised_prompt,
            })
            log.info("variant %d — done (non-streaming)", variant_idx)
        else:
            log.warning("variant %d — no image returned", variant_idx)
            await queue.put({
                "event":   "error",
                "variant": variant_idx,
                "message": "No image returned by model.",
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
    tags=["API 1 · Generate"],
    response_description="Server-Sent Events — partial previews then final images.",
)
async def generate_banners(
    data: str = Form(
        ...,
        description=(
            "JSON string with all banner parameters. "
            'Example: {"occasion":"birthday","style":"cartoon","headline":"Happy 30th!","quality":"medium"}'
        ),
    ),
    ref_image_1: Optional[UploadFile] = File(default=None, description="Reference image 1 (PNG/JPEG/WEBP, max 20 MB)."),
    ref_image_2: Optional[UploadFile] = File(default=None, description="Reference image 2 (PNG/JPEG/WEBP, max 20 MB)."),
    ref_image_3: Optional[UploadFile] = File(default=None, description="Reference image 3 (PNG/JPEG/WEBP, max 20 MB)."),
    ref_image_4: Optional[UploadFile] = File(default=None, description="Reference image 4 (PNG/JPEG/WEBP, max 20 MB)."),
):
    """
    ## Generate 4 banner variants (SSE stream)

    ### `data` JSON fields

    | Field | Type | Notes |
    |-------|------|-------|
    | `occasion` | enum | `birthday` `wedding` `graduation` `corporate` … (13 options) |
    | `style` | enum | `cartoon` `3d_illustration` `pixel_art` `realistic` `surreal` … (17 options) |
    | `headline` | string | Primary text on the banner |
    | `subtext` | string | Date / venue / tagline |
    | `personal_info` | object | `{name, age, hobbies[], profession, message}` |
    | `reference_roles` | string[] | One role per uploaded file |
    | `size` | enum | `1536x1024` (default) \| `1024x1024` \| `1024x1536` |
    | `quality` | enum | `low` \| `medium` (default) \| `high` \| `auto` |
    | `partial_images` | int 0–3 | Preview frames per variant (default 2) |

    ### SSE events

    | `event` | Fields |
    |---------|--------|
    | `files_processed` | `message` |
    | `prompts_ready` | `prompts: string[]` |
    | `partial` | `variant`, `partial_index`, `image_b64` |
    | `final` | `variant`, `image_b64`, `revised_prompt` |
    | `variant_done` | `variant` |
    | `all_done` | — |
    | `error` | `variant?`, `message` |

    ### cURL example

    ```bash
    curl -N -X POST http://localhost:8000/generate \\
      -F 'data={"occasion":"birthday","style":"cartoon","headline":"Happy 30th!","quality":"medium"}' \\
      -F 'ref_image_1=@logo.png'
    ```
    """

    # ── Parse JSON form field ─────────────────────────────────────────────────
    try:
        parsed = GenerateData.model_validate_json(data)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid `data` JSON: {exc}",
        )

    # ── Collect non-empty file slots ──────────────────────────────────────────
    valid_uploads = [
        f for f in (ref_image_1, ref_image_2, ref_image_3, ref_image_4)
        if f is not None and f.filename and f.filename.strip()
    ]

    async def stream() -> AsyncGenerator[str, None]:
        yield sse_comment("connected")

        # ── Read reference images into memory (BytesIO — no disk writes) ──────
        ref_bytes_list: List[bytes] = []
        if valid_uploads:
            try:
                ref_bytes_list = await uploads_to_bytes_list(valid_uploads)
                yield sse({
                    "event":   "files_processed",
                    "message": f"{len(ref_bytes_list)} reference image(s) processed.",
                })
            except HTTPException as exc:
                yield sse({"event": "error", "message": exc.detail})
                return

        # ── Build 4 prompts ───────────────────────────────────────────────────
        try:
            prompts = build_variant_prompts(parsed, ref_count=len(ref_bytes_list))
        except Exception as exc:
            log.exception("Prompt build failed")
            yield sse({"event": "error", "message": f"Prompt build error: {exc}"})
            return

        yield sse({"event": "prompts_ready", "prompts": prompts})

        # ── Launch 4 concurrent generation tasks ──────────────────────────────
        queue: asyncio.Queue = asyncio.Queue()
        total      = len(prompts)
        done_count = 0

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
                log.error("Generation timed out")
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
        log.info("All 4 variants complete | occasion=%s | style=%s", parsed.occasion.value, parsed.style.value)

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


# ─── POST /regenerate ─────────────────────────────────────────────────────────

@app.post(
    "/regenerate",
    summary="Regenerate / refine a selected banner",
    tags=["API 2 · Regenerate"],
    response_description="SSE stream — partial previews then the final regenerated image.",
)
async def regenerate_banner(
    banner_image: UploadFile = File(..., description="Selected banner to refine (PNG/JPEG/WEBP, max 20 MB)."),
    prompt: str = Form(..., min_length=3, max_length=2000,
                       description="Edit instruction, e.g. 'Change background to starry night, keep text'."),
    size: str = Form(default="1536x1024"),
    quality: str = Form(default="medium"),
    partial_images: int = Form(default=2, ge=0, le=3),
):
    """
    ## Refine a selected banner (SSE stream)

    Accepts a banner image file + edit instruction.
    Uses `client.images.edit()` — reference image passed as BytesIO.

    ### cURL example

    ```bash
    curl -N -X POST http://localhost:8000/regenerate \\
      -F 'banner_image=@selected_banner.png' \\
      -F 'prompt=Change the colour scheme to warm sunset tones, keep all text' \\
      -F 'quality=medium'
    ```
    """

    if size not in ("1024x1024", "1536x1024", "1024x1536"):
        raise HTTPException(status_code=422, detail="size must be 1024x1024, 1536x1024, or 1024x1536")
    if quality not in ("low", "medium", "high", "auto"):
        raise HTTPException(status_code=422, detail="quality must be low, medium, high, or auto")

    async def stream() -> AsyncGenerator[str, None]:
        yield sse_comment("connected")

        # ── Read banner into memory ───────────────────────────────────────────
        try:
            raw = await upload_to_bytes(banner_image)
            yield sse({"event": "file_processed", "message": f"Banner '{banner_image.filename}' loaded."})
        except HTTPException as exc:
            yield sse({"event": "error", "message": exc.detail})
            return

        # ── Edit the banner ───────────────────────────────────────────────────
        try:
            image_file     = bytes_to_image_file(raw, filename=banner_image.filename or "banner.png")
            use_streaming  = partial_images > 0 and IMAGE_MODEL.endswith("1.5")
            final_b64:      Optional[str] = None
            revised_prompt: Optional[str] = None

            if use_streaming:
                async with client.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    stream=True,
                    partial_images=partial_images,
                ) as edit_stream:
                    async for event in edit_stream:
                        event_type = getattr(event, "type", type(event).__name__)
                        log.debug("regenerate stream event: %s", event_type)
                        b64 = getattr(event, "b64_json", None)
                        if b64:
                            final_b64 = b64
                        if event_type == "image_generation.partial_image":
                            idx = getattr(event, "partial_image_index", 0)
                            if idx < partial_images:
                                yield sse({"event": "partial", "partial_index": idx, "image_b64": b64})
                        elif event_type in ("image_generation.completed", "response.completed"):
                            revised_prompt = getattr(event, "revised_prompt", None)

            if not final_b64:
                # Non-streaming path (default, or streaming fallback)
                if use_streaming:
                    log.warning("regenerate — streaming yielded no image, falling back")
                    image_file = bytes_to_image_file(raw, filename=banner_image.filename or "banner.png")

                response = await client.images.edit(
                    model=IMAGE_MODEL,
                    image=image_file,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                )
                final_b64      = response.data[0].b64_json if response.data else None
                revised_prompt = getattr(response.data[0], "revised_prompt", None) if response.data else None

            if final_b64:
                yield sse({"event": "final", "image_b64": final_b64, "revised_prompt": revised_prompt})
            else:
                yield sse({"event": "error", "message": "Model returned no image."})

        except openai.BadRequestError as exc:
            log.error("regenerate BadRequest: %s", exc)
            yield sse({"event": "error", "message": getattr(exc, "message", str(exc))})
        except Exception as exc:
            log.exception("regenerate unexpected error")
            yield sse({"event": "error", "message": str(exc)})
        finally:
            yield sse({"event": "done"})

    return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


# ─── Utility endpoints ────────────────────────────────────────────────────────

@app.get("/options", tags=["Utils"], summary="All valid enum values (for frontend dropdowns)")
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
        "streaming": {
            "supported": True,
            "partial_images_range": "0–3",
            "event_type": "image_generation.partial_image",
        },
    }


@app.get("/health", tags=["Utils"], summary="Health check")
async def health():
    return {
        "status":      "ok",
        "service":     "Banner Maker API",
        "image_model": IMAGE_MODEL,
        "version":     "4.0.0",
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("ENV", "production") == "development",
        log_level="info",
    )