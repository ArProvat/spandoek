"""
Banner Maker API  v3  — FastAPI + OpenAI GPT Image
===================================================

Both endpoints accept multipart/form-data so callers send real image files
instead of base64 strings.  Uploaded files are:
  1. Saved to an OS temp file  (tempfile.NamedTemporaryFile)
  2. Read back and base64-encoded for the OpenAI call
  3. Deleted from disk in a finally-block (guaranteed cleanup)

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
import json
import logging
import os
import tempfile
import textwrap
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, List, Literal, Optional

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

client    = AsyncOpenAI(api_key=OPENAI_API_KEY)
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gpt-image-1")

# ─── Accepted image MIME types ────────────────────────────────────────────────

ALLOWED_IMAGE_TYPES = {
     "image/png",
     "image/jpeg",
     "image/jpg",
     "image/webp",
     "image/gif",
}

MAX_IMAGE_BYTES = 20 * 1024 * 1024   # 20 MB per file (keep well under OpenAI's 50 MB)

# ─── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
     title="Banner Maker API",
     version="3.0.0",
     description=(
          "AI-powered personalised banner generation — **multipart/form-data** interface.\n\n"
          "**Workflow:**\n"
          "1. `POST /generate` — Submit banner details + optional reference image files → "
          "streams 4 variants.\n"
          "2. User selects a favourite.\n"
          "3. `POST /regenerate` — Upload the selected banner file + prompt → streams refined result.\n\n"
          "> All image files are saved to OS temp storage, converted to base64 for the OpenAI call, "
          "then **deleted immediately** — nothing is persisted on disk."
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
     # ── New styles (user-requested) ───────────────────────────────────────────
     three_d_illustration = "3d_illustration"
     pixel_art            = "pixel_art"
     minimalistic         = "minimalistic"
     cartoon              = "cartoon"
     realistic            = "realistic"
     surreal              = "surreal"
     two_d                = "2d"
     flat_design          = "flat_design"
     # ── Kept from v2 ─────────────────────────────────────────────────────────
     elegant              = "elegant"
     playful              = "playful"
     bold_modern          = "bold_modern"
     vintage_retro        = "vintage_retro"
     watercolor           = "watercolor"
     neon_glow            = "neon_glow"
     rustic_natural       = "rustic_natural"
     luxury_gold          = "luxury_gold"
     dark_dramatic        = "dark_dramatic"


# ─── Pydantic sub-models (used inside the JSON "data" form field) ─────────────

class PersonalInfo(BaseModel):
     name:       Optional[str]       = Field(None, max_length=100,  description="Person / couple name(s).")
     age:        Optional[int]       = Field(None, ge=1, le=150,    description="Age — used for milestone events.")
     hobbies:    Optional[List[str]] = Field(None,                  description="Hobbies; AI picks matching motifs.")
     profession: Optional[str]       = Field(None, max_length=100,  description="Job title for corporate/farewell banners.")
     message:    Optional[str]       = Field(None, max_length=300,  description="Personal message to include on the banner.")


class GenerateData(BaseModel):
     """
     JSON payload sent as the `data` form field in POST /generate.
     Image files are sent separately as `reference_images` file fields.
     """
     occasion:        BannerOccasion         = Field(..., description="Event type.")
     style:           VisualStyle            = Field(VisualStyle.elegant, description="Visual style.")
     custom_occasion: Optional[str]          = Field(None, max_length=100,
                                                       description="Required when occasion='custom'.")
     personal_info:   Optional[PersonalInfo] = Field(None)
     headline:        Optional[str]          = Field(None, max_length=120,
                                                       description="Primary text rendered on the banner.")
     subtext:         Optional[str]          = Field(None, max_length=200,
                                                       description="Supporting text — date, venue, tagline.")
     description:     Optional[str]          = Field(None, max_length=1000,
                                                       description="Free-form creative direction.")
     reference_roles: Optional[List[str]]    = Field(
          None,
          description=(
               "One role string per reference image uploaded, in the same order as the files. "
               "E.g. ['use this colour palette', 'include this logo top-right']"
          ),
     )
     size:            Literal["1024x1024", "1536x1024", "1024x1536"] = Field("1536x1024")
     quality:         Literal["low", "medium", "high"]               = Field("medium")
     partial_images:  int                    = Field(2, ge=0, le=3)


# ─── Temp-file helpers ────────────────────────────────────────────────────────

async def upload_file_to_b64(upload: UploadFile) -> str:
     """
     Save an UploadFile to a named OS temp file, read it back as base64,
     then delete the temp file.  Returns the raw base64 string.

     Raises HTTPException (400) if the file is too large or wrong type.
     """
     # ── Validate content-type ─────────────────────────────────────────────────
     ct = (upload.content_type or "").lower()
     if ct not in ALLOWED_IMAGE_TYPES:
          raise HTTPException(
               status_code=status.HTTP_400_BAD_REQUEST,
               detail=(
                    f"File '{upload.filename}' has unsupported type '{ct}'. "
                    f"Allowed: {', '.join(sorted(ALLOWED_IMAGE_TYPES))}"
               ),
          )

     # ── Derive extension ──────────────────────────────────────────────────────
     ext_map = {
          "image/png":  ".png",
          "image/jpeg": ".jpg",
          "image/jpg":  ".jpg",
          "image/webp": ".webp",
          "image/gif":  ".gif",
     }
     suffix = ext_map.get(ct, ".png")

     tmp_path: Optional[str] = None
     try:
          # ── Write to temp file ────────────────────────────────────────────────
          with tempfile.NamedTemporaryFile(
               delete=False, suffix=suffix, prefix="banner_upload_"
          ) as tmp:
               tmp_path = tmp.name
               total_bytes = 0
               while True:
                    chunk = await upload.read(65_536)   # 64 KB chunks
                    if not chunk:
                         break
                    total_bytes += len(chunk)
                    if total_bytes > MAX_IMAGE_BYTES:
                         raise HTTPException(
                         status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                         detail=(
                              f"File '{upload.filename}' exceeds the "
                              f"{MAX_IMAGE_BYTES // (1024*1024)} MB limit."
                         ),
                         )
                    tmp.write(chunk)

          log.info(
               "Saved upload '%s' → %s  (%d bytes)",
               upload.filename, tmp_path, total_bytes,
          )

          # ── Read back and encode ──────────────────────────────────────────────
          raw_bytes = Path(tmp_path).read_bytes()
          b64 = base64.b64encode(raw_bytes).decode("ascii")
          log.info("Encoded '%s' to base64 (%d chars)", upload.filename, len(b64))
          return b64

     finally:
          # ── Always delete the temp file ───────────────────────────────────────
          if tmp_path and os.path.exists(tmp_path):
               os.unlink(tmp_path)
               log.info("Deleted temp file %s", tmp_path)


async def upload_files_to_b64(uploads: List[UploadFile]) -> List[str]:
     """Convert a list of UploadFiles to base64 strings concurrently."""
     return list(await asyncio.gather(*(upload_file_to_b64(u) for u in uploads)))

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
     # ── New styles ────────────────────────────────────────────────────────────
     "3d_illustration": (
          "Full 3-D rendered illustration style. Soft volumetric lighting, subsurface scattering on "
          "rounded objects, gentle drop-shadows, depth-of-field blur on background elements. "
          "Characters and props have a polished clay or smooth-plastic look (think Pixar / Blender render). "
          "Rich depth with foreground, midground and background layers clearly separated."
     ),
     "pixel_art": (
          "Retro pixel-art style. Crisp, hard-edged pixels — no anti-aliasing or blur. "
          "Limited colour palette (16–64 colours), dithering for gradients where needed. "
          "Characters and objects built from visible square pixels. "
          "8-bit or 16-bit game aesthetic. Grid-aligned composition."
     ),
     "minimalistic": (
          "Pure minimalist design. Extreme whitespace or single solid-colour background. "
          "At most 2–3 colours total. One dominant visual element, no clutter. "
          "Ultra-thin lines or simple geometric shapes only. "
          "Typography is sparse — a single clean sans-serif word or phrase at most."
     ),
     "cartoon": (
          "Bold cartoon illustration style. Strong black outlines (2–4 px), flat or cel-shaded fills, "
          "exaggerated proportions, expressive faces. Bright, punchy colours. "
          "Think Saturday-morning cartoon or comic-strip aesthetic. "
          "Characters have large eyes, simplified anatomy, and dynamic poses."
     ),
     "realistic": (
          "Photorealistic style. Studio-quality lighting (soft boxes or golden-hour natural light), "
          "accurate shadows and reflections, hyper-detailed textures (fabric weave, wood grain, metal sheen). "
          "Shot with a virtual 50 mm prime lens — shallow depth of field. "
          "Colour grading: natural, slightly warm, high dynamic range."
     ),
     "surreal": (
          "Dreamlike surrealist style inspired by Dalí and Magritte. "
          "Impossible physics: objects float, melt, morph or appear inside each other. "
          "Juxtapose unrelated items in unexpected scales. "
          "Hyper-detailed rendering of bizarre scenes. "
          "Colour palette: muted sky blues and warm ochres punctuated by vivid accent hues."
     ),
     "2d": (
          "Classic 2-D hand-drawn animation style. Clean ink lines with slight variation in stroke weight. "
          "Colour fills stay inside lines with minimal shading (simple cast-shadow flats only). "
          "Backgrounds are painterly washes; characters are crisp. "
          "Think 1990s Disney or Studio Ghibli background art."
     ),
     "flat_design": (
          "Modern flat design / material design aesthetic. Zero gradients, zero shadows, zero textures. "
          "Solid colour blocks only. Icons and illustrations built entirely from geometric primitives "
          "(circles, rectangles, rounded rectangles). "
          "Colour palette: bold, distinct, Material Design or Fluent palette. "
          "Typography: heavy geometric sans-serif."
     ),
     # ── Kept from v2 ─────────────────────────────────────────────────────────
     "elegant": (
          "Refined and sophisticated. Smooth gradients, serif or script typography, generous whitespace."
     ),
     "playful": (
          "Fun and whimsical. Rounded shapes, bright colours, hand-drawn feel, cheerful lettering."
     ),
     "bold_modern": (
          "Strong geometric shapes, high contrast, bold sans-serif type, dynamic asymmetric composition."
     ),
     "vintage_retro": (
          "Worn texture overlays, muted palette, retro badge shapes, distressed lettering, nostalgic feel."
     ),
     "watercolor": (
          "Soft watercolour washes, organic bleed edges, painterly background, delicate brush strokes."
     ),
     "neon_glow": (
          "Dark background, vibrant neon light effects, glowing colour halos, cyberpunk energy."
     ),
     "rustic_natural": (
          "Wood-grain texture, earthy tones, hand-lettered feel, botanical illustration accents."
     ),
     "luxury_gold": (
          "Deep rich backgrounds (black / navy / emerald), lavish gold foil textures, premium typography."
     ),
     "dark_dramatic": (
          "Dark moody palette, cinematic lighting, dramatic shadows, intense theatrical atmosphere."
     ),
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
     if info.name:
          parts.append(f"The banner is for {info.name}.")
     if info.age:
          parts.append(f"They are turning {info.age} years old.")
     if info.profession:
          parts.append(f"Their profession is {info.profession}.")
     if info.hobbies:
          parts.append(
               "Hobbies/interests: " + ", ".join(info.hobbies[:8]) + ". "
               "Subtly incorporate relevant motifs from these interests into the design."
          )
     if info.message:
          parts.append(f'Include this personal message on the banner: "{info.message}".')
     return " ".join(parts) if parts else "No specific personal details provided."


def _text_block(data: GenerateData) -> str:
     parts: list[str] = []
     if data.headline:
          parts.append(f'Primary headline (render prominently): "{data.headline}".')
     if data.subtext:
          parts.append(f'Secondary text (smaller, supporting): "{data.subtext}".')
     return " ".join(parts) if parts else "No specific text required — keep text minimal."


def _occasion_label(data: GenerateData) -> str:
     if data.occasion == BannerOccasion.custom and data.custom_occasion:
          return data.custom_occasion
     return data.occasion.value.replace("_", " ")


def build_variant_prompts(
     data: GenerateData,
     ref_b64_list: List[str],
     ) -> list[str]:
     """
     Build 4 distinct image-generation prompts from the parsed form data.
     Reference images are already base64-encoded by this point; their roles
     are embedded as text guidance in the prompt.
     """
     label     = _occasion_label(data)
     meta      = _OCCASION_META.get(data.occasion.value, _OCCASION_META["custom"])
     style_dir = _STYLE_DIRECTIVE[data.style.value]
     personal  = _personal_block(data.personal_info)
     text      = _text_block(data)
     extra     = data.description or "None."

     # Build reference-image role note (images themselves are passed as content blocks)
     roles = data.reference_roles or []
     if ref_b64_list and not roles:
          # Files uploaded but no roles specified — generic guidance
          ref_note = (
               f"{len(ref_b64_list)} reference image(s) have been provided alongside this prompt. "
               "Use them to inform colour palette, style, and compositional choices."
          )
     elif roles:
          paired = [
               f"Image {i+1}: {role}"
               for i, role in enumerate(roles[:len(ref_b64_list)])
          ]
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
          data.occasion.value, data.style.value, len(ref_b64_list),
     )
     return prompts


# ─── Core streaming worker ────────────────────────────────────────────────────

async def _stream_variant(
     variant_idx:      int,
     prompt:           str,
     size:             str,
     quality:          str,
     partial_images:   int,
     ref_b64_list:     List[str],
     queue:            asyncio.Queue,
) -> None:
     """
     Generate one banner variant via the OpenAI Responses API (streaming).
     Posts SSE-ready dicts to `queue`; always posts a `_done` sentinel.
     """
     log.info("variant %d — starting", variant_idx)
     try:
          # Build content: text prompt + any reference images
          content: list[dict] = [{"type": "input_text", "text": prompt}]
          for b64 in ref_b64_list:
               content.append({
                    "type":      "input_image",
                    "image_url": f"data:image/png;base64,{b64}",
               })

          stream = await client.responses.create(
               model=IMAGE_MODEL,
               input=[{"role": "user", "content": content}],
               stream=True,
               tools=[{
                    "type":           "image_generation",
                    "partial_images": partial_images,
                    "quality":        quality,
                    "size":           size,
               }],
          )

          final_b64:      Optional[str] = None
          revised_prompt: Optional[str] = None

          async for event in stream:
               etype = getattr(event, "type", "")

               if etype == "response.image_generation_call.partial_image":
                    await queue.put({
                         "event":         "partial",
                         "variant":       variant_idx,
                         "partial_index": event.partial_image_index,
                         "image_b64":     event.partial_image_b64,
                    })
               elif etype == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if item and getattr(item, "type", "") == "image_generation_call":
                         final_b64      = item.result
                         revised_prompt = getattr(item, "revised_prompt", None)
               elif etype == "response.done":
                    resp = getattr(event, "response", None)
                    if resp:
                         for out in getattr(resp, "output", []):
                              if getattr(out, "type", "") == "image_generation_call":
                                   final_b64      = out.result
                                   revised_prompt = getattr(out, "revised_prompt", None)

          if final_b64:
               await queue.put({
                    "event":          "final",
                    "variant":        variant_idx,
                    "image_b64":      final_b64,
                    "revised_prompt": revised_prompt,
               })
          else:
               log.warning("variant %d — no final image returned", variant_idx)
               await queue.put({
                    "event":   "error",
                    "variant": variant_idx,
                    "message": "No image returned by model.",
               })

     except openai.BadRequestError as exc:
          log.error("variant %d — BadRequest: %s", variant_idx, exc)
          await queue.put({"event": "error", "variant": variant_idx, "message": exc.message})
     except Exception as exc:
          log.exception("variant %d — unexpected error", variant_idx)
          await queue.put({"event": "error", "variant": variant_idx, "message": str(exc)})
     finally:
          await queue.put({"event": "_done", "variant": variant_idx})


# ─── API 1 — Generate 4 variants ──────────────────────────────────────────────

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
               "JSON string containing all banner parameters. "
               "See `GenerateData` schema for fields. "
               "Example: `{\"occasion\":\"birthday\",\"style\":\"cartoon\","
               "\"headline\":\"Happy 30th!\",\"quality\":\"medium\"}`"
          ),
     ),
     reference_images: List[UploadFile] = File(
          default=[],
          description=(
               "Optional reference image files (PNG / JPEG / WEBP). "
               "Send 0–4 files. Each file's role should be described in `data.reference_roles`."
          ),
     ),
):
     """
     ## Generate 4 banner variants as a streaming SSE response

     ### How to call (multipart/form-data)

     Send two form fields:

     | Field | Type | Description |
     |-------|------|-------------|
     | `data` | JSON string | All banner parameters — see schema below |
     | `reference_images` | File(s) | 0–4 image files (PNG/JPEG/WEBP, max 20 MB each) |

     ### `data` JSON fields

     | Field | Required | Type | Description |
     |-------|----------|------|-------------|
     | `occasion` | ✅ | enum | `birthday` `wedding` `graduation` `corporate` … (13 options) |
     | `style` | | enum | `3d_illustration` `pixel_art` `cartoon` `realistic` `surreal` `2d` `flat_design` `minimalistic` `elegant` `playful` `neon_glow` `watercolor` … (17 options) |
     | `custom_occasion` | | string | Required when `occasion="custom"` |
     | `personal_info` | | object | `{name, age, hobbies[], profession, message}` |
     | `headline` | | string | Primary text on the banner |
     | `subtext` | | string | Date / venue / tagline |
     | `description` | | string | Free-form creative direction |
     | `reference_roles` | | string[] | One role per uploaded file — e.g. `["use this colour palette", "include this logo top-right"]` |
     | `size` | | enum | `1536x1024` (default) \\| `1024x1024` \\| `1024x1536` |
     | `quality` | | enum | `low` \\| `medium` (default) \\| `high` |
     | `partial_images` | | int 0–3 | Preview frames to stream per variant (default 2) |

     ### SSE events emitted

     | `event` | Key fields |
     |---------|------------|
     | `prompts_ready` | `prompts: string[]` — the 4 built prompts |
     | `partial` | `variant`, `partial_index`, `image_b64` |
     | `final` | `variant`, `image_b64`, `revised_prompt` |
     | `variant_done` | `variant` |
     | `all_done` | — |
     | `error` | `variant?`, `message` |

     ### cURL example

     ```bash
     curl -N -X POST http://localhost:8000/generate \\
          -F 'data={"occasion":"birthday","style":"3d_illustration","headline":"Happy 30th, Sarah!","personal_info":{"name":"Sarah","age":30,"hobbies":["photography","travel"]},"quality":"medium"}' \\
          -F 'reference_images=@logo.png' \\
          -F 'reference_images=@palette_ref.jpg'
     ```

     ### JavaScript (fetch) example

     ```js
     const form = new FormData();
     form.append('data', JSON.stringify({
          occasion: 'wedding',
          style: 'surreal',
          personal_info: { name: 'John & Emily', message: 'Two hearts, one journey' },
          headline: 'John & Emily — Together Forever',
          subtext: '22 August 2025 · The Grand Pavilion',
          reference_roles: ['match this venue colour palette'],
     }));
     form.append('reference_images', venuePhotoFile);  // File object from <input type="file">

     const res  = await fetch('/generate', { method: 'POST', body: form });
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
          if (evt.event === 'partial') showPreview(evt.variant, evt.image_b64);
          if (evt.event === 'final')   showFinal(evt.variant, evt.image_b64);
          if (evt.event === 'all_done') console.log('All 4 variants ready!');
          }
     }
     ```
     """

     # ── Parse and validate the JSON data field ────────────────────────────────
     try:
          parsed = GenerateData.model_validate_json(data)
     except (ValidationError, ValueError) as exc:
          raise HTTPException(
               status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
               detail=f"Invalid `data` JSON: {exc}",
          )

     # ── Filter out empty file slots (browser may send blank entries) ──────────
     valid_uploads = [f for f in reference_images if f.filename and f.filename.strip()]

     if len(valid_uploads) > 4:
          raise HTTPException(
               status_code=status.HTTP_400_BAD_REQUEST,
               detail="Maximum 4 reference images allowed.",
          )

     async def stream() -> AsyncGenerator[str, None]:
          yield sse_comment("connected")

          # ── Convert uploaded files to base64 (temp-file path) ─────────────────
          ref_b64_list: List[str] = []
          if valid_uploads:
               try:
                    ref_b64_list = await upload_files_to_b64(valid_uploads)
                    yield sse({
                         "event":   "files_processed",
                         "message": f"{len(ref_b64_list)} reference image(s) processed.",
                    })
               except HTTPException as exc:
                    yield sse({"event": "error", "message": exc.detail})
                    return

          # ── Build 4 prompts ────────────────────────────────────────────────────
          try:
               prompts = build_variant_prompts(parsed, ref_b64_list)
          except Exception as exc:
               log.exception("Prompt build failed")
               yield sse({"event": "error", "message": f"Prompt build error: {exc}"})
               return

          yield sse({"event": "prompts_ready", "prompts": prompts})

          # ── Launch 4 concurrent generation tasks ───────────────────────────────
          queue: asyncio.Queue = asyncio.Queue()
          total = len(prompts)   # always 4
          done_count = 0

          tasks = [
               asyncio.create_task(
                    _stream_variant(
                         variant_idx=i,
                         prompt=prompts[i],
                         size=parsed.size,
                         quality=parsed.quality,
                         partial_images=parsed.partial_images,
                         ref_b64_list=ref_b64_list,
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
          log.info(
               "All 4 variants complete | occasion=%s | style=%s",
               parsed.occasion.value, parsed.style.value,
          )

     return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


# ─── API 2 — Regenerate selected banner ───────────────────────────────────────

@app.post(
     "/regenerate",
     summary="Regenerate / refine a selected banner",
     tags=["API 2 · Regenerate"],
     response_description="SSE stream — partial previews then the final regenerated image.",
)
async def regenerate_banner(
     banner_image: UploadFile = File(
          ...,
          description="The selected banner image file to refine (PNG / JPEG / WEBP, max 20 MB).",
     ),
     prompt: str = Form(
          ...,
          min_length=3,
          max_length=2000,
          description=(
               "Instruction that guides the regeneration. "
               "E.g. 'Change background to starry night sky, keep text unchanged'."
          ),
     ),
     size: str = Form(
          default="1536x1024",
          description="Output size: `1536x1024` | `1024x1024` | `1024x1536`",
     ),
     quality: str = Form(
          default="medium",
          description="Rendering quality: `low` | `medium` | `high`",
     ),
     partial_images: int = Form(
          default=2,
          ge=0,
          le=3,
          description="Preview frames to stream (0–3).",
     ),
     action: str = Form(
          default="auto",
          description="`auto` (model decides) | `edit` (targeted change) | `generate` (fresh image).",
     ),
):
     """
     ## Regenerate / refine a selected banner

     Accepts a banner **image file** + a text prompt. Saves the file to a temp path,
     converts it to base64, and streams the refined result as SSE.

     ### How to call (multipart/form-data)

     | Field | Type | Description |
     |-------|------|-------------|
     | `banner_image` | File | The selected banner (PNG/JPEG/WEBP, max 20 MB) |
     | `prompt` | string | What to change |
     | `size` | string | `1536x1024` (default) |
     | `quality` | string | `medium` (default) |
     | `partial_images` | int | Preview frames 0–3 |
     | `action` | string | `auto` \\| `edit` \\| `generate` |

     ### `action` values

     | Value | Behaviour |
     |-------|-----------|
     | `auto` | Model decides whether to edit or regenerate (best for most cases) |
     | `edit` | Preserve structure, apply targeted change |
     | `generate` | Fresh image loosely inspired by the original |

     ### SSE events

     | `event` | Key fields |
     |---------|------------|
     | `file_processed` | `message` |
     | `partial` | `partial_index`, `image_b64` |
     | `final` | `image_b64`, `revised_prompt` |
     | `done` | — |
     | `error` | `message` |

     ### cURL example

     ```bash
     curl -N -X POST http://localhost:8000/regenerate \\
          -F 'banner_image=@my_selected_banner.png' \\
          -F 'prompt=Change the colour scheme to warm sunset tones and keep all text' \\
          -F 'action=edit' \\
          -F 'quality=medium'
     ```

     ### JavaScript example

     ```js
     async function regenerate(file, prompt, action = 'auto') {
          const form = new FormData();
          form.append('banner_image', file);    // File from <input type="file"> or canvas export
          form.append('prompt', prompt);
          form.append('action', action);

          const res = await fetch('/regenerate', { method: 'POST', body: form });
          // read SSE identically to /generate
     }
     ```

     ### Export canvas as file then regenerate

     ```js
     // Convert canvas to Blob, then send as file
     canvas.toBlob(async (blob) => {
          const file = new File([blob], 'banner.png', { type: 'image/png' });
          await regenerate(file, 'Make the background deep space with stars');
     }, 'image/png');
     ```
     """

     # ── Validate action param ─────────────────────────────────────────────────
     if action not in ("auto", "edit", "generate"):
          raise HTTPException(
               status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
               detail="action must be one of: auto, edit, generate",
          )
     if size not in ("1024x1024", "1536x1024", "1024x1536"):
          raise HTTPException(
               status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
               detail="size must be one of: 1024x1024, 1536x1024, 1024x1536",
          )
     if quality not in ("low", "medium", "high"):
          raise HTTPException(
               status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
               detail="quality must be one of: low, medium, high",
          )

     async def stream() -> AsyncGenerator[str, None]:
          yield sse_comment("connected")

          # ── Save upload → temp file → base64 ──────────────────────────────────
          try:
               banner_b64 = await upload_file_to_b64(banner_image)
               yield sse({
                    "event":   "file_processed",
                    "message": f"Banner '{banner_image.filename}' processed.",
               })
          except HTTPException as exc:
               yield sse({"event": "error", "message": exc.detail})
               return

          # ── Call OpenAI Responses API (streaming) ──────────────────────────────
          try:
               api_stream = await client.responses.create(
                    model=IMAGE_MODEL,
                    input=[{
                         "role": "user",
                         "content": [
                         {
                              "type": "input_text",
                              "text": prompt,
                         },
                         {
                              "type":      "input_image",
                              "image_url": f"data:image/png;base64,{banner_b64}",
                         },
                         ],
                    }],
                    stream=True,
                    tools=[{
                         "type":           "image_generation",
                         "partial_images": partial_images,
                         "quality":        quality,
                         "size":           size,
                         "action":         action,
                    }],
               )

               final_b64:      Optional[str] = None
               revised_prompt: Optional[str] = None

               async for event in api_stream:
                    etype = getattr(event, "type", "")

                    if etype == "response.image_generation_call.partial_image":
                         yield sse({
                         "event":         "partial",
                         "partial_index": event.partial_image_index,
                         "image_b64":     event.partial_image_b64,
                         })
                    elif etype == "response.output_item.done":
                         item = getattr(event, "item", None)
                         if item and getattr(item, "type", "") == "image_generation_call":
                              final_b64      = item.result
                              revised_prompt = getattr(item, "revised_prompt", None)
                    elif etype == "response.done":
                         resp = getattr(event, "response", None)
                         if resp:
                              for out in getattr(resp, "output", []):
                                   if getattr(out, "type", "") == "image_generation_call":
                                        final_b64      = out.result
                                        revised_prompt = getattr(out, "revised_prompt", None)

               if final_b64:
                    yield sse({
                         "event":          "final",
                         "image_b64":      final_b64,
                         "revised_prompt": revised_prompt,
                    })
               else:
                    yield sse({"event": "error", "message": "Model returned no image."})

          except openai.BadRequestError as exc:
               log.error("regenerate BadRequest: %s", exc)
               yield sse({"event": "error", "message": exc.message})
          except Exception as exc:
               log.exception("regenerate unexpected error")
               yield sse({"event": "error", "message": str(exc)})
          finally:
               yield sse({"event": "done"})

     return StreamingResponse(stream(), media_type="text/event-stream", headers=SSE_HEADERS)


# ─── Utility endpoints ────────────────────────────────────────────────────────

@app.get("/options", tags=["Utils"], summary="All valid enum values (for frontend dropdowns)")
async def options():
     """Returns every valid value for `occasion`, `style`, `size`, `quality`, `action`."""
     return {
          "occasions": [e.value for e in BannerOccasion],
          "styles": {
               "new": [
                    "3d_illustration",
                    "pixel_art",
                    "minimalistic",
                    "cartoon",
                    "realistic",
                    "surreal",
                    "2d",
                    "flat_design",
               ],
               "classic": [
                    "elegant",
                    "playful",
                    "bold_modern",
                    "vintage_retro",
                    "watercolor",
                    "neon_glow",
                    "rustic_natural",
                    "luxury_gold",
                    "dark_dramatic",
               ],
          },
          "sizes":   ["1024x1024", "1536x1024", "1024x1536"],
          "qualities": ["low", "medium", "high"],
          "actions":   ["auto", "edit", "generate"],
          "layout_archetypes": [a["name"] for a in _VARIANT_ARCHETYPES],
          "max_reference_images": 4,
          "max_image_size_mb": MAX_IMAGE_BYTES // (1024 * 1024),
     }


@app.get("/health", tags=["Utils"], summary="Health check")
async def health():
     return {
          "status":  "ok",
          "service": "Banner Maker API",
          "model":   IMAGE_MODEL,
          "version": "3.0.0",
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