import logging
from pathlib import Path
from openai import AsyncOpenAI
import base64
import time
import uuid
import asyncio
import os

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


BASE_URL = os.getenv("BASE_URL", "http://localhost:8800").rstrip("/")

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "/tmp/banner_images"))
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

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
