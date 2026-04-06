"""
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
from app.service.banner.banner_schema import GenerateData, PersonalInfo
from app.service.banner.banner_router import router as banner_router

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

app.include_router(banner_router)

@app.on_event("startup")
async def startup_event() -> None:
    cleanup_old_images()
    asyncio.create_task(_periodic_cleanup())
    log.info(
        "v5 started | model=%s | base_url=%s | images_dir=%s",
        IMAGE_MODEL, BASE_URL, IMAGES_DIR,
    )


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
    )