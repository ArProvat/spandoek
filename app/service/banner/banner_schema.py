from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import UploadFile
from typing import Literal
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
