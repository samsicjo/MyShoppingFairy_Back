from pydantic import BaseModel
from typing import List, Dict, Any

class ColorAnalysisDetail(BaseModel):
    hsv: List[int]
    rgb: List[int]
    tone_category: str
    brightness_category: str
    saturation_category: str

class ColorInfo(BaseModel):
    hex: List[str]
    analysis: List[ColorAnalysisDetail]

class FaceColorData(BaseModel):
    eyes: ColorInfo
    nose: ColorInfo
    lips: ColorInfo
    hair: ColorInfo
    skin: ColorInfo

class PersonalColorAnalysis(BaseModel):
    personal_color_type: str
    personal_color_type_not: str
    skin_type_analysis: str
    PCCS_Tone_circle: List[str]
    Hair_color_hex: List[str]
    Hair_color_name: List[str]
    Hair_tone: str
    Accessory_color_recommendation: List[str]
    makeup_tips: str

class PersonalColorResponse(BaseModel):
    personal_color_analysis: str

