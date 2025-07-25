from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union

class ItemInfo(BaseModel):
    category: Optional[str] = None
    item_code: Optional[str] = None
    category_id: Optional[str] = None
    color: Optional[str] = None

class LookInfo(BaseModel):
    look_name: str
    look_description: str
    items: Dict[str, Union[ItemInfo, None]] = Field(default_factory=dict)
    
    class Config:
        # None 값을 허용하도록 설정
        extra = "allow"  

class StyleRecommendation(BaseModel):
    style_name: str
    looks: List[LookInfo]

class GeminiExamplePrompt(BaseModel):
    recommendations: List[StyleRecommendation]

# 룩 적합도 분석을 위한 새로운 스키마
class LookSuitability(BaseModel):
    """개별 룩의 적합도 분석 결과를 담는 모델"""
    look_name: str = Field(..., description="분석된 룩의 이름")
    suitability_score: int = Field(..., description="룩의 적합도 점수 (0-100)")
    reason: str = Field(..., description="점수에 대한 간략한 이유")

class LookSuitabilityAnalysisResponse(BaseModel):
    """룩 적합도 분석 결과의 리스트를 담는 모델"""
    looks_analysis: List[LookSuitability] = Field(..., description="각 룩에 대한 적합도 분석 결과 리스트")
    
    
    