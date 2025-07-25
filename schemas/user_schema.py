from pydantic import BaseModel, EmailStr
from typing import List, Optional


class UserLogin(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    name: str
    email: EmailStr
    password: str
    
class UserPersonalResponse(BaseModel):
    personal_color_name: Optional[str] = None

class UserResponse(BaseModel):
    username: str
    name : str
    email: str
    personal_color_name : Optional[str] = None

    class Config:
        from_attributes = True

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[EmailStr] = None


class user_style_summary_update(BaseModel):
    budget: Optional[int] = None
    occasion: Optional[str] = None
    height: Optional[int] = None
    gender: Optional[str] = None
    top_size: Optional[str] = None
    bottom_size: Optional[int] = None
    shoe_size: Optional[int] = None
    body_feature: Optional[List[str]] = None
    preferred_styles: Optional[List[str]] = None
    user_situation : Optional[List[str]] = None

class user_favorite_look(BaseModel):
    user_id: str
    look_id: str
    look_name: str
    look_description: str

class user_style_summary(BaseModel):
    budget: int
    occasion: str
    height: int
    gender: str
    top_size: str
    bottom_size: int
    shoe_size: int
    body_feature: List[str]
    preferred_styles: List[str]
    user_situation : List[str]

class user_profile(BaseModel):
    id: int
    username: str
    name: str
    email: EmailStr
    password: str
    personal_color_name: str


