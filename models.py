from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json
import dateparser
import numpy as np
from bson import ObjectId
import numpy as np


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return json.JSONEncoder.default(self, o)

# Pydantic models
class User(BaseModel):
    username: str
    email: str
    is_admin: bool = False

class UserInDB(User):
    _id: ObjectId
    hashed_password: bytes
    created_at: datetime

class UserDisplay(BaseModel):
    id: str
    username: str
    email: str
    is_admin: bool

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class Skill(BaseModel):
    name: str
    score: Optional[float] = 0

class Education(BaseModel):
    institution: str
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class Experience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None

class ContactInfo(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    website: Optional[str] = None

class ResumeData(BaseModel):
    id: Optional[str] = None
    candidate_name: str
    contact_info: ContactInfo
    skills: List[Skill] = []
    education: List[Education] = []
    experience: List[Experience] = []
    raw_text: str
    embedding: Optional[List[float]] = None
    file_path: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class JobProfile(BaseModel):
    id: Optional[str] = None
    title: str
    description: str
    required_skills: List[str] = []
    preferred_skills: List[str] = []
    education_requirements: List[str] = []
    experience_years: Optional[int] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class CandidateMatch(BaseModel):
    resume_id: str
    candidate_name: str
    match_score: float
    skill_matches: Dict[str, float]
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    is_admin: bool = False



class ResumeHit(BaseModel):
    candidate_name: str
    snippet: str
class RAGQuery(BaseModel):
    query: str  # User's question about the resume

class ResumeHit(BaseModel):
    candidate_name: str
    snippet: str

class RAGResponse(BaseModel):
    results: List[ResumeHit]
    answer: str
