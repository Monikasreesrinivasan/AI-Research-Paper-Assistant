from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")

class PaperBase(BaseModel):
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    content: str
    file_path: str
    vector_id: Optional[str] = None

class PaperCreate(PaperBase):
    pass

class PaperInDB(PaperBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

class Paper(PaperBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

class SimilarPaper(BaseModel):
    paper_id: str
    title: str
    authors: List[str]
    abstract: Optional[str] = None
    url: Optional[str] = None
    score: float
    source: str  # 'semantic_scholar' or 'arxiv'

class PaperModel(BaseModel):
    id: Optional[str] = None
    title: str
    authors: List[str] = []
    abstract: str = ""
    content: Optional[str] = None
    file_path: Optional[str] = None
    summary: Optional[str] = ""
    topics: List[str] = []
    audio_path: Optional[str] = None
    podcast_path: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class PaperSummaryResponse(BaseModel):
    title: str
    authors: List[str]
    topics: List[str]
    summary: str
    audio_url: Optional[str] = None

class TopicPodcastResponse(BaseModel):
    topic: str
    papers: List[str]
    podcast_url: str
    script: str
