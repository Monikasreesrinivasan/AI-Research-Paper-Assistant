from typing import List, Optional
from pydantic import BaseModel

class PaperSummary(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    summary: str
    topics: List[str]
    audio_path: Optional[str] = None

class PodcastInfo(BaseModel):
    topic: str
    paper_titles: List[str]
    podcast_path: str
    script: str