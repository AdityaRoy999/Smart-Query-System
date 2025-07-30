from pydantic import BaseModel
from typing import List,Optional

class DocumentUpload(BaseModel):
    file_bytes: bytes
    content_type: str

class QueryRequest(BaseModel):
    query: str
    document_id: str

class QueryResponse(BaseModel):
    answer: str
    relevant_chunks: List[dict]
    processing_time: float

class DocumentProcessResponse(BaseModel):
    document_id: str
    chunk_count: int
    processing_time: float
    