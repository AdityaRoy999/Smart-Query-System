from typing import List, Dict, Any

class DocumentChunk:
    def__init__(self,text:str,embedding: List[float]):
    self.text = text
    self.embedding = embedding

class ProcessedDocument:
    def __init__(self,document_id: str, chunks: List[DocumentChunk]):
        self.document_id = document_id
        self.chunks = chunks