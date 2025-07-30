# app/routes/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Optional
import time
import hashlib
from ..utils.file_processing import process_uploaded_file
from ..utils.embeddings import embed_chunks
from ..core.schemas import DocumentProcessResponse

router = APIRouter(prefix="/documents", tags=["documents"])

# In-memory storage for demo (replace with database in production)
documents_store = {}

@router.post("/upload", response_model=DocumentProcessResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        start_time = time.time()
        
        # Read file content
        file_bytes = await file.read()
        
        # Process file and get chunks
        chunks = process_uploaded_file(file_bytes, file.content_type)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract text from document")
        
        # Generate embeddings
        embedded_chunks = await embed_chunks(chunks)
        
        # Create document ID
        document_id = hashlib.sha256(file_bytes).hexdigest()
        
        # Store document (in-memory for demo)
        documents_store[document_id] = embedded_chunks
        
        processing_time = time.time() - start_time
        
        return DocumentProcessResponse(
            document_id=document_id,
            chunk_count=len(embedded_chunks),
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))