# app/routes/queries.py
from fastapi import APIRouter, HTTPException
from typing import Optional
import time
from ..utils.embeddings import search_relevant_chunks, generate_response
from ..core.schemas import QueryRequest, QueryResponse
from ..routes.documents import documents_store

router = APIRouter(prefix="/queries", tags=["queries"])

@router.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    """Ask a question about a processed document."""
    try:
        start_time = time.time()
        
        # Get document from store
        embedded_chunks = documents_store.get(query.document_id)
        if not embedded_chunks:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Search for relevant chunks
        relevant_chunks = search_relevant_chunks(query.query, embedded_chunks)
        if not relevant_chunks:
            return QueryResponse(
                answer="No relevant information found in the document",
                relevant_chunks=[],
                processing_time=time.time() - start_time
            )
        
        # Generate response
        answer = generate_response(query.query, relevant_chunks)
        
        return QueryResponse(
            answer=answer,
            relevant_chunks=relevant_chunks,
            processing_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))