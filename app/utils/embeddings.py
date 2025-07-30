# app/utils/embeddings.py
import os
import time
import requests
import numpy as np
from typing import List, Dict
from ..core.config import settings

def get_embedding(text_chunk: str) -> List[float]:
    """Generates embedding for a single text chunk."""
    embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={settings.api_key}"
    headers = {"Content-Type": "application/json"}
    body = {
        "model": settings.embedding_model,
        "content": {"parts": [{"text": text_chunk}]}
    }
    
    max_retries = 3
    delay = 1
    for attempt in range(max_retries):
        try:
            resp = requests.post(embed_url, headers=headers, json=body, timeout=60)
            resp.raise_for_status()
            embedding = resp.json().get("embedding", {}).get("values")
            if not embedding:
                raise ValueError("API response did not contain embedding values.")
            return embedding
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(f"API request failed after {max_retries} attempts: {e}")
    return None

async def embed_chunks(chunks: List[str]) -> List[Dict]:
    """Embeds a list of text chunks in batches."""
    batch_embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:batchEmbedContents?key={settings.api_key}"
    headers = {"Content-Type": "application/json"}
    
    all_embedded_chunks = []
    
    for i in range(0, len(chunks), settings.batch_size):
        chunk_batch = chunks[i:i + settings.batch_size]
        requests_list = [
            {"model": settings.embedding_model, "content": {"parts": [{"text": chunk}]}}
            for chunk in chunk_batch
        ]
        body = {"requests": requests_list}

        max_retries = 3
        delay = 2
        for attempt in range(max_retries):
            try:
                resp = requests.post(batch_embed_url, headers=headers, json=body, timeout=180)
                resp.raise_for_status()
                
                embeddings = resp.json().get("embeddings", [])
                if len(embeddings) != len(chunk_batch):
                    raise ValueError(f"API Error: Mismatch in batch. Sent {len(chunk_batch)}, got {len(embeddings)}.")

                batch_embedded = [
                    {"id": f"chunk_{i+j}", "text": chunk, "embedding": emb.get("values")}
                    for j, (chunk, emb) in enumerate(zip(chunk_batch, embeddings))
                    if emb.get("values")
                ]
                all_embedded_chunks.extend(batch_embedded)
                break

            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise RuntimeError(f"API request failed after {max_retries} attempts: {e}")

    return all_embedded_chunks

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    v1, v2 = np.array(v1), np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def search_relevant_chunks(query: str, embedded_chunks: List[Dict], top_k: int = None) -> List[Dict]:
    """Finds most relevant chunks to a query."""
    top_k = top_k or settings.top_k_results
    if not embedded_chunks:
        return []
    
    query_embedding = get_embedding(query)
    scored_chunks = []
    
    for chunk in embedded_chunks:
        score = cosine_similarity(query_embedding, chunk["embedding"])
        scored_chunks.append({"text": chunk["text"], "score": score, "embedding": chunk["embedding"]})
    
    return sorted(scored_chunks, key=lambda x: x["score"], reverse=True)[:top_k]

def generate_response(query: str, relevant_chunks: List[Dict]) -> str:
    """Generates natural language response based on query and relevant chunks."""
    generation_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={settings.api_key}"
    context = "\n\n".join([f"Clause: {c['text']}\nScore: {c['score']:.4f}" for c in relevant_chunks])
    
    prompt = f"""
You are a professional assistant trained to review documents. Your task is to give a direct answer to a user's query based *only* on the relevant clauses provided from the document.

**User Query:**
"{query}"

**Relevant Clauses from Document:**
---
{context}
---

**Instructions:**
1.  Provide a direct, concise, natural language answer in one or two sentences.
2.  Start your answer directly with "Yes," "No," or state that the information isn't available.
3.  **Do not** use phrases like "According to the text," "The provided clauses state," or similar preambles.
4.  If the information is not in the clauses, state that the answer cannot be found in the provided sections.
"""
    
    headers = {"Content-Type": "application/json"}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        resp = requests.post(generation_url, headers=headers, json=body, timeout=120)
        resp.raise_for_status()
        response_data = resp.json()
        candidates = response_data.get("candidates", [])
        if candidates and "content" in candidates[0] and "parts" in candidates[0]["content"]:
            return candidates[0]["content"]["parts"][0].get("text", "Sorry, I couldn't generate a valid response.")
        return f"Sorry, the API returned an unexpected response format.\n{response_data}"
    except requests.exceptions.RequestException as e:
        return f"Sorry, I couldn't generate a response due to an API error: {e}"