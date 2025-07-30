import os
import re
import docx
import tempfile
from PyPDF2 import PdfReader
from typing import Tuple
from ..core.config import settings

def extract_text(file_path: str, file_type: str) -> str:
    if file_type == "application/pdf":
        text = ""
        try:
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {e}")
        return text
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {e}")
    
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX.")

def chunk_text(text: str, max_tokens: int = None) -> list[str]:
    max_tokens=max_tokens or settings.max_tokens
    text=re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks=[]
    current_chunk=""

    for sentence in sentences:
        if len(current_chunk.split())+len(sentence.split())<=max_tokens:
            current_chunk+=sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk=sentence+" "

    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_uploaded_file(file_bytes: bytes, content_type: str) -> Tuple[str, list[str]]:
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path=tmp.name
    
    try:
        full_text = extract_text(tmp_path, content_type)
        return chunk_text(full_text)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
                        