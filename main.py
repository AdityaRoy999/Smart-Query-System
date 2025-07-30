from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import documents, queries
from app.core.config import settings

app = FastAPI(tile="Document Query API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(queries.router)

@app.get("/")
async def root():
    return {"message": "Document Query API is running"}