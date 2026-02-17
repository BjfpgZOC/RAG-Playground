from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.retriever.dense_retriever import DenseRetriever
from rag.pipelines.rag_pipeline import RAGPipeline
from qdrant_client import QdrantClient
from rag.settings import QDRANT_URL

app = FastAPI(title = "RAG-Assitant")
retriver = DenseRetriever()
pipeline = RAGPipeline()

@app.get("/health")
def health() -> Dict[str, bool]:
    QdrantClient(url = QDRANT_URL).get_collections()
    return {"ok": True}

class PingRequest(BaseModel):
    message: str

@app.post("/ping")
def ping(req: PingRequest) -> Dict[str, str]:
    return {"you_sent": req.message} 

class RetrieveRequest(BaseModel):
    query: str = Field(..., description = "User query string to retrieve relevant chunks")
    top_k: int = Field(5, ge = 1, le = 50, description = "How many chunks to retrieve (top-K)")
    min_score: Optional[float] = Field(None, description = "Optional score threshold to filter weak matches")

class RetrieveResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]

@app.post("/retrieve", response_model = RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    hits = retriver.retrieve(
        query_text = req.query,
        top_k = req.top_k,
        min_score = req.min_score,
    )

    return RetrieveResponse(query = req.query, results = hits)

class ChatRequest(BaseModel):
    question: str = Field(..., description = "User question to answer with RAG")
    top_k: int = Field(5, ge = 1, le = 50, description = "How many chunks to retrieve (top-K)")
    min_score: Optional[float] = Field(None, description = "Optional score threshold to filter weak matches")
    return_contexts: bool = Field(False, description = "Return retrieved chunks in the response for debugging")

class ChatResponse(BaseModel):
    question: str
    response: str
    context: Optional[List[Dict[str, Any]]] = None

@app.post("/chat", response_model = ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    pipeline.top_k = req.top_k
    pipeline.min_score = req.min_score

    output = pipeline.answer(req.question)

    if not req.return_contexts:
        output["context"] = None

    return ChatResponse(
        question = output["question"],
        response = output["response"],
        context = output["context"]
    )