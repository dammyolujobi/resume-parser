import os
import faiss
import numpy as np
from typing import List
from fastapi import HTTPException, Depends,FastAPI
from pydantic import BaseModel
import cohere
from main import app, db  # assuming main app context

# RAG Configuration
COHERE_API_KEY = "Psgf5ZaD224nY4PPp29tXwYcNe6zMXXclpQ0Dsoi"
if not COHERE_API_KEY:
    raise EnvironmentError("COHERE_API_KEY is required for RAG functionality")
co = cohere.Client(COHERE_API_KEY)

# Embedding model name
EMBED_MODEL = 'embed-english-v2.0'

# FAISS Index Setup (using inner product for cosine similarity)
EMBED_DIM = 1024  # Cohere embed-english-v2.0 outputs 1024-dimensional vectors
INDEX_FILE = "faiss_index.bin"

# Initialize or load FAISS index
if os.path.exists(INDEX_FILE):
    index = faiss.read_index(INDEX_FILE)
else:
    index = faiss.IndexFlatIP(EMBED_DIM)

# Mapping of vector positions to document metadata
metadata: List[dict] = []

# Build/Rebuild FAISS index from MongoDB
async def build_index():
    resumes = await db.resumes.find().to_list(1000)
    for resume in resumes:
        text = resume.get("raw_text")
        if text:
            # Get Cohere embedding
            resp = co.embed(model=EMBED_MODEL, texts=[text])
            emb = np.array(resp.embeddings[0], dtype="float32")
            # normalize for cosine similarity
            emb = emb / np.linalg.norm(emb)
            index.add(np.array([emb]))
            metadata.append({
                "id": str(resume.get("_id")),
                "candidate_name": resume.get("candidate_name"),
                "text": text
            })
    faiss.write_index(index, INDEX_FILE)

# Trigger index build on startup
async def lifespan(app: FastAPI):
    await build_index()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    top_k: int = 3

class QueryResult(BaseModel):
    id: str
    candidate_name: str
    snippet: str
    score: float

class QueryResponse(BaseModel):
    results: List[QueryResult]
    answer: str

def retrieve(query: str, top_k: int = 3):
    # 1) compute query embedding
    resp = co.embed(model=EMBED_MODEL, texts=[query])
    qemb = np.array(resp.embeddings[0], dtype="float32")
    qemb /= np.linalg.norm(qemb)

    # 2) make sure dims match
    if qemb.shape[0] != index.d:
        # rebuild index to match new dim
        print(f"Rebuilding FAISS index: old d={index.d}, new d={qemb.shape[0]}")
        global index, metadata
        index = faiss.IndexFlatIP(qemb.shape[0])
        metadata = []
        # you can either re-load: await build_index()
        # or error out so your startup logic can run again
        raise HTTPException(500, detail="FAISS index dimension mismatch – rebuilding required")

    # 3) search
    D, I = index.search(np.expand_dims(qemb, axis=0), top_k)
    ...

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < len(metadata):
            doc = metadata[idx]
            snippet = doc["text"][:500] + "..."
            results.append(QueryResult(
                id=doc["id"],
                candidate_name=doc["candidate_name"],
                snippet=snippet,
                score=float(score)
            ))
    return results

# RAG endpoint
@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(req: QueryRequest, current_user=Depends(lambda: True)):
    retrieved = retrieve(req.query, req.top_k)
    if not retrieved:
        raise HTTPException(status_code=404, detail="No relevant documents found")

    # Build contexts
    contexts = "\n\n".join(
        f"Candidate: {r.candidate_name}\nText: {r.snippet}" for r in retrieved
    )
    prompt = (
        f"You are a recruitment assistant. Based on the following candidate resumes, answer the question: {req.query}\n\n"
        f"{contexts}\n\nAnswer:"
    )

    # Generate answer via Cohere
    gen = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=200,
        temperature=0.3,
        k=0,
        p=0.75,
        stop_sequences=["--"],
    )
    answer = gen.generations[0].text.strip()

    return QueryResponse(results=retrieved, answer=answer)

# Endpoint to add a new resume to index
def add_to_index(doc_id: str, raw_text: str, candidate_name: str = "Unknown"):
    resp = co.embed(model=EMBED_MODEL, texts=[raw_text])
    emb = np.array(resp.embeddings[0], dtype="float32")
    emb = emb / np.linalg.norm(emb)
    index.add(np.array([emb]))
    metadata.append({"id": doc_id, "candidate_name": candidate_name, "text": raw_text})
    faiss.write_index(index, INDEX_FILE)

class ResumeDoc(BaseModel):
    id: str
    raw_text: str
    candidate_name: str = "Unknown"

@app.post("/rag/index")
async def index_resume(doc: ResumeDoc, current_user=Depends(lambda: True)):
    try:
        add_to_index(doc.id, doc.raw_text, doc.candidate_name)
        return {"message": "Document indexed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")