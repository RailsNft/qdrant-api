from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from config import API_TOKEN, QDRANT_HOST, QDRANT_API_KEY
import logging

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "cv_index"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔍 Recherche
class SearchResult(BaseModel):
    id: str
    score: float
    id_candidat: Optional[str]
    nom: Optional[str]
    prenom: Optional[str]
    email: Optional[str]
    poste_recherche_candidat: Optional[str]

@app.get("/search", response_model=List[SearchResult])
def search(q: str, key: str, domainemycv: Optional[str] = None):
    if key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    query_vector = model.encode(q).tolist()

    search_filter = None
    if domainemycv:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="domainemycv", match=MatchValue(value=domainemycv)
                )
            ]
        )

    try:
        results = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=10,
            with_payload=True,
            filter=search_filter
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"search failed: {e}")

    return [
        SearchResult(
            id=res.id,
            score=res.score,
            id_candidat=res.payload.get("id_candidat"),
            nom=res.payload.get("nom_candidat"),
            prenom=res.payload.get("prenom_candidat"),
            email=res.payload.get("email_candidat"),
            poste_recherche_candidat=res.payload.get("poste_recherche_candidat")
        )
        for res in results
    ]

# 📥 Indexation
@app.post("/index-payload")
def index_payload(points: List[PointStruct], key: str = Query(...)):
    if key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")

    for p in points:
        if not p.payload.get("id_candidat") or not p.payload.get("domainemycv"):
            raise HTTPException(status_code=400, detail="Missing id_candidat or domainemycv")

    try:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant insert failed: {e}")

    return {"status": "indexed", "count": len(points)}

# 🔎 Liste des points
@app.get("/list")
def list_points(key: str = Query(...)):
    if key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        hits = qdrant.scroll(collection_name=COLLECTION_NAME, limit=100, with_payload=True)
        return {"count": len(hits[0]), "items": hits[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"List error: {e}")


# ✅ Création d'index pour filtre
@app.get("/create-index")
def create_index(key: str = Query(...)):
    if key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="domainemycv",
            field_schema="keyword"
        )
        return {"status": "index created", "field": "domainemycv"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index creation error: {e}")


# ❌ Suppression candidat
@app.delete("/delete")
def delete_candidate(id: str, key: str = Query(...)):
    if key != API_TOKEN:
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        qdrant.delete(collection_name=COLLECTION_NAME, points_selector={"points": [id]})
        return {"status": "deleted", "id": id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete error: {e}")


# 🌐 Test simple
@app.get("/")
def root():
    return {"status": "API OK"}
@app.post("/encode")
def encode_text(data: dict):
    try:
        text = data.get("text", "")
        if not text:
            return JSONResponse(status_code=400, content={"error": "Missing 'text' field"})
        vector = model.encode(text).tolist()
        return {"vector": vector}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": "Encoding failed", "details": str(e)})
