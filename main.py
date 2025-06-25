from fastapi import FastAPI, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, Distance, VectorParams,
    Filter, FieldCondition, MatchValue,
    SearchRequest
)
from config import API_TOKEN, QDRANT_HOST, QDRANT_API_KEY

# Init
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "cv_index"

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modèle de retour
class SearchResult(BaseModel):
    id: str
    score: float
    id_candidat: Optional[str]
    nom: Optional[str]
    prenom: Optional[str]
    email: Optional[str]
    poste_recherche_candidat: Optional[str]

# 🔍 Recherche sémantique
@app.get("/search", response_model=List[SearchResult])
def search(q: str, key: str, domainemycv: Optional[str] = None):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    try:
        query_vector = model.encode(q).tolist()
        search_filter = None
        if domainemycv:
            search_filter = Filter(must=[
                FieldCondition(key="domainemycv", match=MatchValue(value=domainemycv))
            ])
        request = SearchRequest(
            vector=query_vector,
            limit=10,
            with_payload=True,
            filter=search_filter
        )
        results = qdrant.search(collection_name=COLLECTION_NAME, search_request=request)
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
    except Exception as e:
        return JSONResponse(content={"error": "search failed", "details": str(e)}, status_code=500)

# 📥 Indexation
@app.post("/index-payload")
def index_payload(data: List[dict] = Body(...)):
    try:
        points = []
        for item in data:
            if "id_candidat" not in item or "domainemycv" not in item:
                return JSONResponse(content={"error": "Missing id_candidat or domainemycv"}, status_code=400)
            id_candidat = item["id_candidat"]
            text = " ".join([str(v) for v in item.values() if isinstance(v, str)])
            vector = model.encode(text).tolist()
            points.append(PointStruct(
                id=id_candidat,
                vector=vector,
                payload=item
            ))
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        return {"status": "ok", "indexed": len(points)}
    except Exception as e:
        return JSONResponse(content={"error": "indexing failed", "details": str(e)}, status_code=500)

# ❌ Suppression
@app.delete("/delete-candidate/{id}")
def delete_candidate(id: str, key: str):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    try:
        qdrant.delete(collection_name=COLLECTION_NAME, points_selector={"points": [id]})
        return {"status": "deleted", "id": id}
    except Exception as e:
        return JSONResponse(content={"error": "deletion failed", "details": str(e)}, status_code=500)

# 📋 Liste des candidats
@app.get("/list")
def list_all(key: str):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    try:
        scroll = qdrant.scroll(collection_name=COLLECTION_NAME, with_payload=True, limit=1000)
        return {
            "count": len(scroll[0]),
            "items": scroll[0]
        }
    except Exception as e:
        return JSONResponse(content={"error": "listing failed", "details": str(e)}, status_code=500)
