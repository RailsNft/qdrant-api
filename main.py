from fastapi import FastAPI, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, Distance, VectorParams,
    Filter, FieldCondition, MatchValue
)
from fastapi.responses import JSONResponse

from config import API_TOKEN, QDRANT_HOST, QDRANT_API_KEY

# ✅ App & modèle
app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
COLLECTION_NAME = "cv_index"

# CORS (optionnel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Modèle de sortie
class SearchResult(BaseModel):
    id: str
    score: float
    id_candidat: str = None
    nom: str = None
    prenom: str = None
    email: str = None
    poste_recherche_candidat: str = None

# ✅ Endpoint principal
@app.get("/search", response_model=List[SearchResult])
def search(q: str, key: str, domainemycv: Optional[str] = None):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    query_vector = model.encode(q).tolist()
    search_filter = None
    if domainemycv:
        search_filter = Filter(
            must=[
                FieldCondition(key="domainemycv", match=MatchValue(value=domainemycv))
            ]
        )

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10,
        with_payload=True,
        filter=search_filter
    )

    return [
        SearchResult(
            id=res.id,
            score=res.score,
            id_candidat=res.payload.get("id_candidat"),
            nom=res.payload.get("nom_candidat"),
            prenom=res.payload.get("prenom_candidat"),
            email=res.payload.get("email_candidat"),
            poste_recherche_candidat=res.payload.get("poste_recherche_candidat"),
        )
        for res in results
    ]
