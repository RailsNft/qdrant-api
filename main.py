from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    PointStruct, Distance, VectorParams,
    Filter, FieldCondition, MatchValue
)
from config import API_TOKEN, QDRANT_HOST, QDRANT_API_KEY

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
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    query_vector = model.encode(q).tolist()
    search_filter = None
    try:
        if domainemycv:
            search_filter = Filter(
                must=[FieldCondition(key="domainemycv", match=MatchValue(value=domainemycv))]
            )
    except Exception:
        search_filter = None

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
            nom=res.payload.get("nom_candidat", ""),
            prenom=res.payload.get("prenom_candidat", ""),
            email=res.payload.get("email_candidat", ""),
            poste_recherche_candidat=res.payload.get("poste_recherche_candidat", "")
        )
        for res in results if "id_candidat" in res.payload
    ]

@app.get("/list")
def list_indexed(key: str):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        limit=10000
    )

    candidats = []
    for point in results:
        payload = point.payload or {}
        if "id_candidat" not in payload:
            continue

        candidats.append({
            "id_candidat": payload.get("id_candidat"),
            "nom_candidat": payload.get("nom_candidat", ""),
            "prenom_candidat": payload.get("prenom_candidat", ""),
            "email_candidat": payload.get("email_candidat", ""),
            "domainemycv": payload.get("domainemycv", "autre")
        })

    return {
        "total": len(candidats),
        "candidats": candidats
    }

@app.post("/delete")
def delete_candidates(key: str, ids: List[str] = Body(...)):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    try:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector={"points": ids}
        )
        return {"status": "ok", "deleted": ids}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/init")
def init_collection(key: str):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    try:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        return {"status": "collection created", "collection": COLLECTION_NAME}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
