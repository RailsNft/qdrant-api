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
from uuid import uuid4

from config import API_TOKEN, QDRANT_HOST, QDRANT_API_KEY

app = FastAPI()
model = SentenceTransformer("paraphrase-albert-small-v2")
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
    id_candidat: str = None
    nom: str = None
    prenom: str = None
    email: str = None
    poste_recherche_candidat: str = None

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
        query_filter=search_filter  # ✅ corrigé ici
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

@app.post("/index-payload")
def index_payload(payload: List[dict] = Body(...)):
    if not payload:
        return {"error": "Empty payload"}

    points = []
    for row in payload:
        texte = " ".join(str(v) for v in row.values() if isinstance(v, str)).strip()
        vector = model.encode(texte).tolist()

        point_id = row.get("id_candidat", str(uuid4()))
        domain = row.get("domainemycv", "autre")

        if not point_id or not domain:
            return {"error": f"Missing id_candidat or domainemycv for row: {row}"}

        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload=row
        ))

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        wait=True,
        points=points
    )

    return {"status": "ok", "indexed": len(points)}
@app.post("/delete")
def delete_candidates(payload: dict = Body(...)):
    ids = payload.get("ids")
    if not ids or not isinstance(ids, list):
        return {"error": "Missing or invalid 'ids'"}

    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector={"points": ids}
    )

    return {"status": "deleted", "count": len(ids)}
