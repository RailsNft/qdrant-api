from fastapi import FastAPI, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import SearchRequest, Filter, FieldCondition, MatchValue

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "cv_index"

app = FastAPI()
from config import API_TOKEN
model = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResult(BaseModel):
    id_candidat: str = None
    id: str
    score: float
    nom: str = None
    prenom: str = None
    email: str = None
    poste_recherche_candidat: str = None

@app.get("/search", response_model=List[SearchResult])
def search(q: str, key: str, domaine_mycv: Optional[str] = None = Query(..., description="Requête texte")):
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    query_vector = model.encode(q).tolist()
    if key != API_TOKEN:
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    search_filter = None
    if domaine_mycv:
        search_filter = Filter(must=[FieldCondition(key="domaine-mycv", match=MatchValue(value=domaine_mycv))])
    results = qdrant.search(
        filter=search_filter,
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10,
        with_payload=True
    )
    return [
        SearchResult(
        SearchResult(
            score=res.score,
            nom=res.payload.get("nom_candidat"),
            prenom=res.payload.get("prenom_candidat"),
            email=res.payload.get("email_candidat"),
            poste_recherche_candidat=res.payload.get("poste_recherche_candidat"),
            id=res.id,
            id_candidat=res.payload.get("id_candidat"),
        )
        for res in results
    ]


from fastapi import Request, Body

@app.post("/index-payload")
async def index_payload(request: Request):
    token = request.headers.get("Authorization")
    if token != f"Bearer {API_TOKEN}":
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)
    data = await request.json()
    fields = [
        "nom_candidat", "prenom_candidat", "email_candidat", "telephone_candidat",
        "mobile_candidat", "rs_linkedln", "adresse_candidat", "code_postal_candidat",
        "ville_candidat", "date_naissance_candidat", "poste_recherche_candidat",
        "resume_formation", "resume_experience", "resume_competence",
        "resume_langue", "loisir_candidat", "desc_mini", "login_candidat"
    ]

    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=model.get_sentence_embedding_dimension(), distance=Distance.COSINE)
            id=res.id,
            id_candidat=res.payload.get("id_candidat"),
        )

    points = []
    for row in data:
        if 'id_candidat' not in row or not row['id_candidat']:
            return JSONResponse(content={"error": "id_candidat est obligatoire"}, status_code=400)
        if 'domaine-mycv' not in row or not row['domaine-mycv']:
            return JSONResponse(content={"error": "domaine-mycv est obligatoire"}, status_code=400)
        id_ = str(row.get("id_candidat"))
        text = " ".join(str(v) for k, v in row.items() if v and isinstance(v, str))
        vector = model.encode(text)
        points.append(PointStruct(id=id_, vector=vector.tolist(), payload=row))
        if 'domaine-mycv' not in payload:
            payload['domaine-mycv'] = 'autre'

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return {"status": "indexed", "count": len(points)}


@app.post("/delete")
async def delete_endpoint(
    ids: List[str] = Body(...),
    request: Request = None
):
    token = request.headers.get("Authorization")
    if token != f"Bearer {API_TOKEN}":
        return JSONResponse(content={"error": "unauthorized"}, status_code=401)

    qdrant.delete(collection_name=COLLECTION_NAME, points_selector={"points": ids})
    return {"status": "deleted", "count": len(ids)}
