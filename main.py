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
from fastapi import Body
from qdrant_client.http.models import PointStruct

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
