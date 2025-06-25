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
