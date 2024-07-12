from fastapi import FastAPI, Path, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import httpx
import os
from dotenv import load_dotenv

_ = load_dotenv()

app = FastAPI()

PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
PINECONE_ENDPOINT = os.environ.get("PINECONE_ENDPOINT")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

assert PINECONE_INDEX is not None
assert PINECONE_ENDPOINT is not None
assert PINECONE_API_KEY is not None

class Vector(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any]

class UpsertRequest(BaseModel):
    vectors: List[Vector]
    namespace: Optional[str] = None

class QueryRequest(BaseModel):
    namespace: Optional[str] = "default"
    vector: List[float]
    topK: int = 3
    includeValues: bool = False
    includeMetadata: bool = True
    filter: Optional[Dict[str, Any]] = None

@app.get("/indexes/{indexName}")
async def describe_index(indexName: str = Path(..., description="The name of the index to describe.")):
    url = f"https://api.pinecone.io/indexes/{indexName}"
    headers = {"Api-Key": PINECONE_API_KEY}
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

@app.post("/vectors/upsert")
async def upsert_vectors(request: UpsertRequest):
    url = f"{PINECONE_ENDPOINT}/vectors/upsert"
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=request.dict())
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

@app.post("/query")
async def query_vectors(request: QueryRequest):
    url = f"{PINECONE_ENDPOINT}/query"
    headers = {
        "Api-Key": PINECONE_API_KEY,
        "Content-Type": "application/json"
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=request.dict())
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    return response.json()

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000, ssl_keyfile="localhost.key", ssl_certfile="localhost.crt")
