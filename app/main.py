from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import asyncio
from ingestion.processor import process_and_index_pdf
from rag.graph import app as rag_app

app = FastAPI(title="PDF-to-AI API")

# Serve static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("app/static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    success = process_and_index_pdf(file_path)
    os.remove(file_path)
    return {"status": "success" if success else "failed"}

async def generate_stream(query: str):
    result = rag_app.invoke({"query": query})
    answer = result["answer"]
    # Simulate streaming
    for char in answer:
        yield char
        await asyncio.sleep(0.01)

@app.post("/query")
async def query_rag(query: dict = Body(...)):
    # Adjust query handling if needed
    query_text = query.get("query", "")
    return StreamingResponse(generate_stream(query_text), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
