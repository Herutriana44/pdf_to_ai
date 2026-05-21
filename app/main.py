import asyncio
import os
import shutil

import uvicorn
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

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
    # Fix #9: use try/finally so temp file is always removed even if processing fails
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Fix #8: run blocking CPU/IO work in a thread so the event loop is not blocked
        success = await asyncio.to_thread(process_and_index_pdf, file_path)

        if not success:
            raise HTTPException(status_code=422, detail="Failed to process PDF.")

        return {"status": "success"}

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


async def generate_stream(query: str):
    # 1. Jalankan graph RAG sampai selesai
    result = await asyncio.to_thread(rag_app.invoke, {"query": query})
    answer = result.get("answer", "")
    
    # Cek di log server apakah backend sebenarnya menghasilkan jawaban
    print(f"[SERVER DEBUG] Answer dari RAG: {answer}")
    
    if not answer:
        yield "I don't have enough information.".encode('utf-8')
        return

    # 2. Kirim per kata atau per beberapa karakter agar chunk lebih stabil
    # Mengirim per kata jauh lebih aman untuk streaming berbasis teks biasa
    words = answer.split(" ")
    for i, word in enumerate(words):
        # Tambahkan spasi kembali kecuali untuk kata pertama
        chunk = f" {word}" if i > 0 else word
        yield chunk.encode('utf-8')  # Wajib di-encode ke bytes
        await asyncio.sleep(0.02)   # Delay sedikit untuk efek mengetik

@app.post("/query")
async def query_rag(query: dict = Body(...)):
    query_text = query.get("query", "")
    if not query_text:
        raise HTTPException(status_code=400, detail="'query' field is required.")
    
    # Gunakan media_type text/plain atau text/event-stream
    return StreamingResponse(generate_stream(query_text), media_type="text/plain")
    
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
