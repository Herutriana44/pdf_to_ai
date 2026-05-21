import os
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitters
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pdf_docs"
client = QdrantClient(url=QDRANT_URL)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def init_collection():
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )

def process_and_index_pdf(file_path: str):
    init_collection()
    doc = fitz.open(file_path)
    text = "".join([page.get_text() for page in doc])
    doc.close()
    
    text_splitter = RecursiveCharacterTextSplitters(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # Generate vectors and upsert
    vectors = embeddings.embed_documents(chunks)
    points = [
        models.PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return True
