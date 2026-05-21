import os
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load .env early so env vars are available when module is imported directly
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pdf_docs"

# Fix: lazy-init so objects are created after env vars are loaded
_client = None
_embeddings = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        )
    return _client


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        # gemini-embedding-001 produces 3072-dim vectors
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embeddings


def init_collection():
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=3072, distance=models.Distance.COSINE),
        )


def process_and_index_pdf(file_path: str) -> bool:
    try:
        init_collection()

        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        doc.close()

        if not text.strip():
            raise ValueError("PDF contains no extractable text.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        embeddings = get_embeddings()
        vectors = embeddings.embed_documents(chunks)

        points = [
            models.PointStruct(id=i, vector=vectors[i], payload={"text": chunks[i]})
            for i in range(len(chunks))
        ]
        get_client().upsert(collection_name=COLLECTION_NAME, points=points)
        return True

    except Exception as e:
        print(f"[processor] Error indexing PDF: {e}")
        return False
