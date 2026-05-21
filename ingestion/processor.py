import os
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Load .env early so env vars are available when module is imported directly
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pdf_docs"

# all-MiniLM-L6-v2: lightweight open-source model, runs locally, 384-dim vectors
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384

# Lazy-init so objects are created after env vars are loaded
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


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def init_collection():
    client = get_client()
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
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
