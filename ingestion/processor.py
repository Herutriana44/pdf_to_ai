import os
import uuid
import fitz
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pdf_docs"

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_client = None
_embeddings = None


def get_client():
    global _client
    if _client is None:
        _client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        )
    return _client


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vector_size():
    """
    Detect embedding dimension automatically.
    """
    embeddings = get_embeddings()

    sample_vector = embeddings.embed_query("test")
    return len(sample_vector)


def init_collection():
    client = get_client()
    vector_size = get_vector_size()

    if client.collection_exists(COLLECTION_NAME):

        collection_info = client.get_collection(COLLECTION_NAME)

        existing_size = (
            collection_info.config.params.vectors.size
        )

        if existing_size != vector_size:
            print(
                f"[processor] Vector mismatch "
                f"(existing={existing_size}, new={vector_size})"
            )

            print(
                "[processor] Recreating collection..."
            )

            client.delete_collection(
                collection_name=COLLECTION_NAME
            )

            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    else:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )


def process_and_index_pdf(file_path: str):

    try:
        init_collection()

        # Extract PDF text
        doc = fitz.open(file_path)

        text = ""

        for page in doc:
            text += page.get_text()

        doc.close()

        if not text.strip():
            raise ValueError(
                "PDF contains no extractable text."
            )

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_text(text)

        if len(chunks) == 0:
            raise ValueError(
                "No chunks generated"
            )

        embeddings = get_embeddings()

        vectors = embeddings.embed_documents(
            chunks
        )

        # Create points
        points = []

        for chunk, vector in zip(
            chunks,
            vectors
        ):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "text": chunk
                    }
                )
            )

        get_client().upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        print(
            f"[processor] Indexed {len(points)} chunks"
        )

        return True

    except Exception as e:
        print(
            f"[processor] Error indexing PDF: {e}"
        )
        return False