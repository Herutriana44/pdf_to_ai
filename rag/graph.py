import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Load .env early so env vars are available when module is imported directly
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "pdf_docs"

# Fix: lazy-init so objects are created after env vars are loaded
_qdrant_client = None
_llm = None
_embeddings = None


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        )
    return _qdrant_client


def get_llm() -> ChatGoogleGenerativeAI:
    global _llm
    if _llm is None:
        _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    return _llm


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        # gemini-embedding-001 produces 3072-dim vectors
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return _embeddings


class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    context: List[str]
    answer: str
    citations: List[str]
    is_hallucination: bool


def decompose_query(state: AgentState) -> dict:
    """Break the user query into sub-queries for better retrieval coverage."""
    prompt = ChatPromptTemplate.from_template(
        "Break the following question into 2-3 focused sub-questions for document retrieval.\n"
        "Return only the sub-questions, one per line.\n\nQuestion: {query}"
    )
    chain = prompt | get_llm()
    result = chain.invoke({"query": state["query"]})
    sub_queries = [line.strip() for line in result.content.strip().split("\n") if line.strip()]
    return {"sub_queries": sub_queries or [state["query"]]}


def retrieve_context(state: AgentState) -> dict:
    """Retrieve relevant document chunks from Qdrant for each sub-query."""
    all_contexts: List[str] = []
    seen: set = set()

    queries = state.get("sub_queries") or [state["query"]]
    for q in queries:
        query_vector = get_embeddings().embed_query(q)
        hits = get_qdrant_client().search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=4,
        )
        for hit in hits:
            text = hit.payload.get("text", "")
            if text and text not in seen:
                seen.add(text)
                all_contexts.append(text)

    return {"context": all_contexts}


def rerank_context(state: AgentState) -> dict:
    """Simple reranking: keep top-N chunks most relevant to the original query."""
    context = state.get("context", [])
    query_words = set(state["query"].lower().split())
    scored = sorted(
        context,
        key=lambda c: len(query_words & set(c.lower().split())),
        reverse=True,
    )
    return {"context": scored[:6]}


def generate_answer(state: AgentState) -> dict:
    """Generate an answer grounded in the retrieved context."""
    context_text = "\n\n---\n\n".join(state.get("context", []))
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, say 'I don't have enough information to answer that.'\n\n"
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    )
    chain = prompt | get_llm()
    result = chain.invoke({"context": context_text, "query": state["query"]})
    citations = [f"Chunk {i+1}" for i in range(len(state.get("context", [])))]
    return {"answer": result.content, "citations": citations}


def evaluate_hallucination(state: AgentState) -> dict:
    """Heuristic hallucination check: flag if answer has low overlap with context."""
    answer = state.get("answer", "")
    context_combined = " ".join(state.get("context", [])).lower()
    answer_words = set(answer.lower().split())
    context_words = set(context_combined.split())

    if not answer_words:
        return {"is_hallucination": False}

    overlap = len(answer_words & context_words) / len(answer_words)
    return {"is_hallucination": overlap < 0.4}


# --- Agent nodes ---

def retrieval_agent(state: AgentState) -> dict:
    decomp_result = decompose_query(state)
    state.update(decomp_result)
    retrieval_result = retrieve_context(state)
    state.update(retrieval_result)
    rerank_result = rerank_context(state)
    return {**decomp_result, **retrieval_result, **rerank_result}


def generation_agent(state: AgentState) -> dict:
    gen_result = generate_answer(state)
    state.update(gen_result)
    eval_result = evaluate_hallucination(state)
    return {**gen_result, **eval_result}


# --- Build graph ---

workflow = StateGraph(AgentState)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("generation_agent", generation_agent)

workflow.set_entry_point("retrieval_agent")
workflow.add_edge("retrieval_agent", "generation_agent")
workflow.add_edge("generation_agent", END)

app = workflow.compile()
