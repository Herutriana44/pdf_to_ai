import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "")
COLLECTION_NAME = "pdf_docs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_qdrant_client = None
_llm = None
_embeddings = None


def get_qdrant_client():
    global _qdrant_client

    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY if QDRANT_API_KEY else None,
        )

    return _qdrant_client


def get_llm():

    global _llm

    if _llm is None:
        _llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL_ID,
            temperature=0.2
        )

    return _llm


def get_embeddings():

    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


class AgentState(TypedDict):

    query: str
    sub_queries: List[str]
    context: List[str]
    answer: str
    citations: List[str]
    is_hallucination: bool


def extract_text(response):

    """
    Convert Gemini response safely to string
    """

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        texts = []

        for item in content:

            if isinstance(item, str):
                texts.append(item)

            elif isinstance(item, dict):
                texts.append(
                    item.get("text", "")
                )

            else:
                texts.append(
                    str(item)
                )

        return "\n".join(texts)

    return str(content)


def decompose_query(state):

    prompt = ChatPromptTemplate.from_template(
        """
Break the question into 2-3 focused retrieval questions.

Return ONLY:
one question per line

Question:
{query}
"""
    )

    chain = prompt | get_llm()

    result = chain.invoke({
        "query": state["query"]
    })

    text = extract_text(result)

    sub_queries = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    if len(sub_queries) == 0:
        sub_queries = [state["query"]]

    return {
        "sub_queries": sub_queries
    }


def retrieve_context(state):

    contexts = []
    seen = set()

    queries = state.get(
        "sub_queries",
        [state["query"]]
    )

    for q in queries:

        vector = get_embeddings().embed_query(q)

        hits = get_qdrant_client().query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=4
        )

        print(f"[LOG] hits payload result {hits}")

        for hit in hits:

            text = hit.payload.get(
                "text",
                ""
            )

            if text and text not in seen:

                seen.add(text)
                contexts.append(text)

    return {
        "context": contexts
    }


def rerank_context(state):

    contexts = state.get(
        "context",
        []
    )

    query_words = set(
        state["query"].lower().split()
    )

    scored = sorted(
        contexts,
        key=lambda x:
        len(
            query_words &
            set(x.lower().split())
        ),
        reverse=True
    )

    return {
        "context": scored[:6]
    }


def generate_answer(state):

    context_text = "\n\n---\n\n".join(
        state.get(
            "context",
            []
        )
    )

    prompt = ChatPromptTemplate.from_template(
        """
Answer ONLY from the context.

If information is missing say:
"I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""
    )

    chain = prompt | get_llm()

    result = chain.invoke({
        "context": context_text,
        "query": state["query"]
    })

    answer = extract_text(result)

    citations = [
        f"Chunk {i+1}"
        for i in range(
            len(
                state.get(
                    "context",
                    []
                )
            )
        )
    ]

    return {
        "answer": answer,
        "citations": citations
    }


def evaluate_hallucination(state):

    answer_words = set(
        state.get(
            "answer",
            ""
        ).lower().split()
    )

    context_words = set(
        " ".join(
            state.get(
                "context",
                []
            )
        ).lower().split()
    )

    if len(answer_words) == 0:

        return {
            "is_hallucination": False
        }

    overlap = (
        len(
            answer_words &
            context_words
        )
        / len(answer_words)
    )

    return {
        "is_hallucination": overlap < 0.4
    }


def retrieval_agent(state):

    state.update(
        decompose_query(state)
    )

    state.update(
        retrieve_context(state)
    )

    state.update(
        rerank_context(state)
    )

    return state


def generation_agent(state):

    state.update(
        generate_answer(state)
    )

    state.update(
        evaluate_hallucination(state)
    )

    return state


workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "retrieval_agent",
    retrieval_agent
)

workflow.add_node(
    "generation_agent",
    generation_agent
)

workflow.set_entry_point(
    "retrieval_agent"
)

workflow.add_edge(
    "retrieval_agent",
    "generation_agent"
)

workflow.add_edge(
    "generation_agent",
    END
)

app = workflow.compile()