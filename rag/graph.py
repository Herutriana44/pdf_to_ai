import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from ragas.metrics import faithfulness
from ragas import evaluate
from datasets import Dataset

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "pdf_docs"
qdrant_client = QdrantClient(url=QDRANT_URL)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

class AgentState(TypedDict):
    query: str
    sub_queries: List[str]
    context: List[str]
    answer: str
    citations: List[str]
    is_hallucination: bool

# Define specialized agent nodes
def retrieval_agent(state: AgentState):
    # Perform decomposition, retrieval, and reranking
    decomp_result = decompose_query(state)
    state.update(decomp_result)
    retrieval_result = retrieve_context(state)
    state.update(retrieval_result)
    rerank_result = rerank_context(state)
    return rerank_result

def generation_agent(state: AgentState):
    # Perform generation and evaluation
    gen_result = generate_answer(state)
    state.update(gen_result)
    eval_result = evaluate_hallucination(state)
    return {**gen_result, **eval_result}

workflow = StateGraph(AgentState)
workflow.add_node("retrieval_agent", retrieval_agent)
workflow.add_node("generation_agent", generation_agent)

workflow.set_entry_point("retrieval_agent")
workflow.add_edge("retrieval_agent", "generation_agent")
workflow.add_edge("generation_agent", END)

app = workflow.compile()
