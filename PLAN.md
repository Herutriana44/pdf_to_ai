# Project Plan: PDF-to-AI RAG System

## Checklist

### Mandatory
- [ ] Upload & indexing PDF documents
- [ ] Semantic retrieval
- [ ] Citation-aware answer generation
- [ ] Multi-step prompt workflow (LangGraph)
- [ ] Basic hallucination mitigation (Ragas)
- [ ] API endpoint for query

### Secondary
- [ ] Multi-agent system
- [ ] Streaming response
- [ ] Query decomposition
- [ ] Re-ranking pipeline
- [ ] MCP/tool-calling integration
- [ ] LLM evaluation pipeline

---

## Execution Phases

1. **Phase 1: Foundation**
   - Environment setup & basic FastAPI structure.
   - PDF processing (PyMuPDF + Marker).
   - Qdrant integration for vector storage.

2. **Phase 2: RAG Pipeline**
   - Semantic retrieval logic.
   - Gemini API integration with LangGraph.
   - Citation-aware generation.

3. **Phase 3: Refinement & Evaluation**
   - Hallucination mitigation (Ragas).
   - API endpoints & documentation.

4. **Phase 4: Advanced Features**
   - Secondary todo implementation.
