# Government AI Assistant

A high-scalability, multi-agent RAG (Retrieval-Augmented Generation) system for government document processing and query assistance.

## System Flow (Short)
1. **Ingestion**: Documents are uploaded and processed through a queued pipeline (PDF/DOCX -> Chunking -> Embedding).
2. **Storage**: Vector data is stored in Weaviate/Qdrant, while metadata/sessions go to PostgreSQL.
3. **Orchestration**: A LangGraph agent manages the user query, utilizing a RAG pipeline (Hybrid Search + Reranking) to retrieve context.
4. **Generation**: LLMs generate formal, validated answers with citations.
5. **Monitoring**: Quality metrics (RAGAS) and logs (ELK/Prometheus) ensure accuracy and system health.

## Prerequisites
- Docker & Docker Compose
- Python 3.11+
- API Keys: OpenAI/Anthropic/Groq

## Configuration
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Open `.env` and fill in the required variables (API keys, database URLs, etc.).

## How to Run
1. Start the infrastructure (Database, Vector DB, Message Broker):
   ```bash
   docker-compose up -d
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app/main.py
   ```

*For detailed architectural documentation, refer to `architecture.drawio` and `SYSTEM_EXPLANATION.txt`.*
