# API Documentation - PDF-to-AI RAG System

## Overview
This API provides endpoints to ingest PDF documents, perform semantic search, and generate citation-aware answers.

---

## 1. Health Check
`GET /health`
- **Description**: Verifies if the service is running.
- **Response**: `{"status": "ok"}`

---

## 2. Ingest PDF
`POST /ingest`
- **Description**: Uploads a PDF file for processing and indexing.
- **Content-Type**: `multipart/form-data`
- **Parameters**: 
    - `file`: PDF file.
- **Response**: `{"status": "success"}`

---

## 3. Query RAG
`POST /query`
- **Description**: Submits a user query and returns a citation-aware answer.
- **Content-Type**: `application/json`
- **Body**: 
    ```json
    {
        "query": "What is the summary of the document?"
    }
    ```
- **Response**:
    ```json
    {
        "answer": "The document covers...",
        "citations": ["doc1", "doc2"]
    }
    ```
