# knowledgepilot_AI
# 🧠 KnowledgePilot AI

KnowledgePilot AI is a Retrieval-Augmented Generation (RAG) chatbot that answers questions from course transcripts using semantic search and OpenAI. Instead of relying only on the language model's knowledge, it retrieves relevant information from a custom knowledge base and generates context-aware responses.

The project was built from scratch to understand the complete RAG pipeline, including preprocessing, embeddings, vector search, prompt engineering, guardrails, memory, API development, and Docker deployment.

---

## 🚀 Features

- Semantic search using Sentence Transformers
- Vector database with PostgreSQL + PGVector
- Retrieval-Augmented Generation (RAG)
- OpenAI GPT-powered answer generation
- Streaming responses
- Conversation memory
- Input validation guardrails
- Retrieval validation guardrails
- Output validation guardrails
- REST API using FastAPI
- Dockerized application
- Swagger UI for API testing

---

# System Architecture

```
                User Question
                      │
                      ▼
                 FastAPI API
                      │
                      ▼
            Input Guardrails
                      │
                      ▼
        Sentence Transformer
             Embedding Model
                      │
                      ▼
      PostgreSQL + PGVector Search
                      │
          Top Relevant Chunks
                      │
                      ▼
           Prompt Construction
                      │
                      ▼
              OpenAI GPT Model
                      │
                      ▼
          Output Guardrails
                      │
                      ▼
             Final Response
```

---

# Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| API Framework | FastAPI |
| LLM | OpenAI GPT-5 Nano |
| Embeddings | Sentence Transformers |
| Vector Database | PostgreSQL + PGVector |
| Database Driver | psycopg2 |
| Containerization | Docker |
| API Testing | Swagger UI |
| Version Control | Git & GitHub |

---

# Project Structure

```
KnowledgePilot_AI/

├── api/
│   └── main.py
│
├── src/
│   ├── embeddings/
│   ├── preprocessing/
│   ├── retrieval/
│   ├── generation/
│   ├── pipeline/
│   ├── guardrails/
│   ├── memory/
│   ├── ingestion/
│   ├── logging/
│   └── utils/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# How It Works

### 1. Data Collection

HTML lecture transcripts are collected and stored.

### 2. Parsing

HTML files are parsed to extract clean text.

### 3. Chunking

Each lecture is divided into manageable text chunks.

### 4. Embedding Generation

Each chunk is converted into a dense vector using Sentence Transformers.

### 5. Vector Storage

Embeddings are stored inside PostgreSQL using the PGVector extension.

### 6. Semantic Retrieval

For every user question:

- Generate embedding
- Search nearest vectors
- Retrieve Top-K relevant chunks

### 7. Prompt Building

Retrieved context and conversation history are combined into a prompt.

### 8. Answer Generation

OpenAI GPT generates a grounded response based on retrieved information.

### 9. Guardrails

The response passes through:

- Input validation
- Retrieval validation
- Output validation

before being returned.

---

# API

## POST `/ask`

Example Request

```json
{
    "question":"What is cosine similarity?"
}
```

Example Response

```json
{
    "answer":"Cosine similarity measures the angle between two vectors...",
    "sources":[
        {
            "lecture":"Lecture 05",
            "chunk_id":17
        }
    ]
}
```

---

# Running Locally

Clone repository

```bash
git clone https://github.com/MANVIS10/knowledgepilot_AI.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Start PostgreSQL

```bash
docker start knowledgepilot-db
```

Run FastAPI

```bash
uvicorn api.main:app --reload
```

Open Swagger

```
http://localhost:8000/docs
```

---

# Docker

Build image

```bash
docker build -t knowledgepilot .
```

Run container

```bash
docker run --env-file .env -p 8000:8000 knowledgepilot
```

---

# Testing

The project includes tests for:

- Retrieval pipeline
- Knowledge Base
- Generator

Run tests

```bash
pytest
```

---

# Current Capabilities

✔ Semantic Retrieval

✔ Vector Search

✔ Context-Aware Responses

✔ Streaming Generation

✔ Conversation Memory

✔ Guardrails

✔ FastAPI Backend

✔ Dockerized Deployment

✔ Swagger Documentation

---

# Future Improvements

- Hybrid Search (BM25 + Vector Search)
- Cross-Encoder Re-ranking
- Redis Response Caching
- User Authentication
- CI/CD Pipeline
- Cloud Deployment
- Monitoring and Metrics

---

# Learning Outcomes

This project helped me gain practical experience with:

- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Vector Databases
- Embedding Models
- Semantic Search
- FastAPI
- PostgreSQL + PGVector
- Docker
- REST API Development
- Modular Python Project Design
- Git & GitHub Workflow

---

# Author

**Manvi Soni**

GitHub: https://github.com/MANVIS10
