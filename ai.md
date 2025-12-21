# AI Context Document: Local RAG Chatbot on Kubernetes

## 1. Purpose
This project implements a fully self‑hosted Retrieval‑Augmented Generation (RAG) system designed to answer questions strictly based on a private corpus of documentation. The system runs inside a Kubernetes cluster, uses a vector database for semantic search, and exposes a clean API for querying the knowledge base.

The primary goals are:
- Provide natural‑language access to internal documentation  
- Ensure all answers are grounded in retrieved content  
- Prevent hallucinations and external knowledge leakage  
- Keep all data and computation inside the operator’s infrastructure  
- Support scalable, containerized deployment  

---

## 2. Technology Stack

### Core Technologies
- **Python 3.11** – Primary language for the RAG engine and API  
- **FastAPI** – REST API layer for serving chat requests  
- **LangChain or LlamaIndex** – RAG orchestration, retrieval logic, prompt construction  
- **SentenceTransformers** – Local embedding model (e.g., `all-MiniLM-L6-v2`)  
- **Qdrant** – Vector database for semantic search (running as a StatefulSet in Kubernetes)  
- **Local LLM (optional)** – e.g., `llama.cpp` for fully offline inference 

### Infrastructure
- **Kubernetes (K8s)** – Orchestration platform for all services  
- **Docker** – Containerization for the Python RAG service  
- **PersistentVolumeClaims (PVCs)** – Storage for Qdrant collections  
- **ClusterIP Services** – Internal service discovery  
- **Ingress Controller (optional)** – External access to the chatbot API or UI  

### Optional Components
- **Streamlit or React** – Frontend UI for interacting with the chatbot  
- **Helm** – Packaging and deployment automation  
- **Prometheus/Grafana** – Observability and metrics  

---

## 3. High‑Level Architecture
