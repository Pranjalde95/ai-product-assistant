# AI Product Assistant 🚀

## 📌 Overview

This project is a prototype of an AI-powered Product Assistant that can answer user product-related queries.

The system was developed in two parts:

* **Core AI Logic (Local Environment)**
* **Cloud Deployment (AWS EC2 with FastAPI)**

---

## 🎯 Objective

To build and deploy a backend system that can:

* Accept user queries
* Process them through an AI-based pipeline
* Return meaningful responses
* Demonstrate real-world deployment using AWS

---

## 🧠 Core AI Implementation (Local)

The assistant was initially developed locally with:

* Data preprocessing of product reviews
* Document creation pipeline
* Vector-based retrieval system
* Query-response pipeline

Note: The full AI pipeline requires higher compute and is demonstrated locally.

---

## 🌐 AWS Deployment (Cloud)

A lightweight version of the API was deployed on AWS EC2 using:

* FastAPI
* Uvicorn
* EC2 instance (Ubuntu)

### Features:

* Public API endpoint
* Interactive Swagger UI (`/docs`)
* Query-response system

---

## ⚙️ Tech Stack

* Python
* FastAPI
* Uvicorn
* FAISS (local)
* Sentence Transformers (local)
* AWS EC2

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/Pranjalde95/ai-product-assistant.git
cd ai-product-assistant

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn app:app --reload
```

---

## 🌍 API Endpoints

### GET /

Returns API status

### POST /ask

Example:

```json
{
  "query": "Best laptop for battery life?"
}
```

---

## ⚠️ Limitations

* Full AI pipeline not deployed on AWS due to free-tier resource limits
* Lightweight version used for cloud deployment

---

## 📈 Future Improvements

* Full RAG integration on the cloud
* Frontend UI (Streamlit / React)
* Auto-deployment pipeline
* Scalable architecture

---

## ⭐ Key Highlight

This project demonstrates:

* AI system design
* Backend API development
* Cloud deployment (AWS)
* Real-world engineering workflow
