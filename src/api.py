from fastapi import FastAPI
from pydantic import BaseModel
import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


print("Loading FAISS index...")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})

print("Using Qwen3 4B via Ollama...")
print("API ready")


def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an intelligent AI product assistant.

Based ONLY on the reviews provided in the context below, answer the following question.

If the reviews are mixed, mention both positive and negative aspects.
If the reviews are mostly positive, highlight the strengths.
If the reviews are mostly negative, point out the weaknesses.

Give a natural, human-like answer in 2-3 sentences.

Question:
{query}

Context:
{context}

Final Answer:
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"].strip()

    if "Final Answer:" in answer:
        answer = answer.split("Final Answer:")[-1].strip()

    return answer


@app.get("/")
def home():
    return {"message": "AI Product Assistant API is running"}


@app.post("/ask")
def ask_question(request: QueryRequest):
    query = request.query

    retrieved_docs = retriever.invoke(query)
    answer = generate_answer(query, retrieved_docs)

    return {
        "query": query,
        "answer": answer
    }