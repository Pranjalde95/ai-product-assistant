import ollama

from data_loader import load_reviews_data, clean_reviews_data, create_documents
from vector_store import create_vector_store


def build_retriever():
    file_path = "data/Reviews.csv"

    df = load_reviews_data(file_path)
    cleaned_df = clean_reviews_data(df)

    # small sample for testing
    sample_df = cleaned_df.sample(n=20, random_state=42)

    documents = create_documents(sample_df)

    vector_store = create_vector_store(documents)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever


def generate_answer(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
You are an AI product assistant.

Use the review context below to answer the user's question clearly and briefly.

If the reviews are mixed, mention both positive and negative points.

Question:
{query}

Context:
{context}

Answer:
"""

    response = ollama.chat(
        model="qwen3:4b",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]


if __name__ == "__main__":
    query = "Is this product good value for money?"

    print("Building retriever...")
    retriever = build_retriever()

    print("Retrieving documents...")
    retrieved_docs = retriever.invoke(query)

    print("Generating final answer with Qwen...")
    answer = generate_answer(query, retrieved_docs)

    print("\nRetrieved Documents:\n")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print()

    print("\nFinal Answer:\n")
    print(answer)