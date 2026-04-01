from data_loader import load_reviews_data, clean_reviews_data, create_documents
from vector_store import create_vector_store


def build_retriever():
    file_path = "data/Reviews.csv"

    df = load_reviews_data(file_path)
    cleaned_df = clean_reviews_data(df)

    # use small sample for testing
    sample_df = cleaned_df.sample(n=500, random_state=42)

    documents = create_documents(sample_df)

    vector_store = create_vector_store(documents)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    return retriever


if __name__ == "__main__":
    retriever = build_retriever()

    query = "good quality product"
    results = retriever.invoke(query)

    print("\nRetriever Results:\n")
    for i, doc in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(doc.page_content)
        print()