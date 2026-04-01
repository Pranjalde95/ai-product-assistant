from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def create_vector_store(documents):
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(documents, embedding_model)
    return vector_store


if __name__ == "__main__":
    sample_docs = [
        Document(
            page_content="Product ID: B001\nRating: 5\nSummary: Excellent product\nReview: I loved this product, works perfectly!",
            metadata={"product_id": "B001", "rating": 5}
        ),
        Document(
            page_content="Product ID: B002\nRating: 1\nSummary: Very bad\nReview: Waste of money, completely disappointed.",
            metadata={"product_id": "B002", "rating": 1}
        ),
        Document(
            page_content="Product ID: B003\nRating: 4\nSummary: Good quality\nReview: Pretty decent product for the price.",
            metadata={"product_id": "B003", "rating": 4}
        ),
        Document(
            page_content="Product ID: B004\nRating: 2\nSummary: Not great\nReview: The quality could be much better.",
            metadata={"product_id": "B004", "rating": 2}
        ),
    ]

    vector_store = create_vector_store(sample_docs)

    vector_store.save_local("faiss_index")

    print("\nVector store created and saved successfully!")

    query = "good quality product"
    results = vector_store.similarity_search(query, k=2)

    print("\nSearch Results:")
    for res in results:
        print(res.page_content)
        print("Metadata:", res.metadata)
        print()