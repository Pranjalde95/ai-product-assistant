import pandas as pd
from langchain_core.documents import Document


def load_reviews_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_reviews_data(df: pd.DataFrame) -> pd.DataFrame:
    useful_columns = ["ProductId", "Score", "Summary", "Text"]
    df = df[useful_columns].copy()

    df = df.dropna(subset=useful_columns)
    df = df.drop_duplicates(subset=useful_columns)
    df = df.reset_index(drop=True)

    return df


def create_review_document(row: pd.Series) -> str:
    review_text = row["Text"][:300]

    return (
        f"Product ID: {row['ProductId']}\n"
        f"Rating: {row['Score']}\n"
        f"Summary: {row['Summary']}\n"
        f"Review: {review_text}"
    )


def create_documents(df: pd.DataFrame) -> list:
    documents = []

    for _, row in df.iterrows():
        text = create_review_document(row)

        doc = Document(
            page_content=text,
            metadata={
                "product_id": row["ProductId"],
                "rating": row["Score"]
            }
        )

        documents.append(doc)

    return documents


if __name__ == "__main__":
    file_path = "data/Reviews.csv"

    df = load_reviews_data(file_path)
    cleaned_df = clean_reviews_data(df)

    sample_df = cleaned_df.sample(n=500, random_state=42)

    documents = create_documents(sample_df)

    print("Original shape:", df.shape)
    print("Cleaned shape:", cleaned_df.shape)

    print("\nSample rows:")
    print(sample_df.head())

    print("\nFirst 3 documents:\n")
    for i, doc in enumerate(documents[:3], 1):
        print(f"--- Document {i} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)
        print()
