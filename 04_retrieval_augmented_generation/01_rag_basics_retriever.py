import os

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Constants
DB_STR = "db"
CHROMA_DB_STR = "chroma_db"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, DB_STR, CHROMA_DB_STR)

# Define the embedding model
embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)

# Load the existing vector store with the embedding function
db = Chroma(
    persist_directory=persistent_dir,
    embedding_function=embeddings,
)

query = "Who is Odysseus' wife?"

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.9},
    # k:3 returns the top 3 closest results
    # 0.9 threshold is too strict, consider a lower value
)

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print("Document {}:\n{}\n".format(i, doc.page_content))
    if doc.metadata:
        print("Source: {}\n".format(doc.metadata.get("source", "Unknown")))
