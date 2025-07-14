import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Constants
DB_STR = "db"
CHROMA_DB_WITH_METADATA_STR = "chroma_db_with_metadata"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
SIMILARITY_SEARCH_TYPE_STR = "similarity"
MMR_SEARCH_TYPE_STR = "mmr"
SIMILARITY_SCORE_THRESHOLD_TYPE_STR = "similarity_score_threshold"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, DB_STR)
persistent_dir = os.path.join(db_dir, CHROMA_DB_WITH_METADATA_STR)

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)


def query_vector_store(store_name, query, embedding_func, search_type, search_kwargs):
    """Function to query a vector store with different search types and parameters"""
    if os.path.exists(persistent_dir):
        print("\n--- Querying the Vector Store {} ---".format(store_name))
        db = Chroma(persist_directory=persistent_dir, embedding_function=embedding_func)

        retriever = db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        relevant_docs = retriever.invoke(query)
        print("\n--- Relevant Documents for {} ---".format(store_name))
        for i, doc in enumerate(relevant_docs, 1):
            print("Document {}:\n{}\n".format(i, doc.page_content))
            if doc.metadata:
                print("Source: {}\n".format(doc.metadata.get("source", "Unknown")))

    else:
        print("Vector store {} does not exist.".format(store_name))


query = "How did Juliet die?"

# Different retrieval methods
# 1. Similarity Search
# Retrieves based on vector similarity.
# It finds the most similar documents to the query vector based on cosine similarity.
# Use this when you want to retrieve the top k most similar documents.
print("\n--- Using Similarity Search ---")
query_vector_store(store_name=CHROMA_DB_WITH_METADATA_STR, query=query, embedding_func=embeddings,
    search_type=SIMILARITY_SEARCH_TYPE_STR, search_kwargs={"k": 3})

# 2. Max Marginal Relevance (MMR)
# Balances between selecting documents that are relevant to the query and diverse among themselves.
# 'fetch_k' specifies the number of documents to initially fetch based on similarity.
# 'lambda_mult' controls the diversity of the results: 1 for minimum diversity, 0 for maximum.
# Use this when you want to avoid redundancy and retrieve diverse yet relevant documents.
# Note: Relevance measures how closely documents match the query.
# Note: Diversity ensures that the retrieved documents are not too similar to each other, providing a broader range of info.
print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(store_name=CHROMA_DB_WITH_METADATA_STR, query=query, embedding_func=embeddings,
    search_type=MMR_SEARCH_TYPE_STR, search_kwargs={"k": 3, "lambda_mult": 0.5})

# 3. Similarity Score Threshold
# This method retrieves documents that exceed a certain similarity score threshold.
# 'score_threshold' sets the minimum similarity score a document must have to be considered relevant.
# Use this when you want to ensure that only highly relevant documents are retrieved, filtering out less relevant ones.
print("\n--- Using Similarity Score Threshold ---")
query_vector_store(store_name=CHROMA_DB_WITH_METADATA_STR, query=query, embedding_func=embeddings,
    search_type=SIMILARITY_SCORE_THRESHOLD_TYPE_STR, search_kwargs={"k": 3, "score_threshold": 0.1})
