import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Constants
BOOKS_DIR_STR = "books"
ODYSSEY_BOOK_STR = "odyssey.txt"
DB_STR = "db"
OPENAI_TEXT_EMBEDDING_ADA_002_STR = "text-embedding-ada-002"
CHROMA_DB_OPENAI_STR = "chroma_db_openai"
HUGGINGFACE_MODEL_STR  = "sentence-transformers/all-mpnet-base-v2"
CHROMA_DB_HUGGINGFACE_STR = "chroma_db_huggingface"
# TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
# CHROMA_DB_CHAR_STR = "chroma_db_char"
# CHROMA_DB_SENT_STR = "chroma_db_sent"
# CHROMA_DB_TOKEN_STR = "chroma_db_token"
# CHROMA_DB_REC_CHAR_STR = "chroma_db_rec_char"
# CHROMA_DB_CUSTOM_STR = "chroma_db_custom"

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, BOOKS_DIR_STR, ODYSSEY_BOOK_STR)
db_dir = os.path.join(current_dir, DB_STR)

if not os.path.exists(file_path):
    raise FileNotFoundError("The file {} does not exist. Please check the path".format(file_path))

loader = TextLoader(file_path=file_path)
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents=documents)

print("\n--- Document Chunks Information ---")
print("Number of document chunks: {}".format(len(docs)))
print("Sample chunk: \n{}\n".format(docs[0].page_content))

def create_vector_store(docs, embeddings, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print("\n--- Creating vector store {} ---".format(store_name))
        Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persistent_dir)
        print("\n--- Finished creating vector store {} ---".format(store_name))
    else:
        print("\n--- Vector store {} ---".format(store_name))


# 1. OpenAI Embeddings
# Useful for general-purpose embeddings with high accuracy.
# Cost of using OpenAI embeddings will depend on your OpenAI API usage and pricing plan.
print("\n--- Using OpenAI Embeddings ---")
openai_embeddings = OpenAIEmbeddings(model=OPENAI_TEXT_EMBEDDING_ADA_002_STR)
create_vector_store(docs=docs, embeddings=openai_embeddings, store_name=CHROMA_DB_OPENAI_STR)

# 2. Hugging Face Transformers
# Uses models from the Hugging Face library. Ideal for leveraging a wide variety of models for different tasks.
# Running Hugging Face models locally on your PC incurs no direct cost other than using computational resources.
print("\n--- Using Hugging Face Transformers ---")
huggingface_embeddings = HuggingFaceEmbeddings(model=HUGGINGFACE_MODEL_STR)
create_vector_store(docs=docs, embeddings=huggingface_embeddings, store_name=CHROMA_DB_HUGGINGFACE_STR)

print("Embedding demonstrations for OpenAI and Hugging Face completed.")


def query_vector_store(store_name, query, embedding_function):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print("\n--- Querying the Vector Store {} ---".format(store_name))
        db = Chroma(persist_directory=persistent_dir, embedding_function=embedding_function)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.1},)
        relevant_docs = retriever.invoke(query)
        print("\n--- Relevant Documents for {} ---".format(store_name))
        for i, doc in enumerate(relevant_docs, 1):
            print("Document {}:\n{}\n".format(i, doc.page_content))
            if doc.metadata:
                print("Source: {}\n".format(doc.metadata.get("source", "Unknown")))
    else:
        print("Vector store {} does not exist.".format(store_name))


query = "Who is Odysseus' wife?"

query_vector_store(store_name=CHROMA_DB_OPENAI_STR, query=query, embedding_function=openai_embeddings)
query_vector_store(store_name=CHROMA_DB_HUGGINGFACE_STR, query=query, embedding_function=huggingface_embeddings)

print("Querying demonstrations completed.")
