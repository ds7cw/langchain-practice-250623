import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Constants
BOOKS_DIR = "books"
ODYSSEY_BOOK = "odyssey.txt"
DB_STR = "db"
CHROMA_DB_STR = "chroma_db"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, BOOKS_DIR, ODYSSEY_BOOK)
persistent_dir = os.path.join(current_dir, DB_STR, CHROMA_DB_STR)

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "The file {} does not exist. Please check the path.".format(file_path)
        )

    # Read the text content from the file
    loader = TextLoader(file_path=file_path)
    documents = loader.load()

    # Split the document into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents=documents)

    print("\n--- Document Chunks Information ---")
    print("Number of document chunks: {}".format(len(chunks)))
    print("Sample chunk: \n{}\n".format(chunks[0].page_content))

    # Create embeddings
    print("--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(
        model=TEXT_EMBEDDING_3_SMALL
    ) # Update model, as/ if required
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=persistent_dir
    )
    print("\n--- Finished creating vector store ---")

else:
    print("Vectore store already exists. No need initialize.")
