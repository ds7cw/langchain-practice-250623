import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Constants
BOOKS_DIR_STR = "books"
DB_STR = "db"
CHROMA_DB_WITH_METADATA_STR = "chroma_db_with_metadata"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"

current_dir = os.path.dirname(os.path.abspath(__file__))
books_dir = os.path.join(current_dir, BOOKS_DIR_STR)
db_dir = os.path.join(current_dir, DB_STR)
persistent_dir = os.path.join(db_dir, CHROMA_DB_WITH_METADATA_STR)

print("Books directory: {}".format(books_dir))
print("Persistent directory: {}".format(persistent_dir))

if not os.path.exists(persistent_dir):
    print("Persistent directory does not exist. Initializing vector store...")
    if not os.path.exists(books_dir):
        raise FileNotFoundError(
            "The directory {} does not exist. Please check the path.".format(books_dir)
        )

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    # Read the text content from each file and store it with metadata
    documents = []

    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()
        for doc in book_docs:
            # Add metadata to each document indicating its source
            doc.metadata = {"source": book_file}
            documents.append(doc)

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents=documents)

    print("\n--- Document Chunks Information ---")
    print("Number of document chunks: {}".format(len(docs)))

    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)
    print("\n--- Embeddings created ---")

    # Create the vector store and persist it
    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persistent_dir
    )
    print("\n--- Finished creating and persisting vector store")

else:
    print("Vector store already exists. No need to initialize.")
