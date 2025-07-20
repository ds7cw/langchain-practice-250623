import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Constants
DB_STR = "db"
CHROMA_DB_FIRECRAWL_STR = "chroma_db_firecrawl"
FIRECRAWL_API_KEY_STR = "FIRECRAWL_API_KEY"
APPLE_URL_STR = "https://ww.apple.com/"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
SIMILARITY_SEARCH_TYPE_STR = "similarity"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, DB_STR)
persistent_dir = os.path.join(db_dir, CHROMA_DB_FIRECRAWL_STR)


def create_vector_store():
    """Crawl website, split content, create embeddings, and persist the vector store."""
    api_key = os.getenv(FIRECRAWL_API_KEY_STR)
    if not api_key:
        raise ValueError("{} environment variable not set.".format(FIRECRAWL_API_KEY_STR))

    loader = FireCrawlLoader(api_key=api_key, url=APPLE_URL_STR, mode="scrape")
    docs = loader.load()

    # Convert metadata values to strings if they are lists
    for doc in docs:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(docs)

    print("\n--- Document Chunks Information ---")
    print("Number of document chunks: {}".format(len(split_docs)))
    print("Sample chunk:\n{}\n".format(split_docs[0].page_content))

    embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)

    db = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory=persistent_dir)


if not os.path.exists(persistent_dir):
    create_vector_store()
else:
    print("Vector store {} already exists. No need to initialize.".format(persistent_dir))

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)
db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)


def query_vector_store(query: str) -> None:
    """Query the vector store with the specified question."""
    retriever = db.as_retriever(search_type=SIMILARITY_SEARCH_TYPE_STR, search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)
    for i, doc in enumerate(relevant_docs, 1):
        print("Document {}:\n{}\n".format(i, doc.page_content))
        if doc.metadata:
            print("Source: {}\n".format(doc.metadata.get("source", "Unknown")))


query = "Apple Intelligence?"

query_vector_store(query=query)
