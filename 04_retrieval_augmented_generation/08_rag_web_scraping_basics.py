import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Constants
DB_STR = "db"
CHROMA_DB_APPLE_STR = "chroma_db_apple"
APPLE_URL_STR = "https://ww.apple.com/"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
SIMILARITY_SEARCH_TYPE_STR = "similarity"

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, DB_STR)
persistent_dir = os.path.join(db_dir, CHROMA_DB_APPLE_STR)

# Step 1: Scrape the content from apple.com using WebBaseLoader
urls = [APPLE_URL_STR]
loader = WebBaseLoader(urls)
documents = loader.load()

# Step 2: Split the scraped content into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents=documents)

print("\n--- Document Chunks Information ---")
print("Number of document chunks: {}".format(len(docs)))
print("Sample chunk:\n{}\n".format(docs[0].page_content))

# Step 3: Create embeddings for the document chunks
embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)

# Step 4: Create and persist the vector store with embeddings
if not os.path.exists(persistent_dir):
    print("\n--- Creating vector store {} ---".format(persistent_dir))
    db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persistent_dir)
    print("--- Finished creating vector store in {} ---".format(persistent_dir))
else:
    print("Vector store {} already exists. No need to initialize".format(persistent_dir))
    db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings)

# Step 5: Query the vector store
retriever = db.as_retriever(search_type=SIMILARITY_SEARCH_TYPE_STR, search_kwargs={"k": 3})

query = "What new products are announced on apple.com?"
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print("Document {}:\n{}\n".format(i, doc.page_content))
    if doc.metadata:
        print("Source: {}\n".format(doc.metadata.get("source", "Unknown")))
