import os

from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# Constants
BOOKS_DIR_STR = "books"
DB_STR = "db"
ROMEO_AND_JULIET_BOOK_STR = "romeo_and_juliet.txt"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
CHROMA_DB_CHAR_STR = "chroma_db_char"
CHROMA_DB_SENT_STR = "chroma_db_sent"
CHROMA_DB_TOKEN_STR = "chroma_db_token"
CHROMA_DB_REC_CHAR_STR = "chroma_db_rec_char"
CHROMA_DB_CUSTOM_STR = "chroma_db_custom"

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, BOOKS_DIR_STR, ROMEO_AND_JULIET_BOOK_STR)
db_dir = os.path.join(current_dir, DB_STR)

if not os.path.exists(file_path):
    raise FileNotFoundError("The file {} does not exist. Please check the path".format(file_path))

loader = TextLoader(file_path=file_path)
documents = loader.load()

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL,)

def create_vector_store(docs, store_name):
    persistent_dir = os.path.join(db_dir, store_name)
    if not os.path.exists(persistent_dir):
        print("\n--- Creating vector store {} ---".format(store_name))
        db = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persistent_dir)
        print("--- Finished creating vector store {} ---".format(store_name))
    else:
        print("Vector store {} already exists. No need to initialize.".format(store_name))


# 1. Character-based Splitting
print("\n--- Using Character-based Splitting ---")
char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(documents=documents)
create_vector_store(char_docs, CHROMA_DB_CHAR_STR)

# 2. Sentence-based Splitting
# Ideal for maintaining semantic coherence within chunks.
print("\n--- Using Sentence-based Splitting ---")
sent_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sent_docs = sent_splitter.split_documents(documents=documents)
create_vector_store(sent_docs, CHROMA_DB_SENT_STR)

# 3. Token-based Splitting
# Splits text into chunks based on tokens (words or subwords), using tokenizers like GPT-2.
# Useful for transformer models with strict token limits.
print("\n--- Using Token-based Splitting ---")
token_splitter = TokenTextSplitter(chunk_overlap=0, chunk_size=512)
token_docs = token_splitter.split_documents(documents=documents)
create_vector_store(token_docs, CHROMA_DB_TOKEN_STR)

# 4. Recursive Character-based Splitting
# Attempts to split text at natural boundaries (sentences, paragraphs) within character limit.
# Balances between maintaining coherence and adhering to character limits.
print("\n--- Using Recursive Character-based Splitting ---")
rec_char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
rec_char_docs = rec_char_splitter.split_documents(documents=documents)
create_vector_store(rec_char_docs, CHROMA_DB_REC_CHAR_STR)

# 5. Custom Splitting
# Allows creating custom splitting logic based on specific requirements.
# Useful for documents with unique structure that standard splitters can't handle.
print("\n--- Using Custom Splitting ---")


class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        # Custom logic for splitting text
        return text.split("\n\n") # Splitting by paragraphs


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(documents=documents)
create_vector_store(custom_docs, CHROMA_DB_CUSTOM_STR)


# Function to query a vector store
def query_vector_store(store_name, query):
    persistent_dir = os.path.join(db_dir, store_name)
    if os.path.exists(persistent_dir):
        print("\n--- Querying the Vector Store {} ---".format(store_name))
        db = Chroma(persist_directory=persistent_dir, embedding_function=embeddings,)
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)

        print("\n--- Relevant Documents for {} ---".format(store_name))
        for i, doc in enumerate(relevant_docs, 1):
            print("Document {}:\n{}\n".format(i, doc.page_content))
            if doc.metadata:
                print("Source: {}\n".format(doc.metadata.get('source', 'Unknown')))
    else:
        print("Vector store {} does not exist.".format(store_name))


query = "How did Juliet die?"

query_vector_store(CHROMA_DB_CHAR_STR, query)
query_vector_store(CHROMA_DB_SENT_STR, query)
query_vector_store(CHROMA_DB_TOKEN_STR, query)
query_vector_store(CHROMA_DB_REC_CHAR_STR, query)
query_vector_store(CHROMA_DB_CUSTOM_STR, query)
