import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Constants
DB_STR = "db"
CHROMA_DB_WITH_METADATA_STR = "chroma_db_with_metadata"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
SIMILARITY_SEARCH_TYPE_STR = "similarity"
GPT_4_MODEL_STR = "gpt-4o"

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, DB_STR, CHROMA_DB_WITH_METADATA_STR)

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)
db = Chroma(embedding_function=embeddings, persist_directory=persistent_dir)

query = "How can I learn more about LangChain?"

retriever = db.as_retriever(
    search_type=SIMILARITY_SEARCH_TYPE_STR,
    search_kwargs={"k": 1},
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print("Document {}:\n{}\n".format(i, doc.page_content))

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents."
    + "If the answer is not found in the documents, respond with 'I'm not sure'."
)

model = ChatOpenAI(model=GPT_4_MODEL_STR)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

result = model.invoke(messages)

print("\n--- Generated Response ---")
print("Full result:\n{}\n".format(result))
print("Content only:\n{}\n".format(result.content))
