import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

# Constants
DB_STR = "db"
FOUR_RAG_STR = "4_rag"
CHROMA_DB_WITH_METADATA_STR = "chroma_db_with_metadata"
TEXT_EMBEDDING_3_SMALL = "text-embedding-3-small"
GPT_4O_MODEL_STR = "gpt-4o"
REACT_OWNER_REPO_STR = "hwchase17/react"
CHAT_HISTORY_STR = "chat_history"

# Load the existing Chroma vector store
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "..", "..", FOUR_RAG_STR, DB_STR)
persistent_dir = os.path.join(db_dir, CHROMA_DB_WITH_METADATA_STR)

# Check if the Chroma vector store already exists
if os.path.exists(persistent_dir):
    print("Loading existing vector store...")
    db = Chroma(persist_directory=persistent_dir,
                embedding_function=None)
else:
    raise FileNotFoundError(
        "The directory {} does not exist. Please check the path.".format(persistent_dir)
    )

embeddings = OpenAIEmbeddings(model=TEXT_EMBEDDING_3_SMALL)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_dir,
            embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model=GPT_4O_MODEL_STR)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(CHAT_HISTORY_STR),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(CHAT_HISTORY_STR),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(
    history_aware_retriever, question_answer_chain)


# Set Up ReAct Agent with Document Store Retriever
# Load the ReAct Docstore Prompt
react_docstore_prompt = hub.pull(REACT_OWNER_REPO_STR)

tools = [
    Tool(
        name="Answer Question",
        func=lambda input, **kwargs: rag_chain.invoke(
            {"input": input, CHAT_HISTORY_STR: kwargs.get(CHAT_HISTORY_STR, [])}
        ),
        description="useful for when you need to answer questions about the context",
    )
]

# Create the ReAct Agent with document store retriever
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_docstore_prompt,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, handle_parsing_errors=True, verbose=True,
)

chat_history = []
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    response = agent_executor.invoke(
        {"input": query, CHAT_HISTORY_STR: chat_history})
    print(f"AI: {response['output']}")

    # Update history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=response["output"]))
