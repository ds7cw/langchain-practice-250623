from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

load_dotenv()

# Constants
STRUCTURED_CHAT_AGENT_STR = "hwchase17/structured-chat-agent"
GPT_4O_MODEL_STR = "gpt-4o"


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p")


def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I could not find any information on that."


tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time.",
    ),
    Tool(
        name="Wikipedia",
        func=search_wikipedia,
        description="Useful for when you need to find information about a topic.",
    ),
]

prompt = hub.pull(owner_repo_commit=STRUCTURED_CHAT_AGENT_STR)
llm = ChatOpenAI(model=GPT_4O_MODEL_STR)

# ConversationBufferMemory stores conversation history, allowing the agent to maintain context across interactions
memory = ConversationBufferMemory()

# Combined the language model & the prompt to create an interactive agent
agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

# Responsible for managing the interaction between the user input, the agent, and the tools
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)

initial_message = "You are an AI assistant that can provide helpful answers using available tools." \
    "\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

# Chat loop
while True:
    user_input = input("User (enter 'exit' to quit): ")
    if user_input.lower() == "exit":
        break

    memory.chat_memory.add_message(HumanMessage(content=user_input))
    response = agent_executor.invoke(input={"input": user_input})
    print("Bot:", response["output"])
    memory.chat_memory.add_ai_message(AIMessage(content=response["output"]))
