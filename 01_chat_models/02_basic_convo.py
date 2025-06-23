from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

# SystemMessage:
#  Sets the context: It tells the AI what kind of assistant it's supposed to be (e.g., “You are a helpful financial advisor”).
#  Guides tone and style: You can instruct the AI to be formal, funny, brief, detailed, etc.
#  Frames behavior: It can limit or enhance the assistant’s capabilities (like “Do not write code” or “Always speak in pirate slang”).
# HumanMessage:
#  Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke model with a list of messages
result = model.invoke(messages)
print("Answer from AI model: {}".format(result.content))

# AIMessage:
# Message from an AI.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."),
    HumanMessage(content="What is 10 times 5?"),
]

# Invoke model with a conversation (human question and AI answers)
result = model.invoke(messages)
print("Answer from AI model: {}".format(result.content))
