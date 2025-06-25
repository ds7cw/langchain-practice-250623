from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

chat_history = []

system_message = SystemMessage(content="You are an all-purpose AI assistant")
chat_history.append(system_message)

# Chat loop
while True:
    query = input("User ('exit' to quit): ")
    if query.lower == "exit":
        break
    chat_history.append(HumanMessage(content=query))

    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print("AI: {}".format(response))

print("---- Message History ----")
print(chat_history)
