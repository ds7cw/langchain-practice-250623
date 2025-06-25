from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]

# OpenAI
model = ChatOpenAI(model="gpt-4o")

result = model.invoke(messages)
print("OpenAI response: {}".format(result.content))

# Anthropic
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print("Anthropic response: {}".format(result.content))

# Google
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print("Google response: {}".format(result.content))
