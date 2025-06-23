# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Invoke the model with a question
result = model.invoke("How long is a quarter in a game of basketball according to NBA rules?")
print("Full result (model.invoke(QUESTION)):")
print(result)
print("Content only (model.invoke(QUESTION).content):")
print(result.content)
