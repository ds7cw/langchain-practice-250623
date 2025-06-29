from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

# Create ChatPromptTemplate using a template string
template = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template=template)

prompt = prompt_template.invoke({"topic": "lawyers"})
result = model.invoke(prompt)

# Prompt with Multiple Placeholders
template_multiple = """You are a helpful assistant.
Human: Tell me a {adjective} short story about a {animal}.
Assistant:"""
prompt_multiple = ChatPromptTemplate.from_template(template_multiple)
prompt = prompt_multiple.invoke({"adjective": "funny", "animal": "panda"})

result = model.invoke(prompt)
print("Response: {}".format(result.content))

# Prompt with System and Human Messages (Using Tuples)
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
result = model.invoke(prompt)
print("Response: {}".format(result.content))
