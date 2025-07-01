from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comdeian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)

# Define additional processing steps
uppercase_output = RunnableLambda(lambda x: x.upper())
count_words = RunnableLambda(lambda x: "Word count: {}\n".format(len(x.split())))

chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

result = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(result)
