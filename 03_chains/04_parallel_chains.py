from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")
SYSTEM_MSG_EXPERT_REVIEWER = "You are an expert product reviewer."

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_MSG_EXPERT_REVIEWER),
        ("human", "List the main features of the product {product_name}."),
    ]
)


def analyze_pros(features):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MSG_EXPERT_REVIEWER),
            ("human", "Given these features: {features}, list the pros of these features."),
        ]
    )
    return pros_template.format_prompt(features=features)


def analyze_cons(features):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_MSG_EXPERT_REVIEWER),
            ("human", "Given these features: {features}, list the cons of these features."),
        ]
    )
    return cons_template.format_prompt(features=features)


def combine_pros_and_cons(pros, cons):
    return "Pros:\n{}\n\nCons:\n{}".format(pros, cons)


# Simplify branch with LCEL
pros_branch_chain = (
    RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
)

cons_branch_chain = (
    RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()
)

# Create combined chain using LCEL
chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(branches={"pros": pros_branch_chain, "cons": cons_branch_chain})
    | RunnableLambda(lambda x: combine_pros_and_cons(x["branches"]["pros"], x["branches"]["cons"]))
)

result = chain.invoke({"product_name": "MacBook Pro"})

print(result)
