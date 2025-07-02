from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o")
SYSTEM_MSG_HELPFUL_ASSISTANT = "You are a helpful assistant."
SYSTEM_KEY = "system"
HUMAN_KEY = "human"
POSITIVE = "positive"
NEGATIVE = "negative"
NEUTRAL = "neutral"

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_KEY, SYSTEM_MSG_HELPFUL_ASSISTANT),
        (HUMAN_KEY, "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_KEY, SYSTEM_MSG_HELPFUL_ASSISTANT),
        (HUMAN_KEY, "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_KEY, SYSTEM_MSG_HELPFUL_ASSISTANT),
        (HUMAN_KEY, "Generate a request for more details for this neutral feedback: {feedback}."),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_KEY, SYSTEM_MSG_HELPFUL_ASSISTANT),
        (HUMAN_KEY, "Generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        (SYSTEM_KEY, SYSTEM_MSG_HELPFUL_ASSISTANT),
        (HUMAN_KEY, "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.")
    ] 
)

# Define the runnable branches for handling feedback
branches = RunnableBranch(
    (
        lambda x: POSITIVE in x,
        positive_feedback_template | model | StrOutputParser() # Positive feedback chain
    ),
    (
        lambda x: NEGATIVE in x,
        negative_feedback_template | model | StrOutputParser() # Negative feedback chain
    ),
    (
        lambda x: NEUTRAL in x,
        neutral_feedback_template | model | StrOutputParser() # Neutral feedback chain
    ),
    escalate_feedback_template | model | StrOutputParser() # Default case
)

# Create classification chain
classification_chain = classification_template | model | StrOutputParser()

# Combine classification and response generation into one chain
chain = classification_chain | branches

good_review = "The product is excellent. I really enjoyed using it and found it very helpful."
bad_review = "The product is terrible. It broke after just one use and the quality is very poor."
neutral_review = "The product is okay. It works as expected but nothing exceptional."
default_review = "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

result = chain.invoke({"feedback": good_review})

print(result)
