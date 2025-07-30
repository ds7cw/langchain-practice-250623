from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain_openai import ChatOpenAI


# Constants
GPT_4O_MODEL_STR = "gpt-4o"
OPENAI_TOOLS_AGENT_OWNER_REPO_STR = "hwchase17/openai-tools-agent"

# This is a basic tool that does not require an input schema.
# Use this approach for simple functions that only need one parameter.
@tool()
def greet_user(name: str) -> str:
    """Greets the user by name."""
    return "Hello, {}!".format(name)


# Pydantic models for tool arguments
# Define a Pydantic model to specify the input schema for tools that need more structured input.
class ReverseStringsArgs(BaseModel):
    text: str = Field(description="Text to be reversed")


# Tool with One Parameter using args_schema
# Use the args_schema to specify the input schema using a Pydantic model.
@tool(args_schema=ReverseStringsArgs)
def reverse_string(text: str) -> str:
    """Reverses the given string."""
    return text[::-1]


# Another Pydantic model for tool arguments
class ConcatenateStringsArgs(BaseModel):
    a: str = Field(description="First string")
    b: str = Field(description="Second string")


# Tool with Two Parameters using args_schema
# This tool requires multiple input parameters, so we use the args_schema to define the schema.
@tool(args_schema=ConcatenateStringsArgs)
def concatenate_strings(a: str, b: str) -> str:
    """Concatenates two strings."""
    print("a", a)
    print("b", b)
    return a + b


# Create tools using the @tool decorator
# The @tool decorator simplifies the process of defining tools by handling the setup automatically.
tools = [
    greet_user, # Tool without args_schema
    reverse_string, # Tool with 1 param using args_schema
    concatenate_strings, # Tool with 2 params using args_schema
]

llm = ChatOpenAI(model=GPT_4O_MODEL_STR)
# Pull prompt template from the hub
prompt = hub.pull(owner_repo_commit=OPENAI_TOOLS_AGENT_OWNER_REPO_STR)

# Create the Reason & Act agent
# This function sets up an agent capable of calling tools based on the provided prompt.
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt, # Guides the agent's responses
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True, # Enable verbose logging
    handle_parsing_errors=True, # Handle parsing errors gracefully
)

greet_response = agent_executor.invoke(input={"input": "Greet Alice"})
print("Response for 'Greet Alice':", greet_response)

reverse_response = agent_executor.invoke(input={"input": "Reverse the string 'hello'"})
print("Response for 'Reverse the string hello':", reverse_response)

concat_response = agent_executor.invoke(input={"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", concat_response)
