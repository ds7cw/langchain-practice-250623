import os
from typing import Type

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.pydantic_v1 import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

load_dotenv()

# Constants
GPT_4O_MODEL_STR = "gpt-4o"
OPENAI_TOOLS_AGENT_OWNER_REPO_STR = "hwchase17/openai-tools-agent"


class SimpleSearchInput(BaseModel):
    query: str = Field(description="Should be a search query")


class MultiplyNumbersArgs(BaseModel):
    x: float = Field(description="First number to multiply")
    y: float = Field(description="Second number to multiply")


class SimpleSearchTool(BaseTool):
    name = "simple_search"
    description = "Useful for when you need to answer question about current events"
    args_schema: Type[BaseModel] = SimpleSearchInput

    def _run(self, query: str) -> str:
        """Use the tool."""
        from tavily import TavilyClient
    
        api_key = os.getenv("TAVILY_API_KEY")
        client = TavilyClient(api_key=api_key)
        results = client.search(query=query)
        return "Search results for: {}\n\n\n{}\n".format(query, results)


class MultiplyNumbersTool(BaseTool):
    name = "multiply_numbers"
    description = "Useful for multiplying two numbers"
    args_schema: Type[BaseModel] = MultiplyNumbersArgs

    def _run(self, x: float, y: float) -> str:
        """Use the tool"""
        result = x * y
        return "The product of {} and {} is {}".format(x, y, result)

tools = [
    SimpleSearchTool(),
    MultiplyNumbersTool(),
]

llm = ChatOpenAI(model=GPT_4O_MODEL_STR)
prompt = hub.pull(owner_repo_commit=OPENAI_TOOLS_AGENT_OWNER_REPO_STR)

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

search_response = agent_executor.invoke(input={"input": "Search for Apple Intelligence"})
print("Response for 'Search for Apple Intelligence':", search_response)

multiply_response = agent_executor.invoke(input={"input": "Multiply 10 and 20"})
print("Response for 'Multiply 10 and 20':", multiply_response)
