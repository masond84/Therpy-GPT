from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
import os

os.environ["SERPAPI_API_KEY"] = "8d939898ac35a7185cbb7a8eb897c7105c197ad425cbd918cea6ba7386205ab6"
os.environ["OPENAI_API_KEY"] = "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"
# Create an instance of the OpenAI model
llm = OpenAI(temperature=0)

tools = load_tools(["serpapi", "llm-math"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)