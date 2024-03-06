
import os

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.agents import load_tools, initialize_agent, AgentType

from dotenv import dotenv_values
env_config = dotenv_values(".env")

os.environ['OPENAI_API_KEY'] = env_config["OPENAI_API_KEY"]


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

# LangChain provides Out-Of-box tools 
# Tools allow LLMs to access various information 
# sources such as Google, Wikipedia, YouTube, Python REPL Databases, etc.,
tools = load_tools(
    ["arxiv"], 
)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

print(agent_chain.run(
    "What's the paper 2401.12599 about?",
))