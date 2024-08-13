import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


#---------------------------------------------------------------
# LangChain Agents

from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-4', temperature=0.7)
agent_executor = create_python_agent(
    llm=llm,
    tool=PythonREPLTool(),
    verbose=True
)
agent_executor.invoke('Calculate 1.7**5.2')












