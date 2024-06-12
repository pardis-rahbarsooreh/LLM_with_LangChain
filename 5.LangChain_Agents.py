import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index_name = "docs-quickstart-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

print(os.environ.get('PINECONE_API_KEY'))



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












