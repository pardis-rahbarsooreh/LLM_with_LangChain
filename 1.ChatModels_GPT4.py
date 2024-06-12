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
# ChatModel GPT-4

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name='gpt-4o', temperature=0.7, max_tokens=1024)
messages = [
    SystemMessage(content='You are a computer scientist and response only in German.'),
    HumanMessage(content='explain API key in one sentence')
]
output = chat.invoke(messages)
print(output.content)
