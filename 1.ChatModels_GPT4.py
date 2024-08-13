import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#---------------------------------------------------------------
# ChatModel GPT-4

from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI

chat = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7, max_tokens=1024)
messages = [
    SystemMessage(content='You are a computer scientist and response only in German.'),
    HumanMessage(content='explain API key in one sentence')
]
output = chat.invoke(messages)
print(output.content)
