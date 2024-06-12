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
# Prompt Templates

from langchain_core.prompts import PromptTemplate
from langchain.schema import(
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain_openai import ChatOpenAI

template = ''' You are an experienced computer scientist.
Write a few sentences about {concept} in {language}'''

prompt = PromptTemplate(
    input_variables=['concept', 'language'],
    template=template
)

# initialize the chat model
chat = ChatOpenAI(model_name='gpt-4', temperature=0.7, max_tokens=1024)

# Function to generate response using dynamic inputs
def get_response(concept, language):
    formatted_prompt = prompt.format(concept=concept, language=language)
    messages = [
        SystemMessage(content='You are an experienced computer scientist.'),
        HumanMessage(content=formatted_prompt)
    ]
    return chat.invoke(messages)

# Example usage
output = get_response('API key', 'English')
print(output.content)
output = get_response('API key', 'German')
print(output.content)
