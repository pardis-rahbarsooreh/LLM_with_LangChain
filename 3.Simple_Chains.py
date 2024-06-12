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
# Simple Chains

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

llm = ChatOpenAI(model_name='gpt-4', temperature=0.5)
template = ''' You are an experienced computer scientist.
Write a few sentences about {concept} in {language}'''

prompt = PromptTemplate(
    input_variables=['concept', 'language'],
    template=template
)

sequence = RunnableSequence(prompt | llm)
output = sequence.invoke({'concept': 'API key', 'language': 'English'})
print(output.content)

