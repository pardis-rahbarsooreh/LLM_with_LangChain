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
# Sequential Chains

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SimpleSequentialChain

# creating the first chain
llm1 = ChatOpenAI(model_name='gpt-4', temperature=0.7)
template = ''' You are an experienced computer scientist.
Write a function that implements the concept of {concept}'''

prompt1 = PromptTemplate(
    input_variables=['concept'],
    template=template
)
chain1 = LLMChain(llm=llm1, prompt=prompt1)

# creating the second chain
llm2 = ChatOpenAI(model_name='gpt-4', temperature=0.7)
template = ''' Given the python function {function}, describe it as detailed as possible'''

prompt2 = PromptTemplate(
    input_variables=['function'],
    template=template
)
chain2 = LLMChain(llm=llm2, prompt=prompt2)

overall_chain = SimpleSequentialChain(chains=[chain1, chain2], verbose=True)
output = overall_chain.invoke('linear regression')
