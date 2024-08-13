import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


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

