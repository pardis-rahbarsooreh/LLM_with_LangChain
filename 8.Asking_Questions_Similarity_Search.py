import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#---------------------------------------------------------------
# Asking Questions (Similarity Search)

from pinecone import Pinecone
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))


# viewing all indexes names
all_indexes = pc.list_indexes().names()
print('all indexes:', all_indexes)
index_name = all_indexes[0]

# defining embeddings model
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()


# getting the previously defined vectorstore to be used for querying
from langchain_pinecone import PineconeVectorStore
vector_store = PineconeVectorStore(
    pc.Index(index_name),
    embeddings)

# defining a query
query = "What is isolation?"

# extracting all the relevant chunks
result = vector_store.similarity_search(query)
print(result)

# iterate over the chunks and only printing the chunk texts
for r in result:
    print(r.page_content)
    print('-'*50)


#---------------------------------------------------------------
# Turning the resulted chunks to natural language for readability
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o', temperature=0.6)

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 3})

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever)

answer = chain.invoke(query)
print('Query:', answer['query'],
      '\nResult:', answer['result'])











