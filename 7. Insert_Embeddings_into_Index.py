import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#---------------------------------------------------------------
# Splitting and Embedding Text Using LangChain
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./Documents/Sample.pdf")
pages = loader.load_and_split()

whole_pdf = ''
for page in pages:
    whole_pdf += page.page_content

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # maximum chunk size
    chunk_overlap=200,    # maximum overlap between chunks size
    length_function=len
)

chunks = text_splitter.create_documents([whole_pdf])
#---------------------------------------------------------------
# Inserting the Embeddings into a Pinecone Index

from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))

# viewing all indexes names
all_indexes = pc.list_indexes().names()
print('all indexes:', all_indexes)


# deleting all indexes
for index in all_indexes:
    print('deleting all indexes!')
    pc.delete_index(index)
print(pc.list_indexes())


# creating a new index
index_name = "sample-index"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )
    print(pc.list_indexes().names())

from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain_community.vectorstores import Pinecone
vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)













