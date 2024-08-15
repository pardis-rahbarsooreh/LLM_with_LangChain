import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

#---------------------------------------------------------------
# Splitting and Embedding Text Using LangChain
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("./Documents/Sample.pdf")
pages = loader.load_and_split()
#print('Page 11 of the document: ')
#print(pages[10].page_content)

whole_pdf = ''
for page in pages:
    whole_pdf += page.page_content
#print('This is the whole PDF as one string: ')
#print(whole_pdf)

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,     # maximum chunk size
    chunk_overlap=20,    # maximum overlap between chunks size
    length_function=len
)

print('--------------------------------------------------------------')
chunks = text_splitter.create_documents([whole_pdf])
#print('101th chunk of the whole pdf is: ')
#print(chunks[100].page_content)
#print(f'Now you have {len(chunks)} chunks')











