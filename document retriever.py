# LOAD DOCUMENTS
# json loader
from langchain.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
file_path = 'DURF_2023/all.jsonl'
#file_path = '/all.jsonl'
from langchain.text_splitter import CharacterTextSplitter

loader = JSONLoader(
    file_path='DURF_2023/all.jsonl',
    #file_path='/all.jsonl',
    jq_schema='.content',
    json_lines=True)
data = loader.load()

# REMOVE ABUNDANT TEXT IN THE DOCUMENTS
for i in range(len(data)):
  data[i].page_content = data[i].page_content[367:-227]

# CHUNK DOCUMENTS INTO SMALLER PIECES
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# LOAD EMBEDDINGS
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# DOCUMENT BASE
from langchain.vectorstores import Chroma
db = Chroma.from_documents(docs, embeddings)

# LOAD THE RETRIEVER [k=4 is to return the top 4 documents]
'''
from langchain.chains import RetrievalQA
qa = RetrievalQA.from_chain_type(
    llm=local_llm,
    chain_type="stuff",
    retriever=db.as_retriever(k=4),
    return_source_documents = True
)
'''

# USE THE RETRIEVER [All the query are stored in q_all]
'''
r_data = []
for q in q_all:
  # pair = qa.run(q)
  pair=qa.__call__(q)
  d = pair['source_documents']
  r_data.append({'question':q,'source_documents':d})
'''


