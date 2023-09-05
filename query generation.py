# Description: Generate queries for the QA system
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

# Remove abundant text in the documents
for i in range(len(data)):
  data[i].page_content = data[i].page_content[367:-227]
data = loader.load()

# CHUNK DOCUMENTS INTO SMALLER PIECES
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data)

# QUERY TEMPLATES
who = 'Who teaches %s at NYU Shanghai?'
what = 'Which courses does professor %s teach?'
email = "Tell me professor %s's email."
degree = 'What is the highest degree of professor %s?'
l = ['Introduction to Computer Programming','Introduction to Computer Science','Algorithm','Machine Learning','Computer Architecture','Natural Language Processing','Reinforcement Learning','Operating System','Distributed System']
## SCRAP PROFESSOR'S NAME
names = []
for doc in data:
  i = doc.page_content.find('\n')
  names.append(doc.page_content[:i])
# GENERATE QUERIES
q_who = [who%i for i in l]
q_what = [what%i for i in names]
q_email = [email%i for i in names]
q_degree = [degree%i for i in names]
q_all = q_who+q_what+q_email+q_degree 