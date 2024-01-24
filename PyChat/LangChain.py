#pip install langchain pypdf openai chromadb tiktoken docx2txt
# Tomado de https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339
# Interacting with a single PDF
# Sin usar embeddings

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ["OPENAI_API_KEY"] = "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"

pdf_loader = PyPDFLoader('C:/ChatGPT_con_mis_Datos/LangChain_docs/RachelGreenCV.pdf')
documents = pdf_loader.load()

# we are specifying that OpenAI is the LLM that we want to use in our chain
chain = load_qa_chain(llm=OpenAI(), verbose=False)
query = 'What is the main achievement documented in the CV?'
response = chain.run(input_documents=documents, question=query)
print(response) 

