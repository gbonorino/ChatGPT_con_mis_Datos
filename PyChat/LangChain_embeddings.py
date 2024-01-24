# Tomado de https://betterprogramming.pub/building-a-multi-document-reader-and-chatbot-with-langchain-and-chatgpt-d1864d47e339
# Interacting with a single PDF usando embeddings
# pip install chromadb

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

pdf_loader = PyPDFLoader('C:/ChatGPT_con_mis_Datos/LangChain_docs/RachelGreenCV.pdf')
documents = pdf_loader.load()

# we split the data into chunks of 1,000 characters, with an overlap
# of 200 characters between the chunks, which helps to give better results
# and contain the context of the information between chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# we create our vectorDB, using the OpenAIEmbeddings tranformer to create
# embeddings from our text chunks. We set all the db information to be stored
# inside the ./data directory, so it doesn't clutter up our source files
vectordb = Chroma.from_documents(
  documents,
  embedding=OpenAIEmbeddings(),
  persist_directory='./LangChain_docs'
)
vectordb.persist()
# Hasta aqui se creo un vector store, o DB, empleando el transformador
#   OpenAIEmbeddings. La informacion se almacena en una BD SQLite3.
# Ahora hay que pasar el contenido del vector store al modelo
# En vez de pasar el documento completo, como en el ejemplo anterior, se pasa
#    el vector store  y el texto mas relevante de acuerdo con la consulta..
# La chain RetrievalQA hace esta tarea.

qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=vectordb.as_retriever(search_kwargs={'k': 7}),
    return_source_documents=True
)

# we can now execute queries against our Q&A chain
result = qa_chain({'query': 'Who is the CV about?'})
print(result['result'])