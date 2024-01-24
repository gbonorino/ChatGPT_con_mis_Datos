import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

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
  persist_directory='./data_memoria'
)
vectordb.persist()

from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI

qa_chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(),
    vectordb.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True
)

import sys

chat_history = []
while True:
    # this prints to the terminal, and waits to accept an input from the user
    query = input('Prompt: ')
    # give us a way to exit the script
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        print(chat_history)
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    result = qa_chain.invoke({'question': query, 'chat_history': chat_history})
    print('Answer: ' + result['answer'])
    # we build up the chat_history list, based on our question and response
    # from the LLM, and the script then returns to the start of the loop
    # and is again ready to accept user input.
    chat_history.append((query, result['answer']))


from langchain_community.document_loaders import TextLoader

loader = TextLoader("./ECON57_Syllabus.txt")
data=loader.load()
print(data)