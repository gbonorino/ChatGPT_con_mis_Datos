
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate

os.environ["OPENAI_API_KEY"] = "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"
llm = OpenAI(model_name =  "text-davinci-003")
print(llm)   # Mostra el modelo que usa

template = """
{texto}
Empiezo con {lenguaje}.
"""

prompt = PromptTemplate(template=template, input_variables=["cuantos", "lenguaje"],)
promt_final = prompt.format(lenguaje="Python",
                            texto = "Mi meta es aprender {cuantos} lenguajes de programacion.")

print(prompt_final)

