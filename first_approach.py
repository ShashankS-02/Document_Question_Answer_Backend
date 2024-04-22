import langchain
import os
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import nltk
nltk.download("punkt")

os.environ["OPENAI_API_KEY"] = "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"

loader = UnstructuredFileLoader("/Users/shashankshandilya/Projects/Question-Answer-Generation-App-main/static/docs/Shashank's Resume (1).pdf")
documents= loader.load()

loader = UnstructuredFileLoader('SamplePDF.pdf', mode='elements')

text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
doc_search = Chroma.from_documents(texts, embeddings)
chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)

data_obj = {}

query = "College name"
college_name_ans = chain.run(query)

query = "Candidate name"
candidate_name_ans = chain.run(query)

data_obj["college_name"] = college_name_ans
data_obj["candidate_name"] = candidate_name_ans
# print(data_obj)
