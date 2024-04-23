import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from fuzzywuzzy import fuzz
import pytest

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


def extract_info(file_path):
    """Extracts information from a resume using Langchain."""
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-small")
    doc_search = Chroma.from_documents(texts, embeddings)
    chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)

    parameter = "Highest education or highest qualification?"
    qualification = chain.run(parameter)

    parameter = "Most recent College Name?"
    college_name = chain.run(parameter)

    parameter = "Any three skills? Separate them with only space."
    three_skills = chain.run(parameter)

    extracted_data = {
        "Highest Qualification": qualification,
        "College Name": college_name,
        "Skills": three_skills.strip().split(" ")  # Split skills and remove extra spaces
    }

    return extracted_data


def is_similar(desired_value, extracted_value, threshold=80):
    """Checks if extracted value is fuzzy-similar to desired value."""
    return fuzz.ratio(desired_value, extracted_value) >= threshold


if __name__ == "__main__":
    pytest.main()
