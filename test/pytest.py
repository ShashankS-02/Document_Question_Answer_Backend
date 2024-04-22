import os
import requests
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import pytest

# Replace with your OpenAI API key
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"

def extract_info(file_path, desired_qualification, desired_college, desired_skills):
    """Extracts information from a resume using Langchain and compares with desired values."""
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

    correct_qualification = all(q in qualification for q in desired_qualification.split(" or "))
    correct_college = college_name == desired_college
    correct_skills = all(skill in extracted_data["Skills"] for skill in desired_skills.split(", "))

    accuracy = (
        int(correct_qualification) + int(correct_college) + int(correct_skills)
    ) / 3

    print(f"\nExtracted Information:\n{extracted_data}")
    print(f"\nDesired Information:")
    print(f"  - Highest Qualification: {desired_qualification}")
    print(f"  - College Name: {desired_college}")
    print(f"  - Skills: {desired_skills}")

    print(f"\nAccuracy: {accuracy:.2f}")  # Format accuracy to two decimal places

    return accuracy

@pytest.mark.parametrize(
    "file_path, desired_qualification, desired_college, desired_skills",
    [
        ("tests/resume1.docx", "Master's Degree", "MIT", "Python, Java, Machine Learning"),
        ("tests/resume2.pdf", "Bachelor's in Computer Science", "Stanford", "C++, Web Development, Data Analysis"),
    ],
)
def test_extract_info(file_path, desired_qualification, desired_college, desired_skills):
    """Tests extract_info function with various resumes and desired information."""
    accuracy = extract_info(file_path, desired_qualification, desired_college, desired_skills)
    assert accuracy == 1.0  # Assert perfect accuracy for each test case

if __name__ == "__main__":
    pytest.main()