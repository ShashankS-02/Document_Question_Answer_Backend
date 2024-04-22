from flask import Flask, jsonify, request, url_for
from flask_cors import CORS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import re
import sys
sys.path.append('.')

app = Flask(__name__)
CORS(app)

file_name = ''


def extract_info():
    d = "Don't make it a sentence."
    m = "Most recent"
    # loader = UnstructuredFileLoader(
    #     '/Users/shashankshandilya/Projects/Question-Answer-Generation-App/static/docs/' + file_name)
    rq = "Return only the qualification." + d
    # documents = loader.load()
    #
    # text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    # texts = text_splitter.split_documents(documents)
    r = "Return only the skills." + d
    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small")
    # doc_search = Chroma.from_documents(texts, embeddings)
    # chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)
    loader = UnstructuredFileLoader('/Users/shashankshandilya/Projects/Question-Answer-Generation-App/static/docs/'
                                    + file_name)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small")
    # doc_search = Chroma.from_documents(texts, embeddings)
    # chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"], model="text-embedding-3-small")
    doc_search = Chroma.from_documents(texts, embeddings)
    chain = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=doc_search)

    data_obj = {}

    # #query = "Candidate Name? If not available return N/A"
    # candidate_name = chain.run(query)
    #
    # query = "Total Experience in years? If not available return N/A"
    # experience = chain.run(query)
    #
    # query = "College name? If not available return N/A"
    # college_name = chain.run(query)
    #
    # query = "Any three skills? Separate them with only space. If not available return N/A"
    # three_skills = chain.run(query)

    parameter = "Highest education or highest qualification?" + rq
    qualification = chain.run(parameter)

    parameter = m + "College Name?"
    college_name = chain.run(parameter)

    parameter = "Any three skills?" + r
    three_skills = chain.run(parameter)

    # data_obj["candidate_name"] = candidate_name
    # data_obj["experience"] = experience
    # data_obj["college_name"] = college_name
    # data_obj["skills"] = three_skills
    # return data_obj

    data_obj["Highest Qualification"] = qualification
    data_obj["College Name"] = college_name
    data_obj["Skills"] = three_skills
    print(data_obj)
    return data_obj


@app.route('/', methods=['POST'])
def upload_file():
    up_file = request.files['file']
    up_file.save('./static/docs/' + up_file.filename)
    print(up_file.filename)
    global file_name
    file_name = up_file.filename
    return jsonify({'message': 'File uploaded successfully'})


@app.route("/autofill", methods=['GET'])
# @cross_origin()
def autofill():
    data_obj = extract_info()
    user_dict = jsonify(data_obj)
    print(user_dict)
    return user_dict


UPLOAD_FOLDER = '/Users/shashankshandilya/Projects/Question-Answer-Generation-App/static/docs'


if __name__ == '__main__':
    app.run(debug=True)
