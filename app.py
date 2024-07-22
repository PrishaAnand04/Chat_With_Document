from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import requests
import dotenv 
from langchain_openai import AzureChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
import fitz
from werkzeug.utils import secure_filename
import os
from urllib.parse import urljoin


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

global_text = ""

client_cc = AzureOpenAI(
    api_key="f35b5881905f403eb0c39bb0a9f45cf5",  
    api_version="2024-02-15-preview",
    azure_endpoint="https://service-ih.openai.azure.com/",
    azure_deployment='gpt3516k'
)

client_em = AzureOpenAI(
    api_key="f35b5881905f403eb0c39bb0a9f45cf5",
    api_version="2024-02-01",
    azure_endpoint="https://service-ih.openai.azure.com/",
    azure_deployment="text-embedding-ada-002"
)

@app.route('/')
def index():
    return render_template('frontend.html')

def extract_from_url(url):
    respon = requests.get(url)
    soup = BeautifulSoup(respon.content, 'html.parser')
    paras = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paras])
    links = soup.find_all('a', href=True)
    all_links = [urljoin(url, link['href']) for link in links[:1000]]
    return text, all_links

def extract_content(url):
    respon = requests.get(url)
    soup = BeautifulSoup(respon.content, 'html.parser')
    paras = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paras])
    return text

def url():
    try:
        data = request.get_json()
        text_input = data["url"]
    except Exception as e:
        return jsonify({"error": str(e)}), 300
    try:
        text, links = extract_from_url(text_input)
        return text, links
    except Exception as e:
        return jsonify({"error": str(e)}), 100

def save_file_and_extract_text(file):
    try:
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            if filename.lower().endswith('.pdf'):
                text = extract_from_pdf(file_path)
            else:
                text = "Unsupported file format. Please upload a PDF file."
            return text
        else:
            return "No file uploaded."
    except Exception as e:
        return str(e)

def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def file_up():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_from_pdf(file_path)
        return text

def chunk_text(text, max_tokens=1500):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def summary_g(text):
    prompt = f"Summarize the following text in maximum 5 lines:\n\n{text}"
    response = client_cc.chat.completions.create(
        model="gpt3516k",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0
    )
    return response.choices[0].message.content

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    global global_text, chunks, global_text_url
    try:
        if 'file' in request.files:
            global_text = save_file_and_extract_text(request.files['file'])
            chunks = chunk_text(global_text)
            summary_f = [summary_g(chunk) for chunk in chunks]
            summary = ' '.join(summary_f)
            return jsonify({"summary": summary})
        elif 'url' in request.json and request.json['url'] != "":
            global_text_url = url()
            if isinstance(global_text_url, tuple):
                chunks = chunk_text(global_text_url[0])
                summary_u = [summary_g(chunk) for chunk in chunks]
                summary_url = ' '.join(summary_u)
                hyperlinks = global_text_url[1]
                return jsonify({"summary": summary_url, "hyperlinks": hyperlinks})
            else:
                return global_text_url
        elif 'url_hp' in request.json and request.json['url_hp'] != "":
            hyperlink_input = request.json['url_hp']
            content = extract_content(hyperlink_input)
            chunks = chunk_text(content)
            summary_hp = [summary_g(chunk) for chunk in chunks]
            summary = ' '.join(summary_hp)
            return jsonify({"summary": summary})
        else:
            return jsonify({"error": "No file or URL provided"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

REVIEWS_CHROMA_PATH = "chroma_data"

def get_vector_db(text):
    chunks = chunk_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_db = Chroma.from_documents(
        documents, AzureOpenAIEmbeddings(
            api_key="f35b5881905f403eb0c39bb0a9f45cf5",
            api_version="2024-02-01",
            azure_endpoint="https://service-ih.openai.azure.com/",
            azure_deployment="text-embedding-ada-002"
        ), persist_directory=REVIEWS_CHROMA_PATH
    )
    return vector_db

def answer_g(question, vector_db):
    reviews_retriever = vector_db.as_retriever(k=10)
    chat_model = AzureChatOpenAI(
        api_key="f35b5881905f403eb0c39bb0a9f45cf5",
        api_version="2024-02-01",
        azure_endpoint="https://service-ih.openai.azure.com/",
        azure_deployment="gpt3516k",
        temperature=0
    )

    review_template_str = """Your job is to use the information provided in the document
    to answer questions. Use the following context to answer questions.
    Be as detailed as possible, but don't make up any information
    that's not from the context. If you don't know an answer, say
    you don't know.
    {context}"""

    review_system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=review_template_str,
        )
    )

    review_human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )

    messages = [review_system_prompt, review_human_prompt]

    review_prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    review_chain = (
        {"context": reviews_retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )

    answer = review_chain.invoke(question)
    return answer

@app.route('/find_answer', methods=['POST'])
def generate_answer():
    global global_text, chunks, global_text_url
    try:
        vector_db = None
        if 'file' in request.files:
            vector_db = get_vector_db(global_text)
        else:
            vector_db = get_vector_db(global_text_url[0])

        if not vector_db:
            return jsonify({"error": "Failed to create vector database"}), 500

        data = request.get_json()
        question = data["question"]
        answer = answer_g(question, vector_db)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False)
