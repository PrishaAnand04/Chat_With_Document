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

dotenv.load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

global_text=""

@app.route('/')
def index():
    return render_template('front_end.html')

def extract_from_url(url):
    url_add=url
    respon=requests.get(url_add)
    soup=BeautifulSoup(respon.content, 'html.parser')
    paras=soup.find_all('p')
    text='\n'.join([p.get_text() for p in paras])
    return text

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

client_em=AzureOpenAI(
    api_key="f35b5881905f403eb0c39bb0a9f45cf5",
    api_version="2024-02-01",
    azure_endpoint="https://service-ih.openai.azure.com/",
    azure_deployment="text-embedding-ada-002"
)  
def url_g():
    try:
        data = request.get_json()
        text_input = data["url"]
    except Exception as e:
        return jsonify({"error": str(e)}), 300
    try:
        text=extract_from_url(text_input)
        return text
    except Exception as e:
        return jsonify({"error":str(e)}),100

client_cc = AzureOpenAI(
  api_key = "f35b5881905f403eb0c39bb0a9f45cf5",  
  api_version = "2024-02-15-preview",
  azure_endpoint= "https://service-ih.openai.azure.com/",
  azure_deployment='gpt3516k'
)

def summary_g(text):
    prompt = f"Summarize the following text:\n\n{text}"
    response = client_cc.chat.completions.create(
        model="gpt3516k",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.5
    )
    return response.choices[0].message.content

def file_up():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_from_pdf(file_path)
        return text

def url():
    text_input = request.json['url']
    text = extract_from_url(text_input)
    return text

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    global global_text
    try:
        if('file' in request.files):
            if 'file' not in request.files :
                return jsonify({'error': 'No file part'}), 400
            global_text= save_file_and_extract_text(request.files['file'])
            summary=summary_g(global_text)
            return jsonify({"summary": summary})
        elif('url' in request.json !=""):
            global_text=url()
            summary_url=summary_g(global_text)
            return jsonify({"summary":summary_url})
        else:
            return jsonify({"error": "No file or URL provided"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

    
REVIEWS_CHROMA_PATH = "chroma_data"
def get_vector_db(text):
    documents = [Document(page_content=text)]
    vector_db = Chroma.from_documents(
        documents, AzureOpenAIEmbeddings(
            api_key="f35b5881905f403eb0c39bb0a9f45cf5",
            api_version="2024-02-01",
            azure_endpoint="https://service-ih.openai.azure.com/",
            azure_deployment="text-embedding-ada-002"
        ), persist_directory=REVIEWS_CHROMA_PATH
    )

def answer_g(quess):
    reviews_vector_db = Chroma(
            persist_directory=REVIEWS_CHROMA_PATH,
            embedding_function=AzureOpenAIEmbeddings(
            api_key="f35b5881905f403eb0c39bb0a9f45cf5",
            api_version="2024-02-01",
            azure_endpoint="https://service-ih.openai.azure.com/",
            azure_deployment="text-embedding-ada-002"
        )
    )
        
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

    reviews_retriever  = reviews_vector_db.as_retriever(k=10)
    review_chain = (
        {"context": reviews_retriever, "question": RunnablePassthrough()}
        | review_prompt_template
        | chat_model
        | StrOutputParser()
    )

    answer=review_chain.invoke(quess)
    return answer

@app.route('/find_answer', methods=['POST'])
def generate_answer():
    global global_text
    try:
        if not global_text:
            return jsonify({"error": "No text available from file or URL"}), 400
        get_vector_db(global_text)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    try:
        data = request.get_json()
        question= data["question"]
        answer=answer_g(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True,use_reloader=False)