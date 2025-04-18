

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import certifi
import json
from flask import Flask, render_template, request, jsonify
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import CSVLoader, DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import warnings
from langchain.vectorstores import Chroma
from langchain_community.llms import Ollama
from datetime import datetime

warnings.filterwarnings('ignore')

app = Flask(__name__)

API_KEY = "C1HRSweTniWdBuLmTTse9w8KpkoiouM5"
PERSIST_DIR = "docs/chroma_rag/"

llm = Ollama(model="mistral", temperature=0.01)

def get_stock_data(ticker, api_key):
    """Get comprehensive stock data from multiple endpoints"""
    all_data = {}
    
    quote_url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
    response = urlopen(quote_url, cafile=certifi.where())
    quote_data = json.loads(response.read().decode("utf-8"))
    if quote_data and isinstance(quote_data, list):
        all_data["quote"] = quote_data[0]
    
    try:
        ratios_url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?apikey={api_key}&limit=1"
        response = urlopen(ratios_url, cafile=certifi.where())
        ratios_data = json.loads(response.read().decode("utf-8"))
        if ratios_data and isinstance(ratios_data, list):
            all_data["ratios"] = ratios_data[0]
    except:
        pass
    
    try:
        metrics_url = f"https://financialmodelingprep.com/api/v3/key-metrics/{ticker}?apikey={api_key}&limit=1"
        response = urlopen(metrics_url, cafile=certifi.where())
        metrics_data = json.loads(response.read().decode("utf-8"))
        if metrics_data and isinstance(metrics_data, list):
            all_data["metrics"] = metrics_data[0]
    except:
        pass
    
    try:
        profile_url = f"https://financialmodelingprep.com/api/v3/profile/{ticker}?apikey={api_key}"
        response = urlopen(profile_url, cafile=certifi.where())
        profile_data = json.loads(response.read().decode("utf-8"))
        if profile_data and isinstance(profile_data, list):
            all_data["profile"] = profile_data[0]
    except:
        pass
    
    return all_data

def prepare_financial_data(data):
    """Transform nested data into flattened dataframe"""
    flattened_data = {}
    
    current_date = datetime.now().strftime("%Y-%m-%d")
    flattened_data["data_retrieved_date"] = current_date
    flattened_data["analysis_date"] = current_date
    
    for section, section_data in data.items():
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                flattened_data[f"{section}_{key}"] = value
    
    df = pd.DataFrame([flattened_data])
    
    df["data_description"] = (
        f"This is current financial data for the stock as of {current_date}. "
        f"It includes quote information, financial ratios, key metrics, and company profile data. "
        f"All monetary values are in USD unless otherwise specified."
    )
    
    return df

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        ticker = request.form.get("ticker", "MSFT")
        question = request.form.get("question", f"{ticker} Financial Report")

        try:
            data = get_stock_data(ticker, API_KEY)
            
            if not data or not any(data.values()):
                return render_template("index.html", result=f"Error: No data found for ticker {ticker}")
            
            df = prepare_financial_data(data)
            
            csv_path = "stock_data.csv"
            df.to_csv(csv_path, index=False)
            
            loader = DataFrameLoader(df, page_content_column="data_description")
            documents = loader.load()
            
            csv_loader = CSVLoader(csv_path)
            csv_documents = csv_loader.load()
            documents.extend(csv_documents)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            texts = text_splitter.split_documents(documents)
            
            hg_embeddings = HuggingFaceEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=texts,
                collection_name="stock_data",
                embedding=hg_embeddings,
                persist_directory=PERSIST_DIR
            )
            
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            template = """
            IMPORTANT INSTRUCTION: You are a Financial Analyst providing information SOLELY based on the data provided below. 
            Do NOT refer to any other knowledge about this company or defer to external sources.
            
            CURRENT DATA: 
            {context}
            
            USER QUERY: {question}
            
            YOUR TASK:
            1. Analyze ONLY the data provided above
            2. Answer the query directly using ONLY this data
            3. If specific information isn't in the data, clearly state what IS available and provide that information instead
            4. Format your response in a professional, easy-to-read format
            5. Do NOT suggest visiting websites or getting data elsewhere
            6. Do NOT apologize for limitations - just work with what you have
            7. ASSUME ALL DATA IS CURRENT AND ACCURATE
            
            RESPONSE (using only the data shown above):
            """
            
            PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
            response = qa({"query": question})
            result = response["result"]


        except Exception as e:
            import traceback
            result = f"Error: {str(e)}\n{traceback.format_exc()}"

    return render_template("index.html", result=result)

@app.route('/template')
def template():
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)