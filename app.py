import os
import pandas as pd
from flask import Flask, request, jsonify # Using Flask for the web server
import google.generativeai as genai
from flask_cors import CORS

# Ensure these are installed: pip install langchain_google_genai langchain faiss-cpu Flask
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
import time

app = Flask(__name__)
CORS(app)  # Enable for all origins

## ConversationalRetrievalChain for memory 

# --- Configuration ---
# Retrieve API key securely from environment variables
# (Cloud Run injects these, DO NOT hardcode in production)
os.environ["GOOGLE_API_KEY"] = os.getenv("OUR_GOOGLE_API_KEY")


prompt_template = """You are a helpful assistant for OpenCart modules. Answer concisely.
Use the following context to answer the user's question.
If you don't know the answer, just say I am unable to answer your questions. Please feel free to raise the ticket. Our support team will get back to you as soon as possible. Thanks.  https://www.opencartextensions.in/ticket.
{context}


Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

save_directory = "./faiss_index_opencart"
EXCEL_FILE_PATH = "./opencart_data_1.xlsx" # Relative path within the Docker container
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1) # temperature=0.0 for deterministic answers


if os.path.exists(save_directory):
    print(f"Loading FAISS index from {save_directory}...")
    # 'allow_dangerous_deserialization=True' is often needed for security reasons when loading local indexes.
    vector_store = FAISS.load_local(save_directory, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully.")
else:
    print(f"Error: FAISS index directory not found at {save_directory}.")
    print("Please ensure you have run the previous code to save the index, and that the directory path is correct.")
    exit() # Exit if the index is not found

start_retrieval_time = time.time()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # 'stuff' means all retrieved docs are "stuffed" into the prompt
    retriever=vector_store.as_retriever(search_kwargs={"k": 1}), # Retrieve top 3 relevant chunks
    return_source_documents=True, # To see which documents were retrieved
    chain_type_kwargs={"prompt": PROMPT}
  )
end_retrieval_time = time.time()
print(f"Retrieval time: {end_retrieval_time - start_retrieval_time:.2f} seconds.")


vector_store = None # Initialize as None
llm = None
embeddings = None

@app.route('/')
def health_check():
    return jsonify({"status": "running", "message": "RAG API is live!"})

@app.route('/ask', methods=['GET'])
def ask_question():
    global qa_chain # Access the global qa_chain

    if qa_chain is None:
        if qa_chain is None: # If initialization failed
            return jsonify({"error": "RAG system failed to initialize"}), 500

    question = request.args.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"Received question: {question}")
    try:
        start_qa_time = time.time()    
        response = qa_chain({"query": question})
        answer = response["result"]
        end_qa_time = time.time()
        print(f"Total QA chain (LLM) time: {end_qa_time - start_qa_time:.2f} seconds.")

        answer  = answer.replace("Based on the context provided","") 
        #source_docs = []
        for doc in response["source_documents"]:
            print(f"Content length: {len(doc.page_content)} characters") # Check length of retrieved content
            print(f"Source File: {doc.metadata.get('source_file', 'N/A')}, Row Number: {doc.metadata.get('row_number', 'N/A')}")
            print("-" * 30)

        #    source_docs.append({
        #        "content": doc.page_content,
        #        "metadata": doc.metadata
        #    })

        return jsonify({
            "answer": answer
        })
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the RAG system when the app starts
    # For Cloud Run, the port is provided by the PORT environment variable
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug = True)
