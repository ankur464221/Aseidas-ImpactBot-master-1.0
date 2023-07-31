'''
requirements.txt file contents:
 
pyqt5==5.13
pyqtwebengine==5.13
langchain==0.0.154
PyPDF2==3.0.1
python-dotenv==1.0.0
streamlit==1.24.0.
faiss-cpu==1.7.4
streamlit-extras
altair<5        #Altair requires older version of (4)
openai
tiktoken

'''

import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Load the .env file

load_dotenv()

# Sidebar contents
with st. sidebar:
    st.title('Impact Bot by Vera Solutions')
    st.markdown('''
## About
This app is an LLM-powered chatbot that will allow you to chat with Annual Reports
## How to use
1. Upload your PDF(s)
2. Click on the "Start Chatting" button
3. Type in your question "What is the total number of beneficiaries for Skoll Foundation?"
''')
                
    add_vertical_space(5)
    st.write('Made with 💚 by Aseidas')



def main():
    st.header("Annual Reports Chatbot")
    st.subheader("This is a chatbot that will allow you to chat with Annual Reports")
    
                
load_dotenv()
 
def main():
    st.header("Chat with PDF 💬")
 
 
    # upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
 
    # st.write(pdf)
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
 
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
            )
        chunks = text_splitter.split_text(text=text)
 
        # # embeddings
        store_name = pdf.name[:-4]
        st.write(f'{store_name}')
        # st.write(chunks)
 
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write('Embeddings Loaded from the Disk')s
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
 
        # embeddings = OpenAIEmbeddings()
        # VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
 
        # Accept user questions/query
        query = st.text_input("Ask questions about your PDFs")
        # st.write(query)
 
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
 
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
 
if __name__ == '__main__':
    main()