from langchain_core.documents import Document
from langchain.chat_models import ChatOpenAI

from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_community.vectorstores import Pinecone
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader        
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import(
    SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
)     
import streamlit as st
import os
import openai
import tempfile
import ollama
import pinecone
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
load_dotenv()
# initiate the chat engine

#pinecone vector database

pine_key=st.secrets["PINECONE_API_KEY"]


openai_api=st.secrets["openai_api_key"]

pc=pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
my_embedding_model=OpenAIEmbeddings(api_key=st.secrets["openai_api_key"],model="text-embedding-ada-002")    
openai.api_key =openai_api

# initiate the chat engine
deepseek="deepseek-r1:1.5b"
llm= ChatOllama(model=deepseek,base_url="http://localhost:11434",temperature=0.7)
llm1=ChatOpenAI(api_key=st.secrets["openai_api_key"], model="gpt-3.5-turbo")

def extract_text_from_pdf(uploaded_file):      
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
             temp_pdf.write(uploaded_file.read())
             temp_path = temp_pdf.name           
             
        
        loader=PyPDFLoader(temp_path)
        pages=loader.load()        
        documents=[Document(page_content=page.page_content) for page in pages]
        #texts=[doc.page_content for doc in pages]
        return documents

def split_text(documents):        
     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
     splits=text_splitter.split_documents(documents)
     return splits


def ask_deepseek(query,context):
     system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. the context is {context}. If you don't know the answer, say that you "
    "don't know. Answer in only 2 sentences maximum and keep the "
    "answer concise."
    "\n"   
)

     prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)  
     vector_store=PineconeVectorStore.from_documents(context,embedding=my_embedding_model,index_name="my-deepseek-pdf-project1")
     retrieval= vector_store.as_retriever()
     
     question_answer_chain = create_stuff_documents_chain(llm1, prompt)
     
     rag_chain = create_retrieval_chain(retrieval, question_answer_chain)
     try:
         results = rag_chain.invoke({"context":context,"input":query})

         final_res=results.get("answer")
     except Exception as e:
         st.write(f"Error while invoking RAG chain: {e}")
         final_res = "An error occurred while processing your query."

         
          
     return final_res
    

         

    




    
