from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_community.llms import OpenAI
#from langchain.llms import OpenAI
#from langchain.vectorstores import Pinecone
#from langchain_pinecone import PineconeVectorStore
from langchain_pinecone.vectorstores import PineconeVectorStore

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from pinecone import  ServerlessSpec    
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
from langchain_core.prompts import(
    SystemMessagePromptTemplate,AIMessagePromptTemplate,HumanMessagePromptTemplate,ChatPromptTemplate
)
#query=st.chat_input("Type your question",type=["pdf"])

load_dotenv()
# initiate the chat engine

#pinecone vector database

pine_key=st.secrets("PINECONE_API_KEY")


openai_api=st.secrets("openai_api_key")
#pc=pinecone.Pinecone(api_key=pine_key)
pc=pinecone.Pinecone(api_key=st.secrets("PINECONE_API_KEY"))
my_embedding_model=OpenAIEmbeddings(api_key=st.secrets("openai_api_key"),model="text-embedding-ada-002")    
openai.api_key =openai_api

# initiate the chat engine
deepseek="deepseek-r1:1.5b"
llm= ChatOllama(model=deepseek,base_url="http://localhost:11434",temperature=0.7)
llm1=OpenAI()

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


#        response = openai.embeddings.create(
 #       model="text-embedding-ada-002",
  #      input=[text]        
   # ) 
    #    return response['data'][0]['embedding']

# def query_pinecone(query):
#     query_embedding = split_text(query)
#     results = index.query(vector=query_embedding, top_k=5, inpineconeclientude_metadata=True)
#     return "\n".join([match["metadata"]["text"] for match in results["matches"]])


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
     
     results = rag_chain.invoke({"context":context,"input":query})

     final_res=results.get("answer")
     
     return final_res
    

# #state management
# if "uploaded" not in st.session_state:
#     st.session_state.uploaded=[{'role': "AI", 'content':"Hello! Please upload your pdf file in .pdf format.."}]
# my_embedding_model=OpenAIEmbeddings(api_key=openai_api,model="text-embedding-ada-002")
# # Chat container
# uploaded_file=st.file_uploader("Upload",type=["pdf"])
# if uploaded_file:
#     with st.spinner("Processing PDF... please wait"):
#         text1=extract_text_from_pdf(uploaded_file)
#         text=split_text(text1)


    
#     #with st.spinner("Uploading PDF file.. Please Wait..."):
    
#     if "my-deepseek-pdf-project1" not in pc.list_indexes().names():  
#         pc.create_index(
#         embedings=my_embedding_model,
#         name="my-deepseek-pdf-project1",
#         dimension=1536,  # Make sure this matches your embeddings
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")        
#                        )
#         index=pc.Index("my-deepseek-pdf-project1")
#     st.success("Pdf stored and processed successfully...!")    

#     query = st.text_input("Ask a question based on the PDF content:")
#     if query: 

#         with st.spinner("Processing your question... Please wait"):            
#             #query_embeddings=query_pinecone(query)         
#             answer=ask_deepseek(query,text)
#             st.write(" Your answer is....")
#             st.success(answer)



        


         

    




    
