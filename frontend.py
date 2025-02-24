import backend
from langchain_openai import OpenAIEmbeddings
import streamlit as st
import os
import pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone
openai_api=st.secrets["openai_api_key"]
# Custom CSS styling for mobile responsiveness
st.markdown("""
<style>
    /* Global Styling */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
        font-family: Arial, sans-serif;
    }

    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        overflow: auto;
    }
    
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Mobile responsive design */
    @media (max-width: 600px) {
        .sidebar .sidebar-content {
            width: 100%;
        }
        .stButton button {
            width: 100%;
            font-size: 1.1rem;
        }
        .stTextInput input {
            font-size: 1.1rem;
        }
        .stSelectbox select {
            font-size: 1rem;
        }
        .stTextInput, .stTextArea {
            width: 100%;
        }
        .stMarkdown, .stCaption {
            font-size: 1rem;
        }
        .stRadio label, .stSelectbox div {
            font-size: 1rem;
        }
    }

    /* Default and Larger Screens */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)




#streamlit
st.title("🧠 Talk To PDF")
st.caption("🚀 Upload PDFs, ask questions, and get responses instantly")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["OpenAI", "deepseek-r1:1.5b"],
        index=0
    )
    
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 💡 Contextual Question Answering
    - 💡 Multi-Page Search and Summarization
    - 💡 Text Extraction and Processing
    - 💡 Semantic Understanding and Paraphrasing
    """)
    
    st.divider()
    st.markdown("Built with [OpenAI](https://open.ai/) | [LangChain](https://python.langchain.com/)")
    st.caption("Developed by Vikash")

    # Monetization Section
    st.markdown("### Monetization Options")
    subscription_plan = st.radio("Choose your plan:", ["Free", "Premium"], index=0)

    if subscription_plan == "Free":
        st.write("📢 Ads will be shown to free users.")
        # Placeholder for an ad network
        st.markdown("<h3>Advertisement</h3><p>Consider upgrading to Premium to remove ads.</p>", unsafe_allow_html=True)
    
    elif subscription_plan == "Premium":
        st.write("🔒 Premium features unlocked! Enjoy an ad-free experience and faster responses.")

    st.divider()
    st.markdown("### License our Technology")
    st.markdown("""
    Interested in licensing DeepSeek's search technology for your business?
    [Contact us](mailto:your.email@example.com) for more information or to schedule a demo!
    """)
from backend import extract_text_from_pdf
from backend import split_text
from backend import ask_deepseek
pc=pinecone.Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
#state management
my_embedding_model=OpenAIEmbeddings(api_key=st.secrets["openai_api_key"],model="text-embedding-ada-002")
# Chat container
uploaded_file=st.file_uploader("Upload",type=["pdf"])

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    with st.spinner("Processing PDF... please wait"):
        text1 = extract_text_from_pdf(st.session_state.uploaded_file)
        text = split_text(text1)
    st.session_state.text = text
    st.session_state.pdf_processed = True
    st.success("PDF stored and processed successfully!")
if "index" not in st.session_state:
    if "my-deepseek-pdf-project1" not in pc.list_indexes().names():  
        pc.create_index(
   #     embedings=my_embedding_model,
        name="my-deepseek-pdf-project1",
        dimension=1536,  # Make sure this matches your embeddings
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")        
                       )
        st.session_state.index=pc.Index("my-deepseek-pdf-project1")
    

if "pdf_processed" in st.session_state and st.session_state.pdf_processed:
    query = st.text_input("Ask a question based on the PDF content:")
    if query: 
        with st.spinner("Processing your question... Please wait"):            
            st.session_state.query=query
            answer=ask_deepseek(st.session_state.query,st.session_state.text)
            st.write(" Your answer is....")
            st.success(answer)
    if st.button("Upload a new PDF"):
       st.session_state.clear()
       st.session_state.uploaded_file=None
       st.session_state.pdf_processed=False
       st.session_state.query=None
       st.session_state.text = None  # Clear the processed text
       
       
       if "index" in st.session_state:
        st.session_state.index.delete(delete_all=True)
       st.success("State has been reset.. You can upload another PDF now..")
    

    

            
