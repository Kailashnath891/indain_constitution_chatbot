import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# Title
st.set_page_config(page_title="Indian Constitution Chatbot", page_icon="🇮🇳")
st.title("📜 Indian Constitution Chatbot")

# Load Constitution text
@st.cache_resource
def load_data():
    loader = TextLoader(
        "C:/Users/skail/OneDrive/Desktop/kailash_QT/new_ollama/indian_constitution.txt",
        encoding="utf-8"
    )
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    docs = text_splitter.split_documents(documents)
    embeddings = OllamaEmbeddings(model="llama3.2:3b")
    db = Chroma.from_documents(docs, embeddings)
    retriever = db.as_retriever()
    llm = OllamaLLM(model="llama3.2:3b")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa

qa = load_data()

# Chat interface
user_question = st.text_input("Ask a question about the Indian Constitution:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Thinking..."):
            result = qa.invoke(user_question)
            st.success(result["result"])
    else:
        st.warning("⚠️ Please enter a question!")
