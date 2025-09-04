import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
# prefer the new package if you installed it:
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # fallback
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Indian Constitution Chatbot", page_icon="🇮🇳")
st.title("📜 Indian Constitution Chatbot (Fast Mode ⚡)")

DOC_PATH = r"C:\Users\skail\OneDrive\Desktop\kailash_QT\new_ollama\indian_constitution.txt"
PERSIST_DIR = "chroma_db"

# Optional: one-click rebuild
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("♻️ Rebuild Index"):
        if os.path.exists(PERSIST_DIR):
            import shutil
            shutil.rmtree(PERSIST_DIR)
        st.success("Index cleared. It will rebuild on next query.")

@st.cache_resource(show_spinner=True)
def load_qa():
    # 1) Embeddings (fast)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 2) Vector DB: build if missing, else load
    if os.path.exists(PERSIST_DIR):
        db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
    else:
        if not os.path.exists(DOC_PATH):
            st.error(f"❌ File not found: {DOC_PATH}")
            st.stop()
        docs_raw = TextLoader(DOC_PATH, encoding="utf-8").load()
        chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs_raw)
        if len(chunks) == 0:
            st.error("❌ No chunks created. Check your document content/encoding.")
            st.stop()
        db = Chroma.from_documents(chunks, embeddings, persist_directory=PERSIST_DIR)

    retriever = db.as_retriever(search_kwargs={"k": 6})

    # 3) LLM (fast model; change to llama3 if you prefer larger)
    llm = OllamaLLM(model="llama3.2:1b")

    # 4) A clearer prompt
    template = """
You are a helpful assistant answering questions about the Constitution of India.
Use ONLY the context to answer. If the answer is not in the context, say:
"I don't know based on the provided document."
Keep answers concise (3–6 sentences). Cite Article/Part if present.

Context:
{context}

Question: {question}
Answer:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa

qa = load_qa()

user_question = st.text_input("Ask a question about the Indian Constitution:")
if user_question:
    # Friendly guard for greetings
    if user_question.strip().lower() in {"hi", "hello", "hey"}:
        st.info("👋 Hi! Ask me something like: “What does Article 14 say?”")
    else:
        with st.spinner("🤔 Thinking..."):
            result = qa.invoke({"query": user_question})

        st.success(result["result"])

        # Show retrieved sources (helps debug why it said “I don’t know”)
        with st.expander("🔎 Show sources"):
            srcs = result.get("source_documents", [])
            st.write(f"Retrieved {len(srcs)} chunk(s).")
            for i, d in enumerate(srcs, 1):
                st.markdown(f"**Source {i}** — {d.metadata.get('source','(unknown)')}")
                st.code(d.page_content[:1200])
