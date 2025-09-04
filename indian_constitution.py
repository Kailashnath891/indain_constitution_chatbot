from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# NEW Ollama imports
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA

# Load Indian Constitution text
print("🔹 Loading documents...")
loader = TextLoader(
    "C:/Users/skail/OneDrive/Desktop/kailash QT/new ollama/indian_constitution.txt",
    encoding="utf-8"
)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents")

# Split into chunks
print("🔹 Splitting into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)
print(f"✅ Created {len(docs)} chunks")

# Create embeddings + vector DB
print("🔹 Creating embeddings...")
embeddings = OllamaEmbeddings(model="llama3.2:3b")
db = Chroma.from_documents(docs, embeddings)
print("✅ Embeddings + DB ready")

# Create retriever
print("🔹 Creating retriever & LLM...")
retriever = db.as_retriever()
llm = OllamaLLM(model="llama3.2:3b")

# Create QA chain
print("🔹 Creating QA chain...")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask a question
print("🔹 Asking question...")
query = "What are the fundamental rights in the Indian Constitution?"
result = qa.invoke(query)

print("\n✅ Answer:")
print(result["result"])
