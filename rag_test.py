import os
import glob
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import sys

sys.stdout.reconfigure(encoding='utf-8')
load_dotenv()

# -------------------------------
# 1. Load text files safely
# -------------------------------
data_folder = "data"
files = glob.glob(os.path.join(data_folder, "*.txt"))

documents = []

for file in files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(file, "r", encoding="latin-1") as f:
            text = f.read()

    documents.append(
        Document(
            page_content=text.strip(),
            metadata={"source": os.path.basename(file)}
        )
    )
    print(f"Loaded: {file}")

print(f"\n[INFO] Total documents loaded: {len(documents)}")

# -------------------------------
# 2. Create embeddings (CORRECT way)
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embeddings
)

print("[SUCCESS] FAISS index created")

# -------------------------------
# 3. Initialize Groq LLM
# -------------------------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-20b"
)

# -------------------------------
# 4️⃣ RAG function
# -------------------------------
def ask_rag(question, k=3):
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
You are a medical assistant.
Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{question}
"""
    return llm.invoke(prompt).content

# -------------------------------
# 5️⃣ Test RAG
# -------------------------------
query = "What are the symptoms of hypertension?"
answer = ask_rag(query)

print("\n[ANSWER]:")
print(answer)
