# ==============================
# Medical Chatbot - Clean Bullet Points Version
# ==============================

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Medical Chatbot")
st.markdown(
    "**Educational purposes only. Not a substitute for professional medical advice.**"
)
st.markdown(
    "**Note:** This chatbot answers queries related to 10 diseases including Anemia, Asthma, COVID-19, Dengue Fever, Diabetes, Foodborne Disease, Hypertension, Influenza, Malaria, and Tuberculosis."
)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD FAISS INDEX ----------------
@st.cache_resource
def load_faiss_index(data_folder="data"):
    documents = []
    for file in os.listdir(data_folder):
        if file.endswith(".txt") and not file.startswith("~$"):
            file_path = os.path.join(data_folder, file)
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    text = f.read().strip()
                    if text:
                        documents.append(Document(page_content=text, metadata={"source": file}))
            except Exception as e:
                print(f"Error reading {file}: {e}")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

vectorstore = load_faiss_index()

# ---------------- INITIALIZE LLM ----------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="openai/gpt-oss-20b"
    )

llm = load_llm()

# ---------------- SMART BULLET FORMAT FUNCTION ----------------
def format_bullets(text):
    """
    Properly formats LLM output:
    - Keeps lead sentences separate.
    - Splits remaining items into individual bullets.
    - Removes duplicates like '• •'.
    """
    text = text.replace("•", "|")  # mark all bullets
    parts = [p.strip() for p in text.split("|") if p.strip()]

    bullets = []
    lead_sentences = []

    for part in parts:
        # Treat as lead if it ends with ':' or contains 'may include', 'can cause', etc.
        if ':' in part or any(word in part.lower() for word in ["may include", "can cause", "may experience"]):
            lead_sentences.append(part)
        else:
            bullets.append(part)

    # Build final formatted output
    output = ""
    if lead_sentences:
        output += " ".join(lead_sentences) + "\n\n"

    output += "\n".join([f"• {b}" for b in bullets])
    return output

# ---------------- RAG FUNCTION ----------------
def ask_rag(question, k=3):
    # Retrieve relevant documents
    docs = vectorstore.similarity_search(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Prompt LLM
    prompt = f"""
You are a helpful medical assistant.
Answer the user's question using ONLY the context below.
If the answer is not present, say "I don't know".
- For symptoms, causes, or prevention, always list them in separate bullet points starting with "•".
- Start with a short lead sentence if needed (e.g., 'Most people may not have symptoms, but severe cases can cause:').

Context:
{context}

Question:
{question}
"""
    raw_answer = llm.invoke(prompt).content
    answer = format_bullets(raw_answer)
    return answer

# ---------------- STREAMLIT UI ----------------
user_input = st.text_input("Enter symptoms, causes, prevention, or disease name:")

if user_input.strip():
    answer = ask_rag(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", answer))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Chatbot:** {message}")
