# ==============================
# Medical Chatbot - Complete Version
# ==============================

import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Medical Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Medical Chatbot")
st.markdown(
    "**Educational purposes only. Not a substitute for professional medical advice.**"
)
st.markdown(
    "**Note:** This chatbot answers queries related to only 10 diseases: "
    "Malaria, Dengue, Tuberculosis, Diabetes, Hypertension, Anemia, Asthma, Influenza, COVID-19, Common Cold."
)

# ---------------- SESSION STATE ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- LOAD MODEL & INDEX ----------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

embedding_model = load_embedding_model()

@st.cache_resource
def load_faiss_index():
    index = faiss.read_index("medical_index.faiss")
    with open("file_mapping.txt", "r") as f:
        filenames = [line.strip() for line in f.readlines()]
    return index, filenames

index, filenames = load_faiss_index()
DATA_FOLDER = "data"

# ---------------- DISEASE LIST ----------------
DISEASES = [
    "malaria", "dengue", "tuberculosis", "diabetes", "hypertension",
    "anemia", "asthma", "influenza", "covid-19", "common cold"
]

# ---------------- UTILITY FUNCTIONS ----------------
def normalize_text(text: str):
    """Lowercase and remove extra spaces."""
    return text.lower().strip()

def find_relevant_disease(query: str):
    """Check if query mentions any disease from dataset (even with minor typos)."""
    query_norm = normalize_text(query)
    for disease in DISEASES:
        pattern = re.compile(rf"\b{disease[:3]}", re.IGNORECASE)  # match first 3 letters
        if pattern.search(query_norm):
            return disease
    return None

# ---------------- RAG RESPONSE ----------------
def generate_rag_response(user_input: str):
    # Check if query is about a known disease
    disease = find_relevant_disease(user_input)
    if not disease:
        return f"""
The question (**{user_input}**) is outside the scope of this chatbot.
This chatbot only answers queries related to these 10 diseases: {', '.join(DISEASES)}.
"""

    # Convert query to embedding
    query_embedding = embedding_model.encode(
        [user_input], convert_to_numpy=True, normalize_embeddings=True
    )

    # Search FAISS index
    distances, indices = index.search(query_embedding, k=1)
    best_match = filenames[indices[0][0]]
    file_path = os.path.join(DATA_FOLDER, best_match)

    # Read file safely (handle Unicode issues)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            retrieved_content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin1") as f:
            retrieved_content = f.read()

    # Prepare human-readable answer
    answer = f"""
You asked: {user_input}

Based on the dataset for **{disease.capitalize()}**:
{retrieved_content}

This is educational only and not a substitute for professional medical advice.
"""
    return answer

# ---------------- STREAMLIT UI ----------------
user_input = st.text_input("Enter symptoms or disease name:")

if user_input.strip():
    response = generate_rag_response(user_input)
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Chatbot", response))

# Display chat history
for sender, message in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Chatbot:** {message}")
