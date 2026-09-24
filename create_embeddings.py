import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Absolute path (safe)
DATA_FOLDER = os.path.join(os.getcwd(), "data")

print("Looking inside folder:", DATA_FOLDER)
print("Files found:", os.listdir(DATA_FOLDER))

documents = []
filenames = []

# Read all txt files safely
for file in os.listdir(DATA_FOLDER):
    if file.endswith(".txt") and not file.startswith("~$"):
        file_path = os.path.join(DATA_FOLDER, file)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read().strip()
                if text:
                    documents.append(text)
                    filenames.append(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

print("Number of documents loaded:", len(documents))

if len(documents) == 0:
    raise ValueError("No documents found! Check folder name and .txt files.")

# Convert text to embeddings
embeddings = model.encode(documents)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Save index
faiss.write_index(index, "medical_index.faiss")

# Save mapping
with open("file_mapping.txt", "w", encoding="utf-8") as f:
    for name in filenames:
        f.write(name + "\n")

print("[SUCCESS] Embeddings created and stored successfully!")
