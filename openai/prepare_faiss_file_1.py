import pandas as pd
import json
import faiss
import numpy as np
from openai import OpenAI
import os

# Set OpenAI key from environment
client = OpenAI(api_key=os.getenv("OUR_OPEN_API_KEY"))

# File paths
csv_file = "opencart_modules_data_1.csv"
json_file = "opencart_modules_chunks_openai_rag.json"
faiss_file = "opencart_modules_index.faiss"

# Step 1: Read CSV
df = pd.read_excel("opencart_data_1.xlsx")
df.columns = df.columns.str.strip()

main_cols = [
    "OpenCart Module OpenCart Extension Name",
    "Price",
    "Description or Features",
    "Compatibility wwith OpenCart Versions",
    "Shop URL or Buy URL",
    "Demo Link",
    "Admin Demo Link"
]
df = df[main_cols]

# Optional: print available columns for reference
print("🧾 CSV Columns:", df.columns.tolist())

# Drop rows with missing title or description
df = df[df["OpenCart Module OpenCart Extension Name"].notnull() & df["Description or Features"].notnull()].reset_index(drop=True)
print(f"✅ Loaded {len(df)} valid rows.")

# Step 2: Create JSON Chunks for RAG
def safe_str(val):
    if pd.isna(val):
        return ""
    return str(val).replace('\n', ' ').replace('\r', ' ').strip()

chunks = []
for _, row in df.iterrows():
    title = safe_str(row.get("OpenCart Module OpenCart Extension Name"))
    desc = safe_str(row.get("Description or Features"))
    price = safe_str(row.get("Price"))
    oc_versions = safe_str(row.get("Compatibility wwith OpenCart Versions"))
    url = safe_str(row.get("Shop URL or Buy URL"))
    demo_link = safe_str(row.get("Demo Link"))
    admin_demo_link = safe_str(row.get("Admin Demo Link"))
    chunk = (
        f"OpenCart Module: {title}. "
        f"Description: {desc}. "
        f"Price: {price}. "
        f"Supported OpenCart Versions: {oc_versions}. "
        f"Learn more: {url}"
    )
    chunks.append(chunk)

# Save chunks to JSON
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
print(f"✅ Saved {len(chunks)} chunks to {json_file}")

# Step 3: Embed Chunks and Save FAISS Index
embedding_dim = 1536  # OpenAI text-embedding-3-small
index = faiss.IndexFlatL2(embedding_dim)
all_embeddings = []

print("⏳ Embedding chunks...")
for i, chunk in enumerate(chunks):
    response = client.embeddings.create(input=[chunk], model="text-embedding-3-small")
    embedding = response.data[0].embedding
    all_embeddings.append(embedding)
    if (i+1) % 10 == 0:
        print(f"  Embedded {i+1} / {len(chunks)}")

# Add all at once (faster and safer)
index.add(np.array(all_embeddings, dtype="float32"))
faiss.write_index(index, faiss_file)
print(f"✅ FAISS index saved to {faiss_file}")
