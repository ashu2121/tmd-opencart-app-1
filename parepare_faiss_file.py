import json
import openai
import faiss
import numpy as np
import pickle
import os
# Set your OpenAI API key
openai.api_key = os.getenv("OUR_OPEN_API_KEY")
#

# Load your dataset
with open("rag_ready_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = [str(entry["text"]).strip() for entry in data if entry.get("text")]

# Break into chunks to avoid long input errors
batch_size = 100
all_embeddings = []

for i in range(0, len(texts), batch_size):
    chunk = texts[i:i + batch_size]
    try:
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(embeddings)
        time.sleep(1)  # avoid rate limiting
    except Exception as e:
        print(f"Error processing batch {i}: {e}")
        break

# Save FAISS index
embedding_vectors = np.array(all_embeddings).astype("float32")
index = faiss.IndexFlatL2(embedding_vectors.shape[1])
index.add(embedding_vectors)
faiss.write_index(index, "rag_index.faiss")

# Save metadata
with open("rag_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ Done: FAISS and text map saved.")
