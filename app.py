
from flask import Flask, request, jsonify
import openai
import faiss
import numpy as np
import pickle
import os
# 🔑 Set your OpenAI API Key
openai.api_key = os.getenv("OUR_OPEN_API_KEY")

# 🔁 Load FAISS index and metadata
index = faiss.read_index("rag_index.faiss")
with open("rag_texts.pkl", "rb") as f:
    texts = pickle.load(f)  

# 🌐 Initialize Flask app
app = Flask(__name__)

@app.route("/chat", methods=["GET"])
def chat():
    try:
        user_query = request.args.get("query")
        if not user_query:
            return jsonify({"error": "Missing 'query' parameter"}), 400

        # Embed user query
        response = openai.embeddings.create(
            input=[user_query],
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

        # Search in FAISS
        D, I = index.search(query_embedding, k=3)
        relevant_chunks = [texts[i] for i in I[0]]

        # Build context
        context = "\n\n".join(relevant_chunks)
        prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"

        # Call OpenAI
        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = completion.choices[0].message.content.strip()
        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 Start server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)