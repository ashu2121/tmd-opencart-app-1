from flask import Flask, request, jsonify
import openai
import faiss
import numpy as np
import json
import os

# 🔑 Set your OpenAI API Key
openai.api_key = os.getenv("OUR_OPEN_API_KEY")

# 🔁 Load FAISS index and metadata from JSON
index = faiss.read_index("opencart_modules_index.faiss")  # <-- use your .faiss file name
with open("opencart_modules_chunks_openai_rag.json", "r", encoding="utf-8") as f:
    texts = json.load(f)  # each chunk is a string

# 🌐 Initialize Flask app
app = Flask(__name__)
SYSTEM_MESSAGE = (
    "I did not find any answer related to your query. Please feel free to raise the ticket. "
    "Our support team will get back to you as soon as possible. Thanks. " 
    "https://www.opencartextensions.in/ticket"
)
SIMILARITY_THRESHOLD = 0.7  # Lower is more similar (for FAISS inner product/cosine)


@app.route("/chat", methods=["GET"])
def chat():
    try:
        user_query = request.args.get("query")
        if not user_query:
            return jsonify({"error": "Missing 'query' parameter"}), 400

        response = openai.embeddings.create(
            input=[user_query],
            model="text-embedding-3-small"
        )
        query_embedding = np.array(response.data[0].embedding).astype("float32").reshape(1, -1)

        D, I = index.search(query_embedding, k=10)
        # Option 1: All similarity scores are "not similar" (depends on distance metric)
        if all(d > SIMILARITY_THRESHOLD for d in D[0]):
            return jsonify({"response": SYSTEM_MESSAGE})

        relevant_chunks = [texts[i] for i in I[0] if i < len(texts)]
        context = "\n\n".join(relevant_chunks)
        prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {user_query}"

        completion = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        reply = completion.choices[0].message.content.strip()

        # Option 2: Check if reply is generic/empty/unhelpful
        if reply.lower().startswith("i don't know") or reply.lower().startswith("sorry") or len(reply) < 30:
            reply = SYSTEM_MESSAGE

        return jsonify({"response": reply})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 🚀 Start server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
