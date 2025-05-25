import json
import faiss
import numpy as np
import openai
import os

# ✅ Set your OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

# ✅ Load embedded chunks from JSON
with open("golf_embeddings.json", "r") as f:
    embedded_chunks = json.load(f)

# ✅ Build the FAISS index
dimension = len(embedded_chunks[0]["embedding"])
index = faiss.IndexFlatL2(dimension)

# Collect vectors and metadata
vectors = []
metadata = []

for chunk in embedded_chunks:
    vectors.append(np.array(chunk["embedding"], dtype="float32"))
    metadata.append(chunk)

# Convert to numpy array and add to index
index.add(np.array(vectors))

# ✅ Function to embed user query
def embed_query(text):
    client = openai.OpenAI()  # New client object for openai>=1.0.0
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ✅ Function to search index
def search(query, top_k=3):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results

def compute_score_stats(scores, par_dict):
    stats = {"eagle": 0, "birdie": 0, "par": 0, "bogey": 0, "double_bogey": 0, "other": 0}
    for hole, score in scores.items():
        par = par_dict.get(hole)
        if par is None:
            continue
        diff = score - par
        if diff == -2:
            stats["eagle"] += 1
        elif diff == -1:
            stats["birdie"] += 1
        elif diff == 0:
            stats["par"] += 1
        elif diff == 1:
            stats["bogey"] += 1
        elif diff == 2:
            stats["double_bogey"] += 1
        else:
            stats["other"] += 1
    return stats

# ✅ Interactive loop
while True:
    user_input = input("\nAsk a question about the golf scores (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    results = search(user_input)
    print("\n🔍 Top Relevant Chunks:")
    for i, res in enumerate(results, 1):
        stats = compute_score_stats(res.get("scores", {}), res.get("par", {}))
        print(f"\nResult {i} ({res['player']} - {res['round']}):\n{res['text']}")
        print(f"  Birdies: {stats['birdie']}, Eagles: {stats['eagle']}, Pars: {stats['par']}, Bogeys: {stats['bogey']}, Double Bogeys: {stats['double_bogey']}, Other: {stats['other']}")
