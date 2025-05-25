import json
import openai
import numpy as np
import faiss
import os
import re

# ✅ Load the new embeddings file
with open("golf_embeddings_myrtle.json", "r") as f:
    metadata = json.load(f)

# Build FAISS index from embeddings
embeddings = np.array([chunk["embedding"] for chunk in metadata]).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

def embed_query(text):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "your-openai-key-here")
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

# ✅ Search FAISS index
def search(query, top_k=3):
    query_vector = embed_query(query).reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    return [metadata[i] for i in indices[0]]

# ✅ Ask GPT-4 with context
def ask_gpt(query, context_chunks):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "your-openai-key-here")
    context_text = "\n".join([
        f"{chunk['player']} | Round: {chunk['round']} | "
        f"Handicap: {chunk.get('handicap', 'N/A')} | "
        f"Gross Score: {chunk['stats'].get('gross_score', 'N/A')}, Net Score: {chunk['stats'].get('net_score', 'N/A')}, "
        f"Front 9 Gross: {chunk['stats'].get('front9_gross', 'N/A')}, Front 9 Net: {chunk['stats'].get('front9_net', 'N/A')}, "
        f"Back 9 Gross: {chunk['stats'].get('back9_gross', 'N/A')}, Back 9 Net: {chunk['stats'].get('back9_net', 'N/A')}, "
        f"Birdies: {chunk['stats'].get('birdie', 'N/A')}, Eagles: {chunk['stats'].get('eagle', 'N/A')}, Pars: {chunk['stats'].get('par', 'N/A')}, "
        f"Bogeys: {chunk['stats'].get('bogey', 'N/A')}, Skins Won: {chunk.get('skins_won', 'N/A')} | "
        f"Scores: " + ", ".join(
            f"{hole}: {score} (Par {chunk.get('par', {}).get(hole, 'N/A')})"
            for hole, score in chunk.get('scores', {}).items()
        )
        for chunk in context_chunks
    ])
    system_prompt = (
        "You are a golf tournament assistant. "
        "The user may define teams in their question. "
        "Use the context to compute individual or team performance. "
        "Only base your answer on the data provided."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content

def compute_score_stats(scores, par_dict, handicap=0):
    stats = {
        "eagle": 0, "birdie": 0, "par": 0, "bogey": 0,
        "double_bogey": 0, "other": 0, "gross_score": 0, "net_score": 0,
        "front9_gross": 0, "front9_net": 0, "back9_gross": 0, "back9_net": 0
    }
    gross_score = 0
    front9_gross = 0
    back9_gross = 0
    holes = sorted(scores.keys(), key=lambda h: int(h.split()[-1]))
    for hole in holes:
        score = scores[hole]
        par = par_dict.get(hole)
        if par is None:
            continue
        gross_score += score
        hole_num = int(hole.split()[-1])
        if hole_num <= 9:
            front9_gross += score
        else:
            back9_gross += score
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
    stats["gross_score"] = gross_score
    stats["net_score"] = gross_score - handicap
    stats["front9_gross"] = front9_gross
    stats["back9_gross"] = back9_gross
    # Split handicap proportionally (half to front, half to back)
    stats["front9_net"] = front9_gross - (handicap // 2)
    stats["back9_net"] = back9_gross - (handicap - (handicap // 2))
    return stats

def calculate_skins(players_scores):
    """
    players_scores: dict of {player_name: {hole_name: score, ...}, ...}
    Returns: dict of {player_name: number_of_skins_won}
    """
    hole_names = list(next(iter(players_scores.values())).keys())
    skins = {player: 0 for player in players_scores}
    carry = 0
    for hole in hole_names:
        scores = {player: scores[hole] for player, scores in players_scores.items()}
        min_score = min(scores.values())
        winners = [player for player, score in scores.items() if score == min_score]
        if len(winners) == 1:
            skins[winners[0]] += 1 + carry
            carry = 0
        else:
            carry += 1
    return skins

def build_summary_table(chunks):
    lines = [
        "Player | Round | Handicap | Gross | Net | Front9 | Front9 Net | Back9 | Back9 Net | Birdies | Eagles | Skins"
    ]
    for chunk in chunks:
        stats = compute_score_stats(
            chunk.get("scores", {}),
            chunk.get("par", {}),
            chunk.get("handicap", 0)
        )
        lines.append(
            f"{chunk['player']} | {chunk['round']} | {chunk.get('handicap', 'N/A')} | "
            f"{stats['gross_score']} | {stats['net_score']} | "
            f"{stats['front9_gross']} | {stats['front9_net']} | "
            f"{stats['back9_gross']} | {stats['back9_net']} | "
            f"{stats['birdie']} | {stats['eagle']} | {chunk.get('skins_won', 'N/A')}"
        )
    return "\n".join(lines)

def extract_round_number(question):
    match = re.search(r'round\s*:?[\s\-]*(\d+)', question, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

# ✅ Interactive loop
while True:
    user_input = input("\nAsk a golf question (or type 'exit'): ")
    if user_input.lower() == "exit":
        break

    round_num = extract_round_number(user_input)
    if round_num:
        # Filter all player chunks for that round
        round_chunks = [c for c in metadata if str(c.get("round", "")).strip().lower() in [f"round {round_num}", round_num]]
        if round_chunks:
            context_text = build_context(round_chunks)
            answer = ask_gpt(user_input, context_text)
            print(answer)
            continue  # Skip the rest of the loop

    # If user requests a summary/table/all players, show all
    if any(word in user_input.lower() for word in ["summary", "table", "all players"]):
        summary_table = build_summary_table(metadata)
        print("\n📊 All Players Summary Table:\n")
        print(summary_table)
        print("\n🤖 GPT-4's Answer:")
        answer = ask_gpt(user_input, [
            {
                "player": "ALL",
                "round": "ALL",
                "handicap": "",
                "stats": {},
                "skins_won": "",
                "scores": {},
                "par": {},
                "text": summary_table
            }
        ])
        print(answer)
        continue

    # Otherwise, do the normal top_chunks search
    top_chunks = search(user_input, top_k=14)
    print("\n🔍 Top matching data:")
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n{i}. {chunk['player']} - {chunk['round']}\n{chunk['text']}")
        stats = compute_score_stats(
            chunk.get("scores", {}),
            chunk.get("par", {}),
            chunk.get("handicap", 0)
        )
        skins_won = chunk.get("skins_won", "N/A")
        print(
            f"  Birdies: {stats['birdie']}, Eagles: {stats['eagle']}, Pars: {stats['par']}, "
            f"Bogeys: {stats['bogey']}, Other: {stats['other']}\n"
            f"  Gross Score: {stats['gross_score']}, Net Score: {stats['net_score']}\n"
            f"  Skins Won: {skins_won}"
        )

    # Collect all player scores for the round(s) in top_chunks
    players_scores = {}
    for chunk in top_chunks:
        players_scores[chunk["player"]] = chunk["scores"]

    if players_scores:
        skins = calculate_skins(players_scores)
        print("\n🏆 Skins Won:")
        for player, num_skins in skins.items():
            print(f"  {player}: {num_skins}")

    # Filter for Pebble Beach, RD 1
    pebble_chunks = [c for c in top_chunks if c["round"] == "RD 1 Scores" and "Pebble Beach" in c["text"]]

    if pebble_chunks:
        # Build players_scores for all 18 holes at Pebble Beach
        players_scores_pebble = {c["player"]: c["scores"] for c in pebble_chunks}
        skins_pebble = calculate_skins(players_scores_pebble)
        print("\n🏆 Skins Won at Pebble Beach:")
        for player, num_skins in skins_pebble.items():
            print(f"  {player}: {num_skins}")

    print("\n🤖 GPT-4's Answer:")
    answer = ask_gpt(user_input, top_chunks)
    print(answer)
