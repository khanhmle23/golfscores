import json
import openai
from tqdm import tqdm
import os

# ✅ Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY") or "your-openai-key-here"

# ✅ Load structured golf data from the new file
with open("output_myrtle.json", "r") as f:
    golf_data = json.load(f)

# ✅ New chunking function that includes par values in the chunk
def create_chunks_with_pars(data: dict):
    chunks = []
    for round_name, round_info in data.items():
        par_dict = round_info["par"]
        players = round_info["players"]

        # Gather all player scores for this round
        players_scores = {player: pdata["scores"] for player, pdata in players.items()}
        skins = calculate_skins(players_scores)

        for player, pdata in players.items():
            handicap = pdata["handicap"]
            scores = pdata["scores"]
            stats = compute_score_stats(scores, par_dict, handicap)
            chunk_text = f"Round {round_name} at Course - Player: {player}, Handicap: {handicap}. Scores per hole: " + \
    "; ".join([f"{hole}: {score} (Par {par_dict[hole]})" for hole, score in scores.items()]) + "."

            chunk = {
                "round": round_name,
                "player": player,
                "text": chunk_text,
                "scores": scores,
                "par": par_dict,
                "handicap": handicap,
                "skins_won": skins.get(player, 0)
            }
            chunk.update(stats)  # <-- This flattens all stats fields to the top level
            chunks.append(chunk)
    return chunks

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

# ✅ Embedding generator
def embed_chunks(chunks):
    embedded = []
    client = openai.OpenAI()  # New client object for openai>=1.0.0
    for chunk in tqdm(chunks, desc="Generating embeddings"):
        response = client.embeddings.create(
            input=chunk["text"],
            model="text-embedding-3-small"
        )
        chunk["embedding"] = response.data[0].embedding
        embedded.append(chunk)
    return embedded

def calculate_skins(players_scores):
    """
    players_scores: dict of {player_name: {hole_name: score, ...}, ...}
    Returns: dict of {player_name: number_of_skins_won}
    """
    # Get all hole names in order
    hole_names = list(next(iter(players_scores.values())).keys())
    skins = {player: 0 for player in players_scores}
    carry = 0  # Number of skins carried over
    for hole in hole_names:
        # Find the lowest score for this hole
        scores = {player: scores[hole] for player, scores in players_scores.items()}
        min_score = min(scores.values())
        winners = [player for player, score in scores.items() if score == min_score]
        if len(winners) == 1:
            # Only one winner, award skins (including any carried over)
            skins[winners[0]] += 1 + carry
            carry = 0
        else:
            # Tie, carry over the skin
            carry += 1
    return skins

def init_stats():
    return {
        "eagle": 0, "birdie": 0, "par": 0, "bogey": 0,
        "double_bogey": 0, "other": 0, "gross_score": 0, "net_score": 0,
        "front9_gross": 0, "front9_net": 0, "back9_gross": 0, "back9_net": 0
    }

def accumulate(total_stats, round_stats):
    for key in total_stats:
        if key in round_stats:
            total_stats[key] += round_stats[key]

# New function to create golf chunks
def create_golf_chunks(golf_data):
    all_chunks = []
    for player, pdata in golf_data["players"].items():
        total_stats = init_stats()
        for round_name, round_info in golf_data["rounds"].items():
            scores = round_info["players"][player]["scores"]
            par = round_info["par"]
            handicap = pdata["handicap"]
            stats = compute_score_stats(scores, par, handicap)
            chunk = {
                "player": player,
                "round": round_name,
                "course": round_info["course"],
                "handicap": handicap,
                "scores": scores,
                "par": par,
                **stats,
                "type": "round"
            }
            all_chunks.append(chunk)
            accumulate(total_stats, stats)
        # Add total chunk
        total_chunk = {
            "player": player,
            "round": "Total",
            "course": "All",
            "handicap": pdata["handicap"],
            **total_stats,
            "type": "total"
        }
        all_chunks.append(total_chunk)
    return all_chunks

# ✅ Run pipeline
chunks = create_chunks_with_pars(golf_data)
embedded_chunks = embed_chunks(chunks)

# ✅ Save to a new JSON file
with open("golf_embeddings_myrtle.json", "w") as f:
    json.dump(embedded_chunks, f, indent=2)

print("✅ Saved updated embeddings to golf_embeddings_myrtle.json")
