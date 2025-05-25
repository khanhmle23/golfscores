import re
import json

# Load all embedded chunks
with open("golf_embeddings.json", "r") as f:
    all_chunks = json.load(f)

# Dynamically collect unique player names from chunks
PLAYER_NAMES = sorted({chunk["player"] for chunk in all_chunks if "player" in chunk})

def extract_players_from_prompt(prompt: str) -> list:
    found = [name for name in PLAYER_NAMES if re.search(rf"\b{name}\b", prompt, re.IGNORECASE)]
    return found

def extract_rounds_from_prompt(prompt: str) -> list:
    # Look for things like "Round 1", "round 3", etc.
    return [f"RD {n}" for n in range(1, 5) if re.search(rf"\bround {n}\b", prompt, re.IGNORECASE)]

def get_relevant_chunks(prompt: str):
    players = extract_players_from_prompt(prompt)
    rounds = extract_rounds_from_prompt(prompt)

    # Filter all chunks where player and round match (if specified)
    return [
        chunk for chunk in all_chunks
        if chunk.get("player") in players and (not rounds or chunk.get("round") in rounds)
    ]
