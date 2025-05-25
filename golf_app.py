import streamlit as st
import json
import openai
import numpy as np
import re

# Load your embeddings and metadata
dataset_path = "golf_embeddings_myrtle.json"
with open(dataset_path, "r") as f:
    chunks = json.load(f)

# Dynamically extract unique player names
player_names = sorted({chunk["player"] for chunk in chunks if "player" in chunk})

# Set your OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else None
if not openai_api_key:
    openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
client = openai.OpenAI(api_key=openai_api_key)

# --- Helpers ---

def embed_query(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return np.array(response.data[0].embedding, dtype="float32")

def search(query, top_k=4):
    query_vec = embed_query(query)
    embeddings = np.array([chunk["embedding"] for chunk in chunks])
    sims = embeddings @ query_vec / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    top_indices = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def extract_round_number(question):
    match = re.search(r'round\s*:?[\s\-]*(\d+)', question, re.IGNORECASE)
    if match:
        return match.group(1)
    return None

def extract_players_from_question(question):
    return [name for name in player_names if re.search(rf"\\b{name}\\b", question, re.IGNORECASE)]

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
    net_score = gross_score - handicap
    front9_net = front9_gross - (handicap // 2)
    back9_net = back9_gross - (handicap - (handicap // 2))
    stats.update({
        "gross_score": gross_score,
        "net_score": net_score,
        "front9_gross": front9_gross,
        "front9_net": front9_net,
        "back9_gross": back9_gross,
        "back9_net": back9_net
    })
    return stats

def build_summary_table(chunks):
    headers = [
        ("Player", 12), ("Round", 8), ("Handicap", 8), ("Gross", 6), ("Net", 6),
        ("Front9", 7), ("Front9 Net", 10), ("Back9", 7), ("Back9 Net", 10),
        ("Birdies", 7), ("Eagles", 7), ("Skins", 6)
    ]
    header_row = " | ".join(h[0].ljust(h[1]) for h in headers)
    lines = [header_row, "-" * len(header_row)]
    for chunk in chunks:
        row = [
            str(chunk['player']).ljust(12),
            str(chunk['round']).ljust(8),
            str(chunk.get('handicap', 'N/A')).ljust(8),
            str(chunk.get('gross_score', 'N/A')).ljust(6),
            str(chunk.get('net_score', 'N/A')).ljust(6),
            str(chunk.get('front9_gross', 'N/A')).ljust(7),
            str(chunk.get('front9_net', 'N/A')).ljust(10),
            str(chunk.get('back9_gross', 'N/A')).ljust(7),
            str(chunk.get('back9_net', 'N/A')).ljust(10),
            str(chunk.get('birdie', 'N/A')).ljust(7),
            str(chunk.get('eagle', 'N/A')).ljust(7),
            str(chunk.get('skins', chunk.get('skins_won', 'N/A'))).ljust(6)
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)

def build_context(chunks):
    return "\n".join([
        f"{chunk['player']} | Round: {chunk['round']} | "
        f"Front9 Gross: {chunk.get('front9_gross','N/A')}, "
        f"Front9 Net: {chunk.get('front9_net','N/A')}, "
        f"Back9 Gross: {chunk.get('back9_gross','N/A')}, "
        f"Back9 Net: {chunk.get('back9_net','N/A')}, "
        f"Gross: {chunk.get('gross_score','N/A')}, Net: {chunk.get('net_score','N/A')}, "
        f"Birdies: {chunk.get('birdie','N/A')}, Eagles: {chunk.get('eagle','N/A')}, "
        f"Skins: {chunk.get('skins', chunk.get('skins_won', 'N/A'))} | "
        "Scores: " + ", ".join(
            f"{hole}: {score} (Par {chunk['par'][hole] if isinstance(chunk.get('par', {}), dict) and hole in chunk['par'] else 'N/A'})"
            for hole, score in chunk.get('scores', {}).items()
        )
        for chunk in chunks
    ])

def ask_gpt(query, context_text):
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

# --- Streamlit UI ---
st.title("Golf Trip Scoring Assistant")
st.markdown(f"**Dataset:** `{dataset_path}`")

# Add a label with sample prompts/tips for users
st.markdown("""
**💡 Sample Prompts:**
- What score did John get for hole 1 round 1? 
- Give me a summary of skins for round 1.
- Give me a summary of front 9 net scores for round 1?
- Give me a summary of back 9 net scores for round 1?
- Give me a summary for Birdies by round.
- Give me a summary of who got Birdies by round.
- Give me a summary of player total net scores for round 1.
- Danh and TJ are Team A, Orlando and Dai are Team B, Eric and Bryce are Team C, Mario and Ryan are Team D, Vuong and Wynn are Team E, Phillip and John are Team F, Jay and Tom are Team G.  Give me a summary of all team scores for front 9 round 1.
""")

# Add a toggle to show/hide the summary table (default: invisible)
show_summary = st.checkbox("Show All Players Summary Table", value=False)

summary_table = build_summary_table(chunks)
if show_summary:
    st.subheader("📊 All Players Summary Table")
    st.code(summary_table)

question = st.text_input("Ask a golf question:")

if question and openai_api_key:
    with st.spinner("Searching and thinking..."):
        round_num = extract_round_number(question)
        if round_num:
            round_chunks = [c for c in chunks if str(c.get("round", "")).strip().lower() in [f"round {round_num}", round_num]]
            if round_chunks:
                st.subheader(f"🔍 All Player Chunks for Round {round_num}")
                for i, chunk in enumerate(round_chunks, 1):
                    st.markdown(
                        f"**{i}. {chunk['player']} - {chunk['round']}**\n\n"
                        f"Handicap: {chunk.get('handicap', 'N/A')}\n\n"
                        f"Scores: {chunk.get('scores', {})}\n\n"
                        f"Text: {chunk.get('text', '')}\n\n"
                        "---"
                    )
                context_text = build_context(round_chunks)
                st.subheader("🧠 GPT-4's Answer")
                answer = ask_gpt(question, context_text)
                st.write(answer)
                st.stop()

        mentioned_players = extract_players_from_question(question)
        if mentioned_players:
            matched_chunks = [c for c in chunks if c["player"] in mentioned_players]
            st.subheader(f"🔍 All Chunks for Mentioned Players: {', '.join(mentioned_players)}")
            for i, chunk in enumerate(matched_chunks, 1):
                st.markdown(
                    f"**{i}. {chunk['player']} - {chunk['round']}**\n\n"
                    f"Handicap: {chunk.get('handicap', 'N/A')}\n\n"
                    f"Scores: {chunk.get('scores', {})}\n\n"
                    f"Text: {chunk.get('text', '')}\n\n"
                    "---"
                )
            context_text = build_context(matched_chunks)
            st.subheader("🧠 GPT-4's Answer")
            answer = ask_gpt(question, context_text)
            st.write(answer)
            st.stop()

        if any(word in question.lower() for word in ["summary", "table", "all players"]):
            st.subheader("🧠 GPT-4's Answer")
            answer = ask_gpt(question, summary_table)
            st.write(answer)
        else:
            # Only use embedding search if not a round or player query
            top_chunks = search(question, top_k=14)
            # st.subheader("🔍 Top Matching Chunks")
            # for i, chunk in enumerate(top_chunks, 1):
            #     st.markdown(
            #         f"**{i}. {chunk['player']} - {chunk['round']}**\n\n"
            #         f"Handicap: {chunk.get('handicap', 'N/A')}\n\n"
            #         f"Scores: {chunk.get('scores', {})}\n\n"
            #         f"Text: {chunk.get('text', '')}\n\n"
            #         "---"
            #     )
            context_text = build_context(top_chunks)
            st.subheader("🧠 GPT-4's Answer")
            answer = ask_gpt(question, context_text)
            st.write(answer)
