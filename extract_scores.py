import pandas as pd
import json

def extract_all_rounds_from_workbook(file_path: str, output_json_path: str) -> dict:
    """
    Extracts structured golf data from all sheets in a workbook and saves it to a JSON file.

    Args:
        file_path (str): Path to the Excel file.
        output_json_path (str): Path where the JSON file should be saved.

    Returns:
        dict: Structured golf data from all rounds.
    """
    xls = pd.ExcelFile(file_path)
    all_data = {}

    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name)

        # Extract round number and course name from the top row
        course_value = df.iloc[0, 0]  # e.g., "Pebble Beach"
        round_value = df.iloc[0, 1]   # e.g., "1"

        # Hole headers and par values
        hole_headers = df.iloc[1, 2:20].tolist()
        par_values = df.iloc[2, 2:20].astype(int).tolist()
        par_dict = {hole: par for hole, par in zip(hole_headers, par_values)}

        # Extract player data
        players_dict = {}
        for idx in range(3, len(df)):
            row = df.iloc[idx]
            player_name = row.iloc[0]
            if pd.isnull(player_name):
                continue
            handicap = int(row.iloc[1])
            scores = {hole: int(row.iloc[i]) for i, hole in enumerate(hole_headers, start=2) if pd.notnull(row.iloc[i])}
            players_dict[player_name] = {
                "handicap": handicap,
                "scores": scores
            }

        # Store structured round data
        all_data[sheet_name] = {
            "round": round_value,
            "course": course_value,
            "par": par_dict,
            "players": players_dict
        }

    # Write to JSON file
    with open(output_json_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"✅ Golf data exported to {output_json_path}")
    return all_data

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract golf scores from Excel and save as JSON.")
    parser.add_argument("file_path", help="Path to the Excel file")
    parser.add_argument("output_json_path", help="Path to save the output JSON file")
    args = parser.parse_args()

    extract_all_rounds_from_workbook(args.file_path, args.output_json_path)
