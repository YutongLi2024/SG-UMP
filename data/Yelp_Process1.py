import json
import pandas as pd
from pathlib import Path


DATA_DIR = "/root/Yelp/yelp_dataset/"
REVIEW_FILE = DATA_DIR + "yelp_academic_dataset_review.json"
OUTPUT_DIR = "/root/Yelp_dataset_LGMRec/"


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def process_interaction():
    reviews = load_json(REVIEW_FILE)
    interaction = pd.DataFrame(reviews)[['user_id', 'business_id', 'text', 'stars', 'date']]
    interaction['timestamp'] = pd.to_datetime(interaction['date']).astype(int) // 10**9  
    interaction = interaction[['user_id', 'business_id', 'text', 'stars', 'timestamp']]
    print(f"Extracted {len(interaction)} rows of interaction data.")
    return interaction


def save_interaction_json(interaction):
    interaction_dict = interaction.to_dict(orient='records')  
    with open(f"{OUTPUT_DIR}/inter.json", 'w', encoding='utf-8') as f:
        json.dump(interaction_dict, f, ensure_ascii=False, indent=4)
    print(f"Saved interaction data to {OUTPUT_DIR}/inter.json")


def run_interaction_processing():
    
    interaction = process_interaction()

    
    save_interaction_json(interaction)


run_interaction_processing()
