import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import clip


DATA_DIR = "/root/Yelp/yelp_dataset/"
PHOTO_DIR = "/root/Yelp/yelp_photos/photos/"
BUSINESS_FILE = DATA_DIR + "yelp_academic_dataset_business.json"
REVIEW_FILE = DATA_DIR + "yelp_academic_dataset_review.json"
PHOTO_FILE = "/root/Yelp/yelp_photos/" + "photos.json"
OUTPUT_DIR = "/root/Yelp_dataset_LGMRec/"


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"


model, preprocess = clip.load("ViT-L/14", device=device)



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



def process_business():
    businesses = load_json(BUSINESS_FILE)
    business_df = pd.DataFrame(businesses)[['business_id', 'categories', 'attributes']]
    print(f"Extracted {len(business_df)} rows of business features.")
    return business_df



def process_photo():
    photos = load_json(PHOTO_FILE)
    photo_df = pd.DataFrame(photos)[['photo_id', 'business_id', 'caption', 'label']]
    
   
    if photo_df['photo_id'].isnull().any():
        print("Warning: Missing photo_id values found. Dropping these rows.")
        photo_df = photo_df.dropna(subset=['photo_id']).reset_index(drop=True)
    
    print(f"Extracted {len(photo_df)} rows of photo features.")
    return photo_df



def align_data(interaction, photo, business):
    
    valid_business_ids = set(interaction['business_id']).intersection(set(photo['business_id']))
    
    
    aligned_interaction = interaction[interaction['business_id'].isin(valid_business_ids)].reset_index(drop=True)
    
    
    aligned_photo = photo[photo['business_id'].isin(valid_business_ids)].reset_index(drop=True)
    if 'photo_id' not in aligned_photo.columns:
        raise ValueError("Error: 'photo_id' column is missing from aligned_photo.")
    

    aligned_business = business[business['business_id'].isin(valid_business_ids)].reset_index(drop=True)
    
    print(f"Aligned {len(aligned_interaction)} items with both text and images.")
    print(f"Aligned {len(aligned_business)} business entries.")
    return aligned_interaction, aligned_photo, aligned_business



def extract_image_features(photo_df, image_dir, batch_size=16):
    image_features = []
    image_paths = [Path(image_dir) / f"{photo_id}.jpg" for photo_id in photo_df['photo_id']]
    default_image_feature = torch.zeros(1, 768).to(device)

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting Image Features"):
        batch_paths = image_paths[i:i+batch_size]
        images = []
        for path in batch_paths:
            try:
                image = Image.open(path).convert("RGB")
                images.append(preprocess(image))
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                images.append(preprocess(Image.new("RGB", (224, 224))))

        if len(images) > 0:
            image_input = torch.stack(images).to(device)
            with torch.no_grad():
                batch_features = model.encode_image(image_input)
                batch_features = batch_features.cpu().numpy()
                image_features.extend(batch_features)  

    if len(image_features) == 0:
        print("Error: No valid image features extracted.")
        return np.array([default_image_feature.cpu().numpy()])  

    image_features = np.array(image_features)
    print(f"Extracted {image_features.shape[0]} image features with dimension {image_features.shape[1]}")
    return image_features



def extract_text_features(interaction, batch_size=16):
    text_features = []
    texts = interaction['text'].tolist()
    default_text_feature = torch.zeros(1, 768).to(device)

    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting Text Features"):
        batch_texts = texts[i:i+batch_size]
        try:
            with torch.no_grad():
                text_input = clip.tokenize(batch_texts, truncate=True).to(device)
                batch_features = model.encode_text(text_input).cpu().numpy()
                text_features.extend(batch_features)  
        except Exception as e:
            print(f"Error processing text batch {i}: {e}")
            text_features.extend([default_text_feature.cpu().numpy()] * len(batch_texts))  

    if len(text_features) == 0:
        print("Error: No valid text features extracted.")
        return np.array([default_text_feature.cpu().numpy()])  

    text_features = np.array(text_features)
    print(f"Extracted {text_features.shape[0]} text features with dimension {text_features.shape[1]}")
    return text_features



def run_clip_feature_extraction():

    interaction = process_interaction()
    business = process_business()
    photo = process_photo()

    
    filtered_interaction, filtered_photo, filtered_business = align_data(interaction, photo, business)

 
    image_features = extract_image_features(filtered_photo, PHOTO_DIR)
    if image_features is None or len(image_features) == 0:
        raise ValueError("Error: Failed to extract any image features.")
    np.save(f"{OUTPUT_DIR}/image_features.npy", image_features)

  
    text_features = extract_text_features(filtered_interaction)
    if text_features is None or len(text_features) == 0:
        raise ValueError("Error: Failed to extract any text features.")
    np.save(f"{OUTPUT_DIR}/text_features.npy", text_features)

   
    filtered_interaction.to_csv(f"{OUTPUT_DIR}/reviews_filtered.inter", index=False)
    print("Saved filtered interaction data to reviews_filtered.inter.")



run_clip_feature_extraction()
