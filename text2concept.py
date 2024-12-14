import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression

# Step 1: Load the Bias in Bios dataset
splits = {
    'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
    'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet',
    'dev': 'data/dev-00000-of-00001-e6551072fff26949.parquet'
}

train_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["test"])
dev_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["dev"])

print("All unique occupations:")
print(train_df['profession'].unique())

# Step 2: Prepare the data
label_encoder = LabelEncoder()
train_df['profession_encoded'] = label_encoder.fit_transform(train_df['profession'])
test_df['profession_encoded'] = label_encoder.transform(test_df['profession'])

y_train = train_df['profession_encoded']
y_test = test_df['profession_encoded']

# Load the CLIP model and processor for Text2Concept
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Step 3: Define male, female, and neutral concepts using professional context
concepts = {
    'male': [
        "a man working", "he is a leader", "he is strong", "his achievements"
    ],
    'female': [
        "a woman working", "she is kind", "she is caring", "her accomplishments"
    ],
    'neutral': [
        "a task at a workplace", "a meeting at work", "team effort at work"
    ]
}

# Generate concept vectors using CLIP's text encoder
def get_concept_vectors(concepts, model, processor):
    vectors = {}
    for key, phrases in concepts.items():
        inputs = processor(text=phrases, return_tensors="pt", padding=True)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
            vectors[key] = embeddings.mean(dim=0)  # Average across phrases
    return vectors

concept_vectors = get_concept_vectors(concepts, clip_model, clip_processor)

# Step 4: Calculate CAVs
def calculate_cav(concept_key, concept_vectors, random_texts, model, processor):
    random_inputs = processor(text=random_texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        random_embeddings = model.get_text_features(**random_inputs)
        random_embeddings = random_embeddings / random_embeddings.norm(dim=-1, keepdim=True)
    
    concept_embedding = concept_vectors[concept_key].unsqueeze(0)
    embeddings = torch.cat([concept_embedding, random_embeddings], dim=0).numpy()
    labels = np.array([1] + [0] * len(random_texts))  # 1 for concept, 0 for random

    lr = LogisticRegression(random_state=42, max_iter=5000)
    lr.fit(embeddings, labels)
    cav = lr.coef_[0]
    return cav / np.linalg.norm(cav)  # Normalize the CAV

# Step 5: Generate random examples for CAV computation
neutral_sentences = [
    "The day started with a task.",
    "A meeting was held at the company.",
    "Work was assigned."
]

random_examples = [np.random.choice(neutral_sentences) for _ in range(500)]

male_cav = calculate_cav('male', concept_vectors, random_examples, clip_model, clip_processor)
female_cav = calculate_cav('female', concept_vectors, random_examples, clip_model, clip_processor)

# Step 6: Calculate TCAV Scores
def calculate_tcav_scores(texts, cavs, model, processor):
    tcav_scores = {'male': 0, 'female': 0}
    
    for text in tqdm(texts, desc="Calculating TCAV scores"):
        inputs = processor(text=[text], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = model.get_text_features(**inputs)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
        
        for concept, cav in cavs.items():
            score = np.dot(text_embedding.numpy(), cav)
            if score > 0:
                tcav_scores[concept] += 1
    
    total = sum(tcav_scores.values())
    for concept in tcav_scores:
        tcav_scores[concept] /= total if total > 0 else 1
    
    return tcav_scores

# Step 7: Calculate TCAV Scores by Occupation
print("Calculating TCAV scores by occupation...")
tcav_scores_by_occupation = {}
cavs = {'male': male_cav, 'female': female_cav}

occupations = label_encoder.classes_

for idx, occupation in enumerate(occupations):
    print(f"\nCalculating TCAV scores for occupation: {occupation} ({idx + 1}/{len(occupations)})")
    occupation_data = test_df[test_df['profession'] == occupation]
    if occupation_data.empty:
        print(f"No data found for occupation: {occupation}")
        continue

    test_texts = occupation_data['hard_text'].tolist()
    tcav_scores = calculate_tcav_scores(test_texts, cavs, clip_model, clip_processor)
    tcav_scores_by_occupation[occupation] = tcav_scores

# Step 8: Calculate TCAV Gender Gap and % Female for Each Occupation
results = []

for profession_name, tcav_score in tcav_scores_by_occupation.items():
    tcav_gender_gap = tcav_score['male'] - tcav_score['female']
    total = len(test_df[test_df['profession'] == profession_name])
    females = len(test_df[(test_df['profession'] == profession_name) & (test_df['gender'] == 1)])
    female_percentage = ((females / total) * 100) if total > 0 else 0

    results.append({
        'occupation': profession_name,
        'tcav_gender_gap': tcav_gender_gap,
        'female_percentage': female_percentage
    })

results_df = pd.DataFrame(results)

print("\nTCAV Gender Gap and Female Percentage by Occupation:")
print(results_df.to_string(index=False))
results_df.to_csv('tcav_results_text2concept.csv', index=False)
