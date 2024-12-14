import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# Loading the dataset using pandas
splits = {
    'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
    'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet',
    'dev': 'data/dev-00000-of-00001-e6551072fff26949.parquet'
}

# Loading data using pandas
train_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["test"])
dev_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["dev"])

# Step 1: Initializing the tokenizer and BART model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Step 2: Defining Gender Concepts (male and female phrases)
male_phrases = ["he", "man", "male", "his", "him"]
female_phrases = ["she", "woman", "female", "her", "hers"]

# Step 3: Method to generate Concept Activations
def get_activations(phrases):
    activations = []
    for phrase in phrases:
        inputs = tokenizer(phrase, return_tensors='pt', max_length=512, truncation=True)
        outputs = model.model.encoder(**inputs)
        activations.append(outputs.last_hidden_state.mean(dim=1).detach().numpy())
    return np.array(activations).squeeze()

# Get activations for male, female, and random (control) phrases
male_activations = get_activations(male_phrases)
female_activations = get_activations(female_phrases)
random_phrases = ["random", "words", "not", "related", "concepts","Laptop", "Coffee","Physics", "Sculpture", "Architecture"]
random_phrases2 = ["Mountain", "Bicycle", "Laptop", "Coffee", "River", "Bookstore", "Elephant", "Telescope", "Architecture",
"Astronomy", "Guitar", "Puzzle", "Cinema", "Clouds", "Algorithm", "Backpack", "Photography", "Chemistry",
"Volcano", "Gardening", "Piano", "Football", "Sunshine", "Galaxy", "Sandbox", "Snowflake", "Castle",
"Chocolate", "Subway", "Microwave", "Ocean", "Jungle", "Orchestra", "Newspaper", "Physics", "Sculpture",
"Satellite", "Windmill", "Dinosaur", "Calendar", "Meditation", "Symphony", "Pirate", "Postcard", "Forest",
"Typewriter", "Robot", "Calendar", "Marshmallow", "Origami"]
random_activations = get_activations(random_phrases)

# Step 4: Train Logistic Regression to Create CAVs
X = np.vstack([male_activations, female_activations, random_activations])
y = np.array([1]*len(male_activations) + [2]*len(female_activations) + [0]*len(random_activations))

# Train Logistic Regression model to create the CAVs
cav_model = LogisticRegression(multi_class='ovr', solver='lbfgs').fit(X, y)
cavs = cav_model.coef_

# Save the CAVs for future use
os.makedirs('cavs', exist_ok=True)
with open('cavs/cav_gender.pkl', 'wb') as f:
    pickle.dump(cavs, f)
import random
from tqdm import tqdm

# Helper function to shuffle and chunk the data into mini-batches
def batch_data_randomized(sentences, batch_size):
    # Shuffle sentences before creating batches
    random.shuffle(sentences)
    for i in range(0, len(sentences), batch_size):
        yield sentences[i:i + batch_size]

# Step 5: Calculate TCAV Scores with Randomized Mini-Batches
def calculate_tcav_scores_with_random_batches(model, tokenizer, sentences, cavs, batch_size=32):
    tcav_scores = {'male': 0, 'female': 0}
    total_sentences = len(sentences)

    # Initialize tqdm progress bar for sentence processing
    with tqdm(total=total_sentences, desc="Calculating TCAV scores with randomized batches", ncols=100) as pbar:
        for batch in batch_data_randomized(sentences, batch_size):
            # Tokenize the batch
            inputs = tokenizer(batch, return_tensors='pt', padding=True, max_length=512, truncation=True)
            outputs = model.model.encoder(**inputs)
            activations = outputs.last_hidden_state.mean(dim=1).detach().numpy()

            grads = np.gradient(activations, axis=0)  # Calculate gradients for the batch
            for i, (concept, cav) in enumerate(zip(['male', 'female'], cavs)):
                dot_product = np.dot(grads, cav)
                tcav_scores[concept] += np.sum(dot_product > 0)

            # Update the progress bar after processing each mini-batch
            pbar.update(len(batch))

    total = sum(tcav_scores.values())
    for concept in tcav_scores:
        tcav_scores[concept] /= total if total != 0 else 1  # Normalize to percentages

    return tcav_scores

# Step 6: Calculate TCAV Scores by Occupation with Randomized Mini-Batches
tcav_scores_by_occupation = {}

# List of unique occupations
occupations = train_df['profession'].unique()

for occupation in occupations:
    # Extract biographies for the given occupation
    occupation_data = train_df[train_df['profession'] == occupation]
    test_sentences = occupation_data['hard_text'].tolist()

    # Calculate TCAV scores for male and female concepts in randomized mini-batches
    tcav_scores = calculate_tcav_scores_with_random_batches(model, tokenizer, test_sentences, cavs, batch_size=32)

    # Store the scores for this occupation
    tcav_scores_by_occupation[occupation] = tcav_scores

# Step 7: Calculate Gender Gap and % Female for Each Occupation
gender_gaps = []
female_proportions = []
occupation_names = []

for occupation, tcav_score in tcav_scores_by_occupation.items():
    # Calculate gender gap (difference between male and female TCAV scores)
    gender_gap = tcav_score['male'] - tcav_score['female']
    gender_gaps.append(gender_gap)

    # Calculate the proportion of females in this occupation
    total = len(train_df[train_df['profession'] == occupation])
    females = len(train_df[(train_df['profession'] == occupation) & (train_df['gender'] == 'female')])
    female_proportion = females / total if total > 0 else 0
    female_proportions.append(female_proportion)

    # Store the occupation name for annotation
    occupation_names.append(occupation)

# Step 8: Plot Gender Gap vs % Female
plt.figure(figsize=(10, 6))

# Create scatter plot
plt.scatter(female_proportions, gender_gaps, marker='o')

# Annotate each point with the occupation name
for i, occupation in enumerate(occupation_names):
    plt.text(female_proportions[i], gender_gaps[i], occupation, fontsize=9)

# Add labels and title
plt.xlabel('% Female')
plt.ylabel('TCAV Gender Gap (Male - Female)')
plt.title('Gender Gap per Occupation vs. % Females')

# Show the plot
plt.show()
# Since we already have profession labels, we can skip the BART model training for classification
# and directly calculate TCAV scores using the given labels and biographies
