import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import pickle
import os
import random
from tqdm import tqdm

# Step 1: Load the Bias in Bios dataset
splits = {
    'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
    'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet',
    'dev': 'data/dev-00000-of-00001-e6551072fff26949.parquet'
}

# Load data using pandas (assuming dataset is in your directory or Huggingface)
train_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["test"])
dev_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["dev"])

# After loading the data, add this code to display all unique occupations
print("All unique occupations:")
print(train_df['profession'].unique())

# Step 2: Define a simple LSTM model for embedding extraction
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, num_layers=1):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)  # Final output layer to get a fixed size vector

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = lstm_out[:, -1, :]  # Get the last hidden state
        return self.fc(out)

# Step 3: Initialize LSTM model, tokenizer, and optimizer
# Simple tokenizer assuming pre-tokenized input
vocab = {word: i for i, word in enumerate(set(" ".join(train_df['hard_text']).split()))}
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128
lstm_model = LSTMModel(vocab_size, embedding_dim, hidden_dim)
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Step 4: Define Gender Concepts (male and female phrases)
male_phrases = ["he", "man", "male", "his", "him"]
female_phrases = ["she", "woman", "female", "her", "hers"]

# Step 5: Tokenizer function
def simple_tokenizer(phrases):
    return [[vocab.get(word, 0) for word in phrase.split()] for phrase in phrases]

# Step 6: Method to generate activations using LSTM
def get_activations(phrases):
    tokenized = simple_tokenizer(phrases)
    inputs = [torch.tensor(sentence, dtype=torch.long).unsqueeze(0) for sentence in tokenized]
    activations = []
    with torch.no_grad():
        for inp in inputs:
            out = lstm_model(inp)
            activations.append(out.squeeze().numpy())
    return np.array(activations)

# Step 7: Get activations for male, female, and random control phrases
random_phrases = ["random", "words", "not", "related", "concepts", "Laptop", "Coffee", "Physics", "Sculpture", "Architecture"]
male_activations = get_activations(male_phrases)
female_activations = get_activations(female_phrases)
random_activations = get_activations(random_phrases)

# Step 8: Train Logistic Regression to create CAVs
X = np.vstack([male_activations, female_activations, random_activations])
y = np.array([1]*len(male_activations) + [2]*len(female_activations) + [0]*len(random_activations))

cav_model = LogisticRegression(multi_class='ovr', solver='lbfgs').fit(X, y)
cavs = cav_model.coef_

# Save the CAVs for future use
os.makedirs('cavs', exist_ok=True)
with open('cavs/cav_gender.pkl', 'wb') as f:
    pickle.dump(cavs, f)

# Step 10: Calculate TCAV Scores
def calculate_tcav_scores(model, sentences, cavs):
    tcav_scores = {'male': 0, 'female': 0}
    total_sentences = len(sentences)

    with tqdm(total=total_sentences, desc="Calculating TCAV scores", ncols=100) as pbar:
        inputs = simple_tokenizer(sentences)
        activations = []
        for inp in inputs:
            out = lstm_model(torch.tensor(inp, dtype=torch.long).unsqueeze(0))
            activations.append(out.squeeze().detach().numpy())
        activations = np.array(activations)

        grads = np.gradient(activations, axis=0)  # Calculate gradients
        for i, (concept, cav) in enumerate(zip(['male', 'female'], cavs)):
            dot_product = np.dot(grads, cav)
            tcav_scores[concept] += np.sum(dot_product > 0)

        pbar.update(total_sentences)

    total = sum(tcav_scores.values())
    for concept in tcav_scores:
        tcav_scores[concept] /= total if total != 0 else 1  # Normalize to percentages

    return tcav_scores

# Step 11: Calculate TCAV Scores by Occupation
tcav_scores_by_occupation = {}
occupations = train_df['profession'].unique()
total_occupations = len(occupations)

for idx, occupation in enumerate(occupations, 1):
    print(f"\nCalculating TCAV scores for occupation: {occupation}")
    print(f"Progress: {idx}/{total_occupations} occupations")
    
    occupation_data = train_df[train_df['profession'] == occupation]
    if occupation_data.empty:
        print(f"Warning: No data found for occupation '{occupation}'")
        continue
    
    test_sentences = occupation_data['hard_text'].tolist()
    tcav_scores = calculate_tcav_scores(lstm_model, test_sentences, cavs)
    tcav_scores_by_occupation[occupation] = tcav_scores
    
    print(f"TCAV scores for {occupation}: Male: {tcav_scores['male']:.4f}, Female: {tcav_scores['female']:.4f}")

# Step 12: Calculate Gender Gap and % Female for Each Occupation
gender_gaps = []
female_proportions = []
occupation_names = []

for occupation, tcav_score in tcav_scores_by_occupation.items():
    # Multiply TCAV scores by 100 to convert to percentages
    gender_gap = (tcav_score['male'] * 100) - (tcav_score['female'] * 100)
    gender_gaps.append(gender_gap)

    total = len(train_df[train_df['profession'] == occupation])
    females = len(train_df[(train_df['profession'] == occupation) & (train_df['gender'] == 'female')])
    female_proportion = females / total if total > 0 else 0
    female_proportions.append(female_proportion * 100)  # Convert to percentage
    occupation_names.append(occupation)

# Plotting Gender Gap vs % Female
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
plt.scatter(female_proportions, gender_gaps, marker='o')

for i, occupation in enumerate(occupation_names):
    plt.text(female_proportions[i], gender_gaps[i], occupation, fontsize=8, alpha=0.7)

plt.xlabel('% Female')
plt.ylabel('TCAV Gender Gap (Male% - Female%)')
plt.title('Gender Gap per Occupation vs. % Females')
plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
plt.axvline(x=50, color='r', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
