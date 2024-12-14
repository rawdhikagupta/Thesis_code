import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pickle
from keras import backend as K
from keras.models import Model 
import os
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

# Step 1: Load the saved model and necessary objects
print("Loading model and necessary objects...")
model = load_model('dnn_model_2.h5', custom_objects={'K': K})
model.summary()
splits = {
    'train': 'data/train-00000-of-00001-0ab65b32c47407e8.parquet',
    'test': 'data/test-00000-of-00001-5598c840ce8de1ee.parquet',
    'dev': 'data/dev-00000-of-00001-e6551072fff26949.parquet'
}

train_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/" + splits["train"])
max_words = 10000
max_len = 200
embedding_dim = 300

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['hard_text'])


# tokenizer_path = 'tokenizer.pickle'
# if not os.path.exists(tokenizer_path):
#     print(f"'{tokenizer_path}' not found. Creating new tokenizer...")
    
#     # Load the test data
#     test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/data/test-00000-of-00001-5598c840ce8de1ee.parquet")
    
#     # Create and fit the tokenizer
#     tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
#     tokenizer.fit_on_texts(test_df['hard_text'])
    
#     # Save the tokenizer
#     with open(tokenizer_path, 'wb') as handle:
#         pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print(f"New tokenizer created and saved to '{tokenizer_path}'")
# else:
#     print(f"Loading existing tokenizer from '{tokenizer_path}'")

# # Load the tokenizer
# with open(tokenizer_path, 'rb') as handle:
#     tokenizer = pickle.load(handle)

# Check if label_encoder file exists
label_encoder_path = 'label_encoder.pickle'
if not os.path.exists(label_encoder_path):
    print(f"'{label_encoder_path}' not found. Creating new label encoder...")
    
    # Load the test data
    test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/data/test-00000-of-00001-5598c840ce8de1ee.parquet")
    
    # Create and fit the label encoder
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    label_encoder.fit(test_df['profession'])
    
    # Save the label encoder
    with open(label_encoder_path, 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"New label encoder created and saved to '{label_encoder_path}'")
else:
    print(f"Loading existing label encoder from '{label_encoder_path}'")

# Load the label encoder
with open(label_encoder_path, 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the test data
test_df = pd.read_parquet("hf://datasets/LabHC/bias_in_bios/data/test-00000-of-00001-5598c840ce8de1ee.parquet")
# Constants (make sure these match your training script)
max_words = 10000
max_len = 200

sample_data = test_df.sample(n=100)  # Adjust the sample size as needed
sample_texts = sample_data['hard_text'].tolist()
true_labels = sample_data['profession'].tolist()

#To check whether the model is loaded properly or not. 
# Preprocess the sample texts
input_sequences = tokenizer.texts_to_sequences(sample_texts)
input_sequences = pad_sequences(input_sequences, maxlen=max_len)

# Make predictions
predictions = model.predict(input_sequences)
predicted_classes = np.argmax(predictions, axis=1)

# Convert predicted classes back to profession labels
predicted_labels = label_encoder.inverse_transform(predicted_classes)

# Calculate accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"Model accuracy on sample data: {accuracy:.2f}")

# Step 3: Generate concept examples
# def generate_concept_examples(concept_words, num_examples=1000):
#     examples = []
#     for _ in range(num_examples):
#         example = np.random.choice(concept_words, size=max_len, replace=True)
#         examples.append(' '.join(example))
#     return examples
# def generate_random_examples(concept_words, num_examples=1000):
#     examples = []
#     for _ in range(num_examples):
#         # Create more natural sentences instead of random word combinations
#         # Include some profession-neutral words and structure
#         neutral_words = ['the', 'is', 'a', 'professional', 'person', 'who', 'works', 'as']
#         sentence_parts = (
#             np.random.choice(neutral_words, size=2).tolist() +
#             [np.random.choice(concept_words)] +
#             np.random.choice(neutral_words, size=2).tolist()
#         )
#         examples.append(' '.join(sentence_parts))
#     return examples

# Expand gender word lists to include more relevant terms
# male_words = ["he", "man", "male", "his", "him", "himself","mr"]
# female_words = ["she", "woman", "female", "her", "hers", "herself", "mrs", "ms"]
# random_words = [
#     'this', 'is', 'a', 'random', 'sentence', 'for', 'testing','occupation'
#     # 'person', 'who', 'works', 'them', 'their', 'professional', 
#     # 'them','their', 'those'
#     # 'expert', 'specialist','leader',
#     # 'practitioner', 'graduate', 'certified', 'licensed', 'experienced', 'qualified',
#     # 'consultant', 'instructor', 'dedicated', 'passionate', 
#     # 'committed', 'innovative', 'creative', 'analytical', 'strategic', 'detail-oriented', 
#     # 'collaborative', 'efficient', 'skilled', 'knowledgeable', 'accomplished'
# ]

# male_examples = generate_concept_examples(male_words)
# female_examples = generate_concept_examples(female_words)
# random_examples = generate_random_examples(random_words)

# Define male, female, and neutral concepts using professional context
male_words = ["he", "man", "his", "him", "Mr.", "male"]
female_words = ["she", "woman", "her", "Ms.", "female"]
neutral_words = ["day", "task", "object", "company", "meeting", "work"]

# Define sentences to capture professional context for male, female, and random concepts
male_sentences = [
    "John is a dedicated software engineer known for his innovative solutions and leadership in project management.",
    "He has led multiple successful teams, driving efficiency and productivity.",
    "His technical expertise in programming languages has earned him recognition in the tech community."
]

female_sentences = [
    "Emily is a talented software engineer who excels in collaborative projects and user experience design.",
    "She is known for her creativity and attention to detail, often contributing to team brainstorming sessions.",
    "Emily's work has significantly improved product usability, and she actively mentors junior developers."
    "She has travelled a lot to attend fashion shows around the world"
    "Jane is a renowned fashion model known for her versatility and elegance, having graced the covers of major fashion magazines and walked the runways for top designers around the world."
]

neutral_sentences = [
    "The day started with a task.",
    "A meeting was held at the company.",
    "Work was assigned."
]

# Step 5: Generate concept examples
def generate_concept_examples(concept_words, concept_sentences, num_examples=1000):
    examples = []
    for _ in range(num_examples // 2):  # Half from phrases, half random
        # Add a random sentence or phrase
        examples.append(np.random.choice(concept_sentences))
        # Randomly combine words to generate synthetic examples
        example = np.random.choice(concept_words, size=max_len, replace=True)
        examples.append(' '.join(example))
    return examples

male_examples = generate_concept_examples(male_words, male_sentences)
female_examples = generate_concept_examples(female_words, female_sentences)
random_examples = generate_concept_examples(neutral_words, neutral_sentences)

# Step 4: Calculate CAVs
def calculate_cav(concept_examples, random_examples, layer_name='dense_1'):
    layer_output = model.get_layer(layer_name).output
    activation_model = Model(inputs=model.input, outputs=layer_output)

    concept_activations = activation_model.predict(
        pad_sequences(tokenizer.texts_to_sequences(concept_examples), maxlen=max_len))
    random_activations = activation_model.predict(
        pad_sequences(tokenizer.texts_to_sequences(random_examples), maxlen=max_len))
    
    # Reshape the activations to 2D
    concept_activations_2d = concept_activations.reshape(concept_activations.shape[0], -1)
    random_activations_2d = random_activations.reshape(random_activations.shape[0], -1)
    
    X = np.vstack([concept_activations_2d, random_activations_2d])
    y = np.array([1] * len(concept_activations) + [0] * len(random_activations))
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use LinearSVC with increased max_iter and added tol
    # svm = LinearSVC(random_state=42, max_iter=5000, tol=1e-4, dual=False)
    # svm.fit(X_scaled, y)
    # return svm.coef_[0]
    lr = LogisticRegression(random_state=42, max_iter=5000, tol=1e-4)
    lr.fit(X_scaled, y)
    cav = lr.coef_[0]  # Returns the coefficients which represent the CAV
    
    # Increase the contrast between concept and random examples
    # Maybe use more distinctive examples for male/female concepts
    
    # Add L2 normalization to the CAV
    cav = cav / np.linalg.norm(cav)
    
    # Consider using a different classifier (e.g., SVM with larger margin)
    # instead of logistic regression
    return cav

print("Calculating CAVs...")
male_cav = calculate_cav(male_examples, random_examples)
female_cav = calculate_cav(female_examples, random_examples)

# Step 5: Calculate TCAV Scores
def calculate_tcav_scores(texts, cavs, layer_name='dense_1'):
    layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    
    tcav_scores = {'male': 0, 'female': 0}
    batch_size = 32
    
    with tqdm(total=len(texts), desc="Calculating TCAV scores") as pbar:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:min(i + batch_size, len(texts))]
            
            input_sequences = tokenizer.texts_to_sequences(batch)
            input_sequences = pad_sequences(input_sequences, maxlen=max_len)
            
            activations = layer_model.predict(input_sequences, verbose=0)
            
            # Add padding to activations before calculating gradient
            padded_activations = np.pad(activations, ((1, 1), (0, 0), (0, 0)), mode='edge')            
            grads = np.gradient(padded_activations, axis=0)[1:-1]  # Remove padding from result
            grads = grads.reshape( -1,200)
            # grads = grads.reshape(-1, grads.shape[1])  
            # grads = grads.reshape(-1, 200)  # Reshape to (batch_size * 200, 1)
            
            # Add normalization:
            grads = grads / (np.linalg.norm(grads, axis=-1, keepdims=True) + 1e-10)
            
            for i, (concept, cav) in enumerate(zip(['male', 'female'], cavs)):
                dot_product = np.dot(grads, cav)
                tcav_scores[concept] += np.sum(dot_product > 0)
            
            pbar.update(len(batch))
    
    # Normalize scores
    total = sum(tcav_scores.values())
    for concept in tcav_scores:
        tcav_scores[concept] /= total if total != 0 else 1
    
    return tcav_scores

# Step 6: Calculate TCAV Scores by Occupation
print("Calculating TCAV scores by occupation...")
tcav_scores_by_occupation = {}
occupations = label_encoder.classes_
total_occupations = len(occupations)

for idx, occupation in enumerate(occupations):
    profession_name = occupation
    print(f"\nCalculating TCAV scores for occupation: {profession_name}")
    print(f"Progress: {idx+1}/{total_occupations} occupations")
    
    occupation_data = test_df[test_df['profession'] == occupation]
    if occupation_data.empty:
        print(f"Warning: No data found for occupation '{occupation}'")
        continue
    
    test_texts = occupation_data['hard_text'].tolist()
    # Pass both CAVs at once
    tcav_scores = calculate_tcav_scores(test_texts, [male_cav, female_cav])
    tcav_scores_by_occupation[occupation] = tcav_scores
    
    print(f"TCAV scores for {occupation}: Male: {tcav_scores['male']:.4f}, Female: {tcav_scores['female']:.4f}")

# Step 7: Calculate TCAV Gender Gap and % Female for Each Occupation
results = []

for profession_name, tcav_score in tcav_scores_by_occupation.items():
    # Calculate TCAV Gender Gap
    tcav_gender_gap = tcav_score['male'] - tcav_score['female']
    
    # Calculate % Female
    total = len(test_df[test_df['profession'] == profession_name])
    females = len(test_df[(test_df['profession'] == profession_name) & (test_df['gender'] == 1)])
    female_percentage = ((females / total) * 100) if total > 0 else 0

    results.append({
        'occupation': profession_name,
        'tcav_gender_gap': tcav_gender_gap,
        'female_percentage': female_percentage
    })

results_df = pd.DataFrame(results)

# Print the results
print("\nTCAV Gender Gap and Female Percentage by Occupation:")
print(results_df.to_string(index=False))

# Optionally, save the results to a CSV file
results_df.to_csv('tcav_results_loaded_model.csv', index=False)
print("\nResults saved to 'tcav_results_loaded_model.csv'")
