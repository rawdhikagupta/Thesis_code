import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from sklearn.preprocessing import StandardScaler

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
max_words = 10000
max_len = 200
embedding_dim = 300

# Tokenize the text
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_df['hard_text'])

X_train = tokenizer.texts_to_sequences(train_df['hard_text'])
X_test = tokenizer.texts_to_sequences(test_df['hard_text'])

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

label_encoder = LabelEncoder()
train_df['profession_encoded'] = label_encoder.fit_transform(train_df['profession'])
test_df['profession_encoded'] = label_encoder.transform(test_df['profession'])

y_train = train_df['profession_encoded']
y_test = test_df['profession_encoded']

# Load pre-trained fastText embeddings
print("Loading fastText embeddings...")
fasttext_model = api.load("fasttext-wiki-news-subwords-300")

# Create embedding matrix
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in tokenizer.word_index.items():
    if i < max_words:
        try:
            embedding_vector = fasttext_model[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), embedding_dim)

# Step 3: Build and train the neural network model
gru_units = 100
attention_dim = 100

# Input layer
input_layer = Input(shape=(max_len,))

# Embedding layer
embedding_layer = Embedding(max_words, embedding_dim, 
                            weights=[embedding_matrix], 
                            input_length=max_len, 
                            trainable=True)(input_layer)

# Bidirectional GRU layer
bi_gru = Bidirectional(GRU(gru_units, return_sequences=True))(embedding_layer)

# Attention mechanism
attention = Dense(attention_dim, activation='tanh')(bi_gru)
attention = Dense(1, use_bias=False)(attention)
attention_weights = Lambda(lambda x: K.softmax(x, axis=1))(attention)
context_vector = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([bi_gru, attention_weights])

# Output layer
output_layer = Dense(len(label_encoder.classes_), activation='softmax')(context_vector)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
print("Training the model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
print("Evaluating the model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# # Plot training history
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

# Save the model
model.save('dnn_model_2.h5')
print("Model saved as 'dnn_model_2.h5'")

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
    "She is also an established model who has done photo shoots and appeared in magazines like Vogue, Teen Magazine, InStyle, Flaunt and more."
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
