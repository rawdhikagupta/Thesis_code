import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Lambda
from keras.models import Model
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gensim.downloader as api
from keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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

# Step 1: Load the saved model
print("Loading model...")
model = load_model('dnn_model.h5', custom_objects={'K': K})

# Step 2: Preprocess the test data
print("Preprocessing test data...")
X_test_sequences = tokenizer.texts_to_sequences(test_df['hard_text'])
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_len)

# Step 3: Make predictions
print("Making predictions on test data...")
predictions = model.predict(X_test_padded)
predicted_classes = np.argmax(predictions, axis=1)

# Step 4: Calculate accuracy
true_labels = test_df['profession_encoded'].values  # Use the encoded labels
accuracy = accuracy_score(true_labels, predicted_classes)
print(f"Model accuracy on test data: {accuracy:.2f}")