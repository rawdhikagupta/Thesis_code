# README: TCAV Analysis Codebase

## Overview
This repository contains code for analyzing gender bias in a deep neural network (DNN) using **Testing with Concept Activation Vectors (TCAV)**. The provided scripts implement techniques to measure bias in various professional categories by utilizing trained models and a dataset of professional bios.

## Files and Descriptions

### 1. `plotting_results.py`
- Contains visualization utilities for analyzing TCAV results.
- Key features:
  - Processes and visualizes gender gaps across various professions.
  - Generates scatter plots and bar charts to showcase bias trends.

### 2. `tcav_dnn_preloaded.py`
- Implements the TCAV methodology on a pre-trained DNN.
- Key features:
  - Loads a pre-trained model (`dnn_model_2.h5`).
  - Tokenizes text data and predicts professional labels.
  - Calculates **Concept Activation Vectors (CAVs)** using male, female, and neutral concept examples.
  - Measures TCAV scores and gender gaps for various professions.

### 3. `tcav_dnn.py`
- Contains a complete pipeline for training a DNN and applying TCAV analysis.
- Key features:
  - Builds and trains a bi-directional GRU-based neural network with attention mechanisms.
  - Utilizes fastText embeddings for input representation.
  - Implements the TCAV methodology to assess bias in predictions.

### 4. `dnn_model_2.h5`
- Pre-trained DNN model used for predictions and TCAV analysis.
- Includes layers for embedding, bidirectional GRU, attention mechanism, and dense output.

## Dataset
- The code uses the **Bias in Bios** dataset, which includes professional bios labeled with professions and genders.
- Split into training, testing, and development sets in Parquet format.
- Files referenced in the scripts:
  - `data/train-00000-of-00001-0ab65b32c47407e8.parquet`
  - `data/test-00000-of-00001-5598c840ce8de1ee.parquet`
  - `data/dev-00000-of-00001-e6551072fff26949.parquet`

## Key Methodology

### 1. **Model Training and Prediction** (in `tcav_dnn.py`)
- Builds a GRU-based text classification model.
- Embedding layer initialized with fastText embeddings.
- Trains on professional bios to predict professions.

### 2. **Concept Generation** (in `tcav_dnn_preloaded.py` and `tcav_dnn.py`)
- Male, female, and neutral concept examples are generated using:
  - Concept-specific words (e.g., "he", "she").
  - Profession-relevant sentences for better contextual alignment.

### 3. **CAV Calculation**
- Uses logistic regression to derive CAVs that separate concept activations from random activations.

### 4. **TCAV Score Computation**
- Measures the sensitivity of model predictions to male and female concepts.
- Normalizes scores and calculates the gender gap for each profession.

### 5. **Visualization** (in `plotting_results.py`)
- Analyzes and plots:
  - Gender gap vs. TCAV scores.
  - Gender bias trends across professions.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries:
  - `tensorflow`, `keras`, `numpy`, `pandas`, `sklearn`, `matplotlib`, `tqdm`, `gensim`
- Install dependencies via pip:
  ```sh
  pip install -r requirements.txt
  ```

### Additional Requirements
- Pre-trained model (`dnn_model_2.h5`).
- Bias in Bios dataset files (Parquet format).

## Usage

### 1. Train the Model
- Run `tcav_dnn.py` to train a GRU-based model and save it as `dnn_model_2.h5`:
  ```sh
  python tcav_dnn.py
  ```

### 2. Perform TCAV Analysis
- Use the pre-trained model for TCAV analysis:
  ```sh
  python tcav_dnn_preloaded.py
  ```
- Outputs:
  - TCAV scores for male and female concepts.
  - Gender gap analysis by profession.
  - Results saved to `tcav_results_loaded_model.csv`.

### 3. Visualize Results
- Generate plots using `plotting_results.py`:
  ```sh
  python plotting_results.py
  ```

## Results
- The results showcase gender bias in the model predictions for different professions.
- Plots highlight the magnitude and direction of bias (e.g., male-leaning vs. female-leaning).

## Future Improvements
- Expand the concept examples for better representativeness.
- Test with additional datasets for generalizability.
- Use advanced interpretability techniques to complement TCAV analysis.

## License
This project is open-source and available under the [MIT License](LICENSE).


