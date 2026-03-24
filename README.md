Next Word Prediction Using Shakespeare’s Hamlet

A GRU-based text generation model trained on Shakespeare’s language.

📝 Table of Contents
Overview
Architecture
Model-Performance
Key Components
Usage Examples
Files
How to Use
Observations
Potential Improvements
Requirements
Future Work
📌 Overview

This project implements a Next Word Prediction Model using a GRU-based Recurrent Neural Network trained on text from Shakespeare's Hamlet.
The model predicts the most likely next word given a sequence of words.

🏗️ Architecture
Embedding Layer        → 100-dimensional vectors  
GRU Layer 1            → 150 units, return_sequences=True  
Dropout                → 0.2  
GRU Layer 2            → 100 units  
Dense Layer (Softmax)  → Output: 4818 vocabulary words
📊 Model Performance
Metric	Result
Training Accuracy	~67%
Validation Accuracy	~5–6%
Training Loss	6.98 → 1.47
Validation Loss	6.86 → 10.16

❗ Validation metrics show clear signs of overfitting.

🔧 Key Components
Data Preprocessing
Lowercasing text
Word-level tokenization
N-gram sequence generation (max length = 14)
Padding sequences
Train/test split = 80/20
Training Configuration
Optimizer: Adam
Loss: Categorical Crossentropy
Epochs: 50
Early Stopping: Patience = 50
💡 Usage Examples
# Example 1
Input: "To be or not to be"
Output: "the"

# Example 2
Input: "O farwel honest"
Output: "of"
📁 Files
File	Description
next_word_lstm.h5	Trained GRU model
tokenizer.pickle	Tokenizer with vocabulary mappings
hamlet.txt	Training corpus (Shakespeare’s Hamlet)
🚀 How to Use
1. Load Model
from tensorflow.keras.models import load_model
model = load_model('next_word_lstm.h5')
2. Load Tokenizer
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
3. Predict the Next Word
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Keep only last relevant tokens
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    # Pad sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

    # Predict
    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)

    # Map index → word
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            return word
    return None
⚠️ Observations
Strong signs of overfitting
Validation accuracy remains low
Vocabulary size: 4818 words
🔮 Potential Improvements
Train on additional Shakespeare plays
Apply learning-rate scheduling
Regularization: L2, recurrent dropout
Try LSTM / BiLSTM / GRU+Attention
Use pretrained embeddings: GloVe, Word2Vec
Adjust dropout / model size
📚 Requirements
tensorflow
numpy
pandas
nltk
scikit-learn
🎯 Future Work
Implement beam search
Add temperature-based sampling
Build a web interface (Flask / Streamlit)
Experiment with Transformer models (BERT, GPT-like)
