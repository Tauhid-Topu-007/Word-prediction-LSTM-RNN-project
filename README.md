Next Word Prediction Using Shakespeare's Hamlet
📌 Overview
This project implements a next word prediction model using a GRU-based Recurrent Neural Network trained on Shakespeare's Hamlet. The model learns patterns in the text to predict the most likely next word given an input sequence.

🏗️ Architecture
Embedding Layer: Converts words to dense vectors (100-dimensional)

GRU Layers: Two stacked GRU layers for sequence learning

First GRU: 150 units with return_sequences=True

Second GRU: 100 units

Dropout: 0.2 dropout rate for regularization

Dense Layer: Softmax activation over vocabulary size (4818 words)

📊 Model Performance
Training Accuracy: ~67% after 50 epochs

Validation Accuracy: ~5-6% (indicating overfitting)

Training Loss: Decreased from 6.98 to 1.47

Validation Loss: Increased from 6.86 to 10.16

🔧 Key Components
Data Preprocessing
Text converted to lowercase

Tokenization of words with word indices

Creation of n-gram sequences (max sequence length: 14)

Padding sequences to uniform length

Train-test split: 80/20

Training Configuration
Optimizer: Adam

Loss Function: Categorical Crossentropy

Epochs: 50

Early Stopping: Patience of 50 epochs

💡 Usage Examples
python
# Example 1
Input: "To be or not to be"
Output: "the"

# Example 2
Input: "O farwel honest"
Output: "of"
📁 Files
next_word_lstm.h5 - Saved trained model

tokenizer.pickle - Saved tokenizer with word index mapping

hamlet.txt - Raw Shakespeare's Hamlet text

🚀 How to Use
Load the model:

python
from tensorflow.keras.models import load_model
model = load_model('next_word_lstm.h5')
Load the tokenizer:

python
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
Predict next word:

python
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None
⚠️ Observations
The model shows signs of overfitting (increasing validation loss despite decreasing training loss)

Validation accuracy remains low (~5-6%), suggesting the model may be memorizing patterns rather than generalizing

The vocabulary contains 4818 unique words from the play

🔮 Potential Improvements
Add more training data from other Shakespeare plays

Implement learning rate scheduling

Add more regularization techniques

Experiment with different architectures (LSTM, Bidirectional LSTM)

Use pre-trained word embeddings (GloVe, Word2Vec)

Increase model capacity or adjust dropout rates

📚 Requirements
text
tensorflow
numpy
pandas
nltk
scikit-learn
🎯 Future Work
Implement beam search for better predictions

Add temperature parameter for controlled randomness

Create a web interface for interactive predictions

Experiment with transformer-based architectures

