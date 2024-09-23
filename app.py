from flask import Flask, request, jsonify 
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from bs4 import BeautifulSoup
import requests
import math

app = Flask(__name__)
GENIUS_API_TOKEN = '4wc7clE6jVqRtRR0JZfyWvsIGo9pVz72WiaRXTQcU707eWMt5nNXfkxx3Xh1IMMi0NxoVtseb-0_kTDSAGwfmw'
model = tf.keras.models.load_model('sentiment-analysis-model.h5')

data = pd.read_csv("data/filteredData.csv")
data['seq'] = data['seq'].fillna('').astype(str)
tokenizer = Tokenizer(num_words=30000, oov_token="<OOV>")
tokenizer.fit_on_texts(data['seq']) # Fit on training data


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Perform sentiment analysis (you can use any sentiment model/library)
    sequences = tokenizer.texts_to_sequences([text])  # Convert text to sequence of tokens
    padded_sequence = pad_sequences(sequences, maxlen=100, padding='post')  # Pad sequences

    # Get the sentiment prediction
    prediction = model.predict(padded_sequence)
    sentiment_score = float(prediction[0][0])  # Convert numpy array to float
    sentiment_score = math.ceil(sentiment_score * 10000) / 100
    return jsonify({'sentiment': sentiment_score})

if __name__ == '__main__':
    app.run(port=3000, debug=True)
