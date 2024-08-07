import zipfile
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences




# Define paths
zip_file_path = 'archive.zip'
extract_dir = 'data/unfilteredData.csv'

# Create the directory if it doesn't exist
if not os.path.exists(extract_dir):
    print("going into this")
    os.makedirs(extract_dir)
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
# Unzip the file


print(f"Extracted all files to {extract_dir}")

data = pd.read_csv("data/unfilteredData.csv")

data['Lyrics'] = data['Lyrics'].apply(lambda x: x.lower())
data['Lyrics'] = data['Lyrics'].apply(lambda x: ''.join([c for c in x if c.isalpha() or c.isspace()]))
data = data[['Name', 'Artist', 'Lyrics']]
data.to_csv("data/filteredData.csv", index = False)

tokenizer = Tokenizer(num_words= 10000, oov_token= "<OOV>")
tokenizer.fit_on_texts(data['Lyrics'])
sequences = tokenizer.texts_to_sequences(data['Lyrics'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')

labels = pd.get_dummies(data['sentiment']).values

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)