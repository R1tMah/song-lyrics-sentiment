import zipfile
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
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
    print(f"Extracted all files to {extract_dir}")


   
   
   




data = pd.read_csv("data/unfilteredData.csv")


data['seq'] = data['seq'].apply(lambda x: x.lower())
data['seq'] = data['seq'].apply(lambda x: ''.join([c for c in x if c.isalpha() or c.isspace()]))
data.to_csv("data/filteredData.csv", index = False)


print("Starting the model part. ")


tokenizer = Tokenizer(num_words= 10000, oov_token= "<OOV>")
tokenizer.fit_on_texts(data['seq'])#lol
sequences = tokenizer.texts_to_sequences(data['seq'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post', padding='post')


labels = data['label'].values.astype(np.float32)


X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(30000, 32, input_length=100),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])


model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)


loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {accuracy * 100:.2f}%")





