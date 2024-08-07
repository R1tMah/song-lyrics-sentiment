import zipfile
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)
# Define paths
zip_file_path = 'archive.zip'
extract_dir = 'data'

# Create the directory if it doesn't exist
if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Extracted all files to {extract_dir}")

data = pd.read_csv("data/songs.csv")

data['lyrics'] = data['lyrics'].apply(lambda x: x.lower())
data['lyrics'] = data['lyrics'].apply(lambda x: ''.join([c for c in x if c.isalpha() or c.isspace()]))