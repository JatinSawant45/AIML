#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


# In[2]:


# Function to extract features from a single audio file
def extract_features(file_name):
    try:
        # Load the audio file
        audio, sample_rate = librosa.load(file_name, sr=22050)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma.T, axis=0)
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        spec_contrast_mean = np.mean(spec_contrast.T, axis=0)
        
        # Return the combined features
        return np.hstack([mfccs_mean, chroma_mean, spec_contrast_mean])
    
    except Exception as e:
        # Print detailed error message
        print(f"Error encountered while parsing file: {file_name}")
        print(f"Error: {e}")
        return None



# In[3]:


# Directory containing the dataset
dataset_dir = 'C:/Users/Lenovo/Downloads/genres_original/genres_original'

# Genres in the dataset
genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

# List to hold features and labels
features_list = []
labels_list = []

# Loop through each genre folder
for genre in genres:
    genre_dir = os.path.join(dataset_dir, genre)
    for file in os.listdir(genre_dir):
        file_path = os.path.join(genre_dir, file)
        # Print the file being processed for debugging
        print(f"Processing file: {file_path}")
        features = extract_features(file_path)
        if features is not None:
            features_list.append(features)
            labels_list.append(genre)




# In[4]:


# Convert to numpy arrays
X = np.array(features_list)
y = np.array(labels_list)

# Encode the genre labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


# In[5]:


from sklearn.ensemble import RandomForestClassifier

# Train a Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


# In[6]:


# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


# In[8]:


# Function to predict genre for a new audio file
def predict_genre(file_name, model, label_encoder):
    features = extract_features(file_name)
    if features is not None:
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        return label_encoder.inverse_transform(prediction)[0]
    else:
        return "Error in feature extraction"



# In[9]:


# Predict the genre of a new song
new_song_path = "C:/Users/Lenovo/Downloads/genres_original/genres_original/rock/rock.00000.wav"
predicted_genre = predict_genre(new_song_path, classifier, label_encoder)
print(f"The predicted genre is: {predicted_genre}")


# In[11]:


from sklearn.metrics import classification_report, accuracy_score
# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# In[ ]:




