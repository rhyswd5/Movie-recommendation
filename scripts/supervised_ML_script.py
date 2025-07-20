# ---------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ---------------------------------------------
import pandas as pd
import numpy as np
import questionary
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------
# SET WORKING DIRECTORY TO SCRIPT LOCATION
# ---------------------------------------------
# Ensures that relative paths work regardless of where the script is run
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# ---------------------------------------------
# LOAD MOVIE DATA
# ---------------------------------------------
# Read in the IMDb Top 1000 movies datasets
movies = pd.read_csv("../Data/imdb_top_1000.csv")


# ---------------------------------------------
# CLEANING AND PREPROCESSING
# ---------------------------------------------
# Convert year to integer and split the genre string into separate columns
movies['Year'] = movies['Year'].astype('int')
movies = movies.join(movies['Genre'].str.split(', ', expand=True)).drop(['Genre', 'Poster_Link', 'IMDB_Rating', 'Certificate', 'Overview', 'Meta_score', 'No_of_Votes', 'Gross'], axis = 1)

# Rename columns for simplicity
movies = movies.set_axis(['Title', 'Year', 'Length','Director','Star1','Star2','Star3','Star4','Genre1','Genre2','Genre3'], axis = 1)


# ---------------------------------------------
# SELECT USER FAVORITE MOVIES
# ---------------------------------------------
# Let the user choose their favorite movies from the top 100
movie_titles = np.sort(movies['Title'].head(100).dropna().unique().tolist())

selected_titles = questionary.checkbox(
    "Select your favorite movies (use space to select, enter to confirm):",
    choices=movie_titles
).ask()

# Show selections or warn if none selected
if selected_titles:
    print("\nYou selected the following movies:")
    for title in selected_titles:
        print(f"- {title}")
else:
    print("No movies selected.")

# Create a filtered DataFrame of liked movies
liked_movies = movies[movies['Title'].isin(selected_titles)]


# ---------------------------------------------
# LABEL MOVIES AS LIKED OR NOT LIKED
# ---------------------------------------------
# Label each movie as 1 (liked) if it's in the selected list, else 0
movies['Liked'] = movies['Title'].isin(liked_movies['Title']).astype(int)


# ---------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------
# Combine features (genres, director, lead actor) into a single string
# This text will be used to learn patterns in what makes a movie "likable"
movies['features'] = (
    movies['Genre1'].fillna('') + ' ' +
    movies['Genre2'].fillna('') + ' ' +
    movies['Director'].fillna('') + ' ' +
    movies['Star1'].fillna('') + ' ' +
    movies['Star2'].fillna('') + ' ' +
    movies['Star3'].fillna('') + ' ' +
    movies['Star4'].fillna('')
)


# ---------------------------------------------
# TF-IDF VECTORIZATION
# ---------------------------------------------
# Convert text features into a numeric matrix using TF-IDF
# TF-IDF scores help identify the most distinguishing features
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(movies['features'])


# ---------------------------------------------
# DEFINE TARGET VARIABLE
# ---------------------------------------------
# Our target is whether the user liked the movie (1) or not (0)
y = movies['Liked']


# ---------------------------------------------
# SPLIT DATA FOR TRAINING AND TESTING
# ---------------------------------------------
# Stratify to preserve like/dislike ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)


# ---------------------------------------------
# TRAIN SUPERVISED ML MODEL
# ---------------------------------------------
# Train a Random Forest classifier to predict "liking"
model = RandomForestClassifier()
model.fit(X_train, y_train)


# ---------------------------------------------
# PREDICT PROBABILITY OF LIKING EACH MOVIE
# ---------------------------------------------
# Use the model to predict the probability a user would like each movie
# Index 1 corresponds to probability of "Liked = 1"
movies['Predicted_Probability'] = model.predict_proba(X)[:, 1]


# ---------------------------------------------
# SHOW TOP 10 RECOMMENDED MOVIES
# ---------------------------------------------
# Only show movies the user didn't already select, sorted by highest predicted interest
recommendations = movies[movies['Liked'] == 0].sort_values(by='Predicted_Probability', ascending=False).head(10)
print(recommendations[['Title', 'Predicted_Probability']])
