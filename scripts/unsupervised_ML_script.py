# ---------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ---------------------------------------------
import pandas as pd
import os
import questionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------
# SET WORKING DIRECTORY TO SCRIPT LOCATION
# ---------------------------------------------
# Ensures that relative paths work regardless of where the script is run
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


# ---------------------------------------------
# LOAD MOVIE DATA
# ---------------------------------------------
# Read in the IMDb Top 1000 movies dataset
movies = pd.read_csv("../Data/imdb_top_1000.csv")


# ---------------------------------------------
# CLEANING AND PREPROCESSING
# ---------------------------------------------
# Convert year to integer and split the genre string into separate columns
movies['Year'] = movies['Year'].astype('int')
movies = movies.join(movies['Genre'].str.split(', ', expand=True)).drop(['Genre', 'Poster_Link', 'IMDB_Rating', 'Certificate', 'Overview', 'Meta_score', 'No_of_Votes'], axis = 1)

# Remove commas from Gross column and convert to float
movies['Gross'] = (movies['Gross'].replace(',', '', regex=True).astype(float))

# Rename columns for simplicity
movies = movies.set_axis(['Title', 'Year', 'Length','Director','Star1','Star2','Star3','Star4','Gross','Genre1','Genre2','Genre3'], axis = 1)

# ---------------------------------------------
# Build a balanced sublist of movies that are top-grossing and evenly distributed by genre
# ---------------------------------------------
# Drop rows with missing data
movies_filtered = movies.dropna(subset=['Gross', 'Genre1'])

# Sort by Gross (descending)
movies_sorted = movies.sort_values(by='Gross', ascending=False)

# Select a subset of movies from each major genre
sample_per_genre = 20  # Number of movies to sample per genre
unique_titles = set()

# Get most common genres
top_genres = movies_sorted['Genre1'].value_counts().head(10).index.tolist()
print(top_genres)

# Sample movies from each genre
for genre in top_genres:
    genre_subset = movies_sorted[movies_sorted['Genre1'] == genre]

    # Take top N grossing movies, then sample
    top_grossing = genre_subset.head(100)  # Look at top 50 for variety
    sampled = top_grossing.sample(
        n=min(sample_per_genre, len(top_grossing)),
        random_state=42
    )['Title'].tolist()

    unique_titles.update(sampled)

# Convert to sorted list
movie_titles = sorted(unique_titles)


# ---------------------------------------------
# SELECT USER FAVORITE MOVIES
# ---------------------------------------------
# Let the user choose their favorite movies from the top 100

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
tfidf_matrix = tfidf.fit_transform(movies['features'])

# ---------------------------------------------
# Compute cosine similarity matrix
# ---------------------------------------------
similarity_matrix = cosine_similarity(tfidf_matrix)


# ---------------------------------------------
# Find similar movies using similarity matrix
# ---------------------------------------------
# Get indices of liked movies
liked_indices = movies[movies['Title'].isin(selected_titles)].index

# Compute average similarity to all other movies
similarity_scores = similarity_matrix[liked_indices].mean(axis=0)

# Set scores of liked movies to -1 so they don't appear in results
similarity_scores[liked_indices] = -1


# ---------------------------------------------
# Show top N recommendations
# ---------------------------------------------
top_indices = similarity_scores.argsort()[::-1][:10]
recommendations = movies.iloc[top_indices][['Title', 'Year']]

print("\nRecommended movies:")
for i, row in recommendations.iterrows():
    print(f"{row['Title']} ({row['Year']})")
