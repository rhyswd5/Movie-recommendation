# ---------------------------------------------
# IMPORTING REQUIRED LIBRARIES
# ---------------------------------------------
import inquirer as inq
import pandas as pd
import os


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
movies = pd.read_csv('../data/imdb_top_1000.csv')


# ---------------------------------------------
# CLEANING AND PREPROCESSING
# ---------------------------------------------
# Convert year to integer and split the genre string into separate columns
movies['Year'] = movies['Year'].astype('int')
movies = movies.join(movies['Genre'].str.split(', ', expand=True)).drop(['Genre', 'Poster_Link', 'IMDB_Rating', 'Certificate', 'Overview', 'Meta_score', 'No_of_Votes', 'Gross'], axis = 1)

# Rename columns for ease
movies = movies.set_axis(['Title', 'Year', 'Length','Director','Star1','Star2','Star3','Star4','Genre1','Genre2','Genre3'], axis = 1)


# ---------------------------------------------
# SELECT USER FAVORITE MOVIES
# ---------------------------------------------
# Let the user choose their favorite movies from the top 100
# Creating a dataframe of top 100 rated movies which will be used to select the users liked movies
movies_top_100 = movies.iloc[0:200, :]
movies_top_100 = movies_top_100.sample(frac=1).reset_index(drop=True)
movie_names = movies_top_100['Title'].tolist()

choices1 = movie_names[0:10]
choices2 = movie_names[10:20]
choices3 = movie_names[20:30]
choices4 = movie_names[30:40]
choices5 = movie_names[40:50]
choices6 = movie_names[50:60]
choices7 = movie_names[60:70]
choices8 = movie_names[70:80]
choices9 = movie_names[80:90]
choices10 = movie_names[90:100]

# 10 questions that lets the user select their favourite movie out of 10 options
# then creates a dataframe of those movies
question = [
    inq.List('Movie 1',
            message="Choose which movie is your favourite among these movies",
            choices=choices1,
            ),
    inq.List('Movie 2',
            message="Choose which movie is your favourite among these movies",
            choices=choices2,
            ),
    inq.List('Movie 3',
            message="Choose which movie is your favourite among these movies",
            choices=choices3,
            ),
     inq.List('Movie 4',
             message="Choose which movie is your favourite among these movies",
             choices=choices4,
             ),
     inq.List('Movie 5',
             message="Choose which movie is your favourite among these movies",
             choices=choices5,
             ),
     inq.List('Movie 6',
             message="Choose which movie is your favourite among these movies",
             choices=choices6,
             ),
     inq.List('Movie 7',
             message="Choose which movie is your favourite among these movies",
             choices=choices7,
             ),
     inq.List('Movie 8',
             message="Choose which movie is your favourite among these movies",
             choices=choices8,
             ),
     inq.List('Movie 9',
             message="Choose which movie is your favourite among these movies",
             choices=choices9,
             ),
     inq.List('Movie 10',
             message="Choose which movie is your favourite among these movies",
             choices=choices10,
             ),
 ]
answers = list(inq.prompt(question).values())

# Creating a dataframe of all the chosen movies with there attributes and 
# removing the chosen movies from movie dataset so it doesnt turn up as a recomended movie
liked_movies = movies_top_100.loc[movies_top_100['Title'].isin(answers)]
movies.drop(movies[movies['Title'].isin(liked_movies['Title'])].index, inplace = True)


# ---------------------------------------------
# RULE-BASED SCORING FUNCTIONS
# ---------------------------------------------
# These functions award points based on how similar each movie is to those the user liked
def compare_years(y1, y2):
    if abs(y1 - y2) < 5:
        return 4
    elif abs(y1 - y2) < 10:
        return 2
    return 0

def compare_directors(d1, d2):
    return 6 if d1 == d2 else 0

def compare_genres(g1, g2, g3, liked_g1, liked_g2, liked_g3):
    return sum(g in [liked_g1, liked_g2, liked_g3] for g in [g1, g2, g3]) * 3

def compare_actors(a1, a2, a3, a4, liked_actors):
    return sum(a in liked_actors for a in [a1, a2, a3, a4]) * 2

# Summing the different scores for every movie

movies['Score'] = 0

for idx, row in movies.iterrows():
    year_score = liked_movies['Year'].apply(lambda y: compare_years(row['Year'], y)).sum()

    director_score = liked_movies['Director'].apply(lambda d: compare_directors(row['Director'], d)).sum()

    genre_score = liked_movies.apply(
        lambda r: compare_genres(row['Genre1'], row['Genre2'], row['Genre3'], r['Genre1'], r['Genre2'], r['Genre3']),
        axis=1
    ).sum()

    liked_actors = liked_movies[['Star1', 'Star2', 'Star3', 'Star4']].values.flatten()

    actor_score = compare_actors(row['Star1'], row['Star2'], row['Star3'], row['Star4'], liked_actors)

    total_score = year_score + director_score + genre_score + actor_score
    
    movies.at[idx, 'Score'] = total_score


# ---------------------------------------------
# FINAL RECOMMENDATIONS
# ---------------------------------------------
# Sort by score and show the top 10 movie recommendations
top_10_recomended_movies = movies.sort_values(by = 'Score', ascending=False).head(10)
print('Top 10 recomended movies: ', top_10_recomended_movies['Title'].tolist())
