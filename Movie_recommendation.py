import inquirer as inq
import pandas as pd
import numpy as np

# Importing movie data
movies = pd.read_csv('../data/imdb_top_1000.csv')

# Cleaning data
movies['Released_Year'] = movies['Released_Year'].astype('int')
movies = movies.join(movies['Genre'].str.split(', ', expand=True)).drop(['Genre', 'Poster_Link', 'IMDB_Rating', 'Certificate', 'Overview', 'Meta_score', 'No_of_Votes', 'Gross'], axis = 1)
movies.set_axis(['Title', 'Year', 'Length','Director','Star1','Star2','Star3','Star4','Genre1','Genre2','Genre3'], axis = 1, inplace=True)

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
# then creates a dataframe of those 10 movies
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

# Determine whether each movie in movies dataframe was released close to any of the 
# movies in the liked_movies dataframe then assigning a score. 
def compare_years(y1, y2):
    if y1 > y2 - 5 and y1 < y2 + 5:
        return 4
    elif y1 > y2 - 10 and y1 < y2 + 10:
        return 2      

def years(y):
    return liked_movies.apply(lambda row: compare_years(y, row['Year']), axis = 1)

year_scores = movies.apply(lambda row: years(row['Year']), axis = 1).sum(axis = 1)
movies['Score'] = year_scores

# Determine if there is a match in the director between any any movie in the
# movies and liked_movies dataframe.
def compare_directors(d1, d2):
    if d1 == d2:
        return 10      

def directors(d):
    return liked_movies.apply(lambda row: compare_directors(d, row['Director']), axis = 1)

director_scores = movies.apply(lambda row: directors(row['Director']), axis = 1).sum(axis = 1)
movies['Score'] = movies['Score'].add(director_scores)

# Creates a score based on how many times the genre for each movie in the movie data frame
# appears in the liked_movies dataframe.
def compare_genres(g1, g2, g3, liked_g1, liked_g2, liked_g3):
    x = 0
    if g1 is not None and (g1 == liked_g1 or g1 == liked_g2 or g1 == liked_g3):
        x += 1
    if g2 is not None and (g2 == liked_g1 or g2 == liked_g2 or g2 == liked_g3):
        x += 1
    if g3 is not None and (g3 == liked_g1 or g3 == liked_g2 or g3 == liked_g3):
        x += 1
    return x

def genres(g1, g2, g3):
    return liked_movies.apply(lambda row: compare_genres(g1, g2, g3, row['Genre1'], row['Genre2'], row['Genre3']), axis = 1)

genre_scores = movies.apply(lambda row: genres(row['Genre1'], row['Genre2'], row['Genre3']), axis = 1).sum(axis = 1)
movies['Score'] = movies['Score'].add(genre_scores)

# Creates a score based on how many times the actors for each movie in the movie data frame
# appears in the liked_movies dataframe. 
def compare_actors(a1, a2, a3, a4, liked_a1, liked_a2, liked_a3, liked_a4):
    x = 0
    if a1 is not None and (a1 == liked_a1 or a1 == liked_a2 or a1 == liked_a3 or a1 == liked_a4):
        x += 5
    if a2 is not None and (a2 == liked_a1 or a2 == liked_a2 or a2 == liked_a3 or a2 == liked_a4):
        x += 5
    if a3 is not None and (a3 == liked_a1 or a3 == liked_a2 or a3 == liked_a3 or a3 == liked_a4):
        x += 5
    if a4 is not None and (a4 == liked_a1 or a4 == liked_a2 or a4 == liked_a3 or a4 == liked_a4):
        x += 5
    return x

def actors(a1, a2, a3, a4):
    return liked_movies.apply(lambda row: compare_actors(a1, a2, a3, a4, row['Star1'], row['Star2'], row['Star3'], row['Star4']), axis = 1)

actors_scores = movies.apply(lambda row: actors(row['Star1'], row['Star2'], row['Star3'], row['Star4']), axis = 1).sum(axis = 1)
movies['Score'] = movies['Score'].add(genre_scores)

# Gives the 10 movies with the highest final score
top_10_recomended_movies = movies.sort_values(by = 'Score', ascending=False).head(10)
print('Top 10 recomended movies: ', top_10_recomended_movies['Title'].tolist())
