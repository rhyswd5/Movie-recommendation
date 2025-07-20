# <b>Movie Recommendation System</b>
## Project Summary

<p>This project explores three approaches to recommending movies to users based on their favourite selections. Starting with a simple rule-based system, it progresses to a supervised machine learning model and finally an unsupervised learning approach using TF-IDF and cosine similarity.</p>
<p>Each iteration aimed to improve the quality of recommendations, the scalability of the system, and the user experience when selecting their favourite films.</p>

### Why This Project? 
The goal of this project was to explore different approaches to content-based movie recommendations and understand how progressively more advanced methods—rule-based, supervised learning, and unsupervised learning—affect recommendation quality, scalability, and personalisation. Each version reflects a stage in developing a more robust, user-aware recommender system.

## Project Evolution & Rationale
### <b>1. Rule-Based Recommender</b>

#### Approach
A selection of 100 movies is presented to the user in 10 sets of 10. The user is prompted to select their favourite movie from each set, resulting in a total of 10 user-preferred films.

Each remaining movie is assigned a total score based on how closely it matches the user’s favourites across several attributes:

- Year Similarity:
Movies released within 5 years of a liked film receive 4 points, while those within 10 years receive 2.

- Director Match:
A direct match in director earns 6 points, emphasising the impact of directorial style.

- Genre Overlap:
Each genre match between a recommended and a liked movie earns 3 points, rewarding shared themes or moods.

- Actor Overlap:
For each shared actor (among the top 4), 2 points are awarded. This accounts for actor familiarity or ensemble preferences.

After scoring, the dataset is sorted by total score in descending order and the top 10 movies are returned.

#### Motivation

The rule-based model served as the initial prototype for this project. It was designed to provide a simple, interpretable way to generate movie recommendations without relying on complex machine learning techniques.

- Building the logic manually helped deepen my understanding of how content-based filtering works, including the importance of features like genres, directors, and actors.

- This approach required no model training, making it ideal for early testing, debugging, and user interaction design.

- Every recommendation could be traced back to explicit rules (e.g. shared actors or similar release years), making the system transparent and explainable.

- It provided a solid baseline to evaluate the effectiveness of more advanced approaches like supervised and unsupervised learning.

This version laid the groundwork for experimenting with user preference modelling, evaluation metrics, and automated learning in later stages of the project.

#### Limitations
The rule-based recommender was a valuable baseline, but it has several key limitations:
- The model relies on fixed rules and does not learn from user behaviour or improve over time.

- The point values assigned to shared genres, actors, and directors are manually defined and may not reflect true user preferences or relative importance.

- The system cannot capture complex relationships (e.g. a user liking certain actors only in specific genres).

- Adding new features (e.g. plot summaries, user ratings) requires manual rule updates, making the model harder to scale.

- There’s no built-in way to objectively assess the accuracy or quality of recommendations beyond manual inspection.

- Recommendations are generated from a fixed subset of films, limiting diversity and exposure to lesser-known options.

<br/>

### <b>2. Supervised Machine Learning (TF-IDF + Random Forest Classifier)</b>
#### Approach

Users choose favourite movies from a dropdown list. A label is applied to all movies based on user selection (1 if the movie is liked, 0 if otherwise). Features (e.g. genres, directors, and cast) are combined into text and vectorised using TF-IDF. The target variable (Liked) is then used to train a Random Forest Classifier. The data is split into training and test sets using train_test_split with stratification to preserve label balance. After training, the model predicts a probability score for each movie, representing the probabilty that the user would like it.

#### Key Improvements
The transition from a rule-based recommender to a supervised machine learning model introduced several key improvements:

- The Random Forest classifier can detect complex, non-linear relationships between genres, directors, and cast members — something rule logic struggles with.
- Rather than applying static rules for everyone, the model adapts to the individual user's choices and generalises their preferences to unseen films.
- The supervised model enables use of standard evaluation metrics (e.g. accuracy, precision, recall, ROC-AUC) to measure performance objectively.
- Year was excluded from the supervised model as:
    - It showed weak correlation with user preferences compared to content-based features like genres, directors, and actors.
    - Including “Year” introduced potential bias toward newer films without capturing actual taste.
    - As a numerical variable, it was incompatible with the TF-IDF-based text vectorisation used for other features.
    - Omitting it helped maintain a cleaner, more interpretable model focused on the movie’s content.
- Added support for probabilistic ranking (predicting "likelihood of liking").
- Reduced the need for repetitive, hard-coded comparisons (e.g. checking actor/director overlaps manually) by using vectorised representations.

#### Limitations
While the supervised model improves personalisation and prediction quality, it has several limitations:
- Movies are labelled as either "liked" or "not liked" based on user selection, which may oversimplify user preferences and ignore neutral or unknown opinions.

- Depending on the number of movies the users selects, the model could be trained on a very small set of like examples, which may not be sufficient to learn highly accurate patterns, especially for users with diverse tastes.

- The current feature set (genre, director, top actors) excludes richer data like plot summaries, reviews, or audience ratings, which could improve recommendation quality.

- Numerical fields such as movie year or revenue are not included in the model due to the use of TF-IDF, which may miss temporal or popularity trends.

- While the model can be evaluated using standard classification metrics, it lacks real-world validation through user engagement or satisfaction scoring.

<br/>

### <b>3. Unsupervised Learning (TF-IDF + Cosine Similarity)</b>
#### Approach

Users select favourite movies from a carefully balanced and recognisable sample of top-grossing films, spread across genres. Features (e.g. genres, directors, and cast) are combined into text and vectorised using TF-IDF.  Pairwise cosine similarity is computed between all movies in the dataset. The average similarity between the user’s favourites and all other movies determines the recommendation ranking. The top N most similar movies (excluding the ones the user selected) are returned as personalised suggestions.


#### Key Improvements

The unsupervised model builds on the strengths of both rule-based and supervised approaches, while addressing some of their limitations:

- Unlike the supervised model, it works immediately after user input — ideal for rapid deployment or smaller datasets.

- Eliminates the need for hard-coded logic, reducing bias and making the system more flexible.

- The user selects favourites from a curated pool of recognisable, high-grossing films across multiple genres, improving relevance and engagement.

- Cosine similarity captures subtler relationships between movies based on shared feature combinations, not just direct matches.

- Easily extended to larger datasets or richer text inputs (e.g. plot summaries, keywords) without structural changes.

- Movies are scored on a continuous similarity scale, allowing more refined recommendation lists than the binary predictions of the supervised model.

#### Limitations
While the unsupervised model improves flexibility and avoids labelling constraints, it has several limitations:

- The model does not adapt over time or learn from user interactions or satisfaction ratings.

- All content features (genre, director, actors) are treated equally in TF-IDF, even though some may be more influential to user preference.

- The model doesn't interpret meaning — it treats names and terms as raw text, which may miss deeper semantic connections (e.g. genre tone or film style).

- High textual similarity doesn't always imply genuine user interest — two movies may share actors or genres but differ drastically in tone or quality.

- Without ground truth labels, the model can’t be objectively evaluated using accuracy, precision, or recall.

- Movies with unique combinations of features or limited metadata may struggle to appear in recommendations due to lack of overlap with common titles.

## Model Comparison Summary
The following table summarises the capabilities and trade-offs of each approach.
| Feature                         | Rule-Based       | Supervised ML         | Unsupervised ML       |
|----------------------------------|------------------|------------------------|------------------------|
| Requires Training               | No             | Yes                 | No                 |
| Learns from User Data           | No             | Yes                 | No                 |
| Handles Complex Patterns        | No             | Yes (non-linear)    | Partial (via TF-IDF) |
| Needs Labeled Data              | No             | Yes                 | No                 |
| Scalable to Larger Datasets     | Limited       | Yes                 | Yes                |
| Interpretable Logic             | Yes            | Less Transparent    | Limited            |
| Evaluation Metrics Available    | No             | Yes (e.g. ROC-AUC) | Limited (indirect)  |
| Adapts Over Time                | No             | Not Yet             | No                 |

## Possible Future Improvements
- Include movie plot summaries or reviews in TF-IDF input for deeper semantic understanding.
- 
- Replace the fixed movie selection list with a searchable interface across all 1000 titles, allowing users to choose films they genuinely like.

- Add collaborative filtering to blend content and user preference patterns.

- Deploy the model as a web app using Flask or Streamlit for broader usability.

- Use embeddings (e.g. Sentence Transformers) to capture deeper relationships between features.

## How to Run this Project
1. Clone the repository:
```bash    
git clone https://github.com/rhyswd5/Movie-recommendation.git
        
cd movie-recommendation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run one of the scripts:
```bash
python supervised_ML_script.py
```


## Project Structure
```plaintext
├── data/
│   └── imdb_top_1000.csv
├── scripts
│   └── rule_based_script.py
│   └── supervised_ML_script.py
│   └── unsupervised_ML_script.py
├── notebooks
│   └── rule_based_script.py
│   └── supervised_ML_script.py
│   └── unsupervised_ML_script.py
├── requirements.txt
├── README.md
```
