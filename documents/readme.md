# Goal: Predict if the song will make the top 20 charts in Spotify or not. Determine which model is the best predictor and why.

BACKGROUND
Data pulled from Kaggle but originally from Spotipy Python Library.
Link: https://www.kaggle.com/sashankpillai/spotify-top-200-charts-20202021

DATA METRICS
-1,556 records and 27 feature variables

KEY FEATURES
Highest Charting Position
Number of Times Charted
Week of Highest Charting
Song Name
Artist
Release Date
Popularity
Weeks Charted
Tempo

VARIABLES CREATED
-Split Week of Highest Charting into Beginning and End of the Week
-Find the days it took from release date to end of the week that it reached the highest position
-Create binomial variable for Highest Position to be tested
-Convert population and valence to multivariate variable


RESULTS
Model, Accuracy, AUC, F1 Score
Linear Regression, 0.851948, 0.5, 0.92200561
Random Forest, 0.880519, 0.610986, 0.9340974
Decision Tree, 0.825974, 0.66022, 0.8977099
MLP Classifier, 0.864935, 0.5263, 0.9200561

MODEL CHOSEN
-random forest based off F1 score
-used F1 score because binary classification and more weight to values

