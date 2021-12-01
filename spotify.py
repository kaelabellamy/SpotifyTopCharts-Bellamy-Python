#%%
#pip install scipy
#pip install -U scikit-learn

import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats
from pandas._libs.missing import NA
from plotnine import *
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

#%%
#import data
spot = pd.read_csv("C:/Users/kaela/Desktop/datamining/dataminingproject/spotify_dataset.csv") 
spot.head()

# %%
#look at data and do some light cleaning
spot_c = spot.filter(['Highest Charting Position', 'Week of Highest Charting', 
        'Number of Times Charted', 
        'Release Date', 'Song Name', 'Popularity', 
         'Song ID', 'Genre', 'Acousticness', 'Tempo',
         'Duration', 'Valence', 'Chord'])
spot_c.columns = spot_c.columns.str.replace(' ','_')
spot_c = spot_c.dropna()
#%%
spot_c=spot_c.fillna('')
spot_c=spot_c.replace(' ', '')


### GOAL: TO PREDICT BINARY VALUE OF TOP CHARTS: WILL IT BE IN TOP 20 OR NOT? 
# 0 = IT IS, 1 = IT IS NOT



#%%
###TRYING TO CREATE NUMBER OF DAYS TO TOP AFTER RELEASE DATE - VARIABLE 1,2, and 3
#check out data set values
spot_c.Release_Date.value_counts()
spot_c.Week_of_Highest_Charting.value_counts()
#%%
#split Week of Highest Charting into two new columns, creates two new variables
spot_c[['BeginHighWeek', 'EndHighWeek']] = spot_c['Week_of_Highest_Charting'].str.split('--', expand=True)
#%%
#convert dates to be recognized as dates
spot_c['EndHighWeek'] = pd.to_datetime(spot_c['EndHighWeek'], yearfirst=True)
spot_c['BeginHighWeek'] = pd.to_datetime(spot_c['BeginHighWeek'], yearfirst=True)
spot_c['Release_Date'] = pd.to_datetime(spot_c['Release_Date'], yearfirst=True)
# %%
#find the # of days it took to get to the top for each song aka new column
spot_c['DaystoTop'] = abs(spot_c['EndHighWeek']-spot_c['Release_Date'])


# %%
### VARIABLE 4
#create binomial value for top charts 0 if <= 20 1 if >20 
spot_c['HighestPos'] = 0 + (spot_c['Highest_Charting_Position'] > 20.0) 


# %%
### VARIABLE 5
#change pop variable to float from string
spot_c['Popularity'] = pd.to_numeric(spot_c['Popularity'],errors = 'coerce')
#divide popularity into multivariate
spot_c['Pop'] = 0 + (spot_c['Popularity'] > 20) + (spot_c['Popularity'] > 40) + (spot_c['Popularity'] > 60) + (spot_c['Popularity'] > 80)

#%%
### VARIABLE 6 
#create valence multivariate
spot_c['Valence'] = pd.to_numeric(spot_c.Valence, errors='coerce')
spot_c['Valence1'] = 0 + (spot_c['Valence'] > 0.5) 
spot_c['Valence2'] = 0 + (spot_c['Valence'] > 0.33) + (spot_c['Valence'] > 0.67)
spot_c['Valence3'] = 0 + (spot_c['Valence'] > 0.25) + (spot_c['Valence'] > 0.5) + (spot_c['Valence'] > 0.75)
spot_c['Valence4'] = 0 + (spot_c['Valence'] > 0.2) + (spot_c['Valence'] > 0.4) + (spot_c['Valence'] > 0.6) + (spot_c['Valence'] > 0.8)
spot_c['Valence5'] = 0 + (spot_c['Valence'] > 0.1) + (spot_c['Valence'] > 0.2) + (spot_c['Valence'] > 0.3) + (spot_c['Valence'] > 0.4) + (spot_c['Valence'] > 0.5) + (spot_c['Valence'] > 0.6) + (spot_c['Valence'] > 0.7) + (spot_c['Valence'] > 0.8) + (spot_c['Valence'] > 0.9)

# %%
#clean up data for models
spotfinal = spot_c.drop(['BeginHighWeek', 'Popularity', 'Week_of_Highest_Charting'], axis=1, inplace=True)

# %%
#model version 1 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g1 = spot_c.groupby(['Pop','DaystoTop','Valence1'])
g1_groupcount = g1.count()
g1_groupsum = g1.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g1_groupcount.to_csv('g1_groupcount.csv')
g1_groupcount1 = pd.read_csv('g1_groupcount.csv', header=0)

# %%
#create new data frame
g1_group = pd.DataFrame()
#%%
#add in columns for data frame
g1_group['Pop'] = g1_groupcount1['Pop']
g1_group['DaystoTop'] = g1_groupcount1['DaystoTop']
g1_group['DaystoTop'] = g1_group['DaystoTop'].str.replace('days', '')
g1_group['DaystoTop'] = g1_group['DaystoTop'].astype(float)
g1_group['DaystoTop'] = 0 + (g1_group['DaystoTop'] > 50.0) + (g1_group['DaystoTop'] > 100.0) + (g1_group['DaystoTop'] > 200.0)+ (g1_group['DaystoTop'] > 500.0) + (g1_group['DaystoTop'] > 1000.0)+ (g1_group['DaystoTop'] > 1500.0)

# %%
#add in more columns for data frame
g1_group['HighestPos'] = spot_c['HighestPos']

# %%
X = g1_group.drop('HighestPos', axis=1) # Features
y = g1_group['HighestPos'] # Target variable
#%%
#Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))





# %%
#model version 2 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g2 = spot_c.groupby(['Pop','DaystoTop','Valence2'])
g2_groupcount = g2.count()
g2_groupsum = g2.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g2_groupcount.to_csv('g2_groupcount.csv')
g2_groupcount1 = pd.read_csv('g2_groupcount.csv', header=0)

# %%
#create new data frame
g2_group = pd.DataFrame()
#%%
#add in columns for data frame
g2_group['Pop'] = g2_groupcount1['Pop']
g2_group['DaystoTop'] = g2_groupcount1['DaystoTop']
g2_group['DaystoTop'] = g2_group['DaystoTop'].str.replace('days', '')
g2_group['DaystoTop'] = g2_group['DaystoTop'].astype(float)
g2_group['DaystoTop'] = 0 + (g2_group['DaystoTop'] > 50.0) + (g2_group['DaystoTop'] > 100.0) + (g2_group['DaystoTop'] > 200.0)+ (g2_group['DaystoTop'] > 500.0)+ (g2_group['DaystoTop'] > 1000.0)+ (g2_group['DaystoTop'] > 1500.0)

# %%
#add in more columns for data frame
g2_group['HighestPos'] = spot_c['HighestPos']

# %%
X = g2_group.drop('HighestPos', axis=1) # Features
y = g2_group['HighestPos'] # Target variable
#%%
#Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))




# %%
#model version 3 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g3 = spot_c.groupby(['Pop','DaystoTop','Valence3'])
g3_groupcount = g3.count()
g3_groupsum = g3.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g3_groupcount.to_csv('g3_groupcount.csv')
g3_groupcount1 = pd.read_csv('g3_groupcount.csv', header=0)

# %%
#create new data frame
g3_group = pd.DataFrame()
#%%
#add in columns for data frame
g3_group['Pop'] = g3_groupcount1['Pop']
g3_group['DaystoTop'] = g3_groupcount1['DaystoTop']
g3_group['DaystoTop'] = g3_group['DaystoTop'].str.replace('days', '')
g3_group['DaystoTop'] = g3_group['DaystoTop'].astype(float)
g3_group['DaystoTop'] = 0 + (g3_group['DaystoTop'] > 50.0) + (g3_group['DaystoTop'] > 100.0) + (g3_group['DaystoTop'] > 200.0)+ (g3_group['DaystoTop'] > 500.0)+ (g3_group['DaystoTop'] > 1000.0)+ (g3_group['DaystoTop'] > 1500.0)

# %%
#add in more columns for data frame
g3_group['HighestPos'] = spot_c['HighestPos']

# %%
X = g3_group.drop('HighestPos', axis=1) # Features
y = g3_group['HighestPos'] # Target variable
#%%
#Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))






# %%
#model version 4 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g4 = spot_c.groupby(['Pop','DaystoTop','Valence4'])
g4_groupcount = g4.count()
g4_groupsum = g4.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g4_groupcount.to_csv('g4_groupcount.csv')
g4_groupcount1 = pd.read_csv('g4_groupcount.csv', header=0)

# %%
#create new data frame
g4_group = pd.DataFrame()
#%%
#add in columns for data frame
g4_group['Pop'] = g4_groupcount1['Pop']
g4_group['DaystoTop'] = g4_groupcount1['DaystoTop']
g4_group['DaystoTop'] = g4_group['DaystoTop'].str.replace('days', '')
g4_group['DaystoTop'] = g4_group['DaystoTop'].astype(float)
g4_group['DaystoTop'] = 0 + (g4_group['DaystoTop'] > 50.0) + (g4_group['DaystoTop'] > 100.0) + (g4_group['DaystoTop'] > 200.0)+ (g4_group['DaystoTop'] > 500.0)+ (g4_group['DaystoTop'] > 1000.0)+ (g4_group['DaystoTop'] > 1500.0)

# %%
#add in more columns for data frame
g4_group['HighestPos'] = spot_c['HighestPos']

# %%
X = g4_group.drop('HighestPos', axis=1) # Features
y = g4_group['HighestPos'] # Target variable
#%%
#Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))



# %%
#model version 5 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g5 = spot_c.groupby(['Pop','DaystoTop','Valence5'])
g5_groupcount = g5.count()
g5_groupsum = g5.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g5_groupcount.to_csv('g5_groupcount.csv')
g5_groupcount1 = pd.read_csv('g5_groupcount.csv', header=0)

# %%
#create new data frame
g5_group = pd.DataFrame()
#%%
#add in columns for data frame
g5_group['Pop'] = g5_groupcount1['Pop']
g5_group['DaystoTop'] = g5_groupcount1['DaystoTop']
g5_group['DaystoTop'] = g5_group['DaystoTop'].str.replace('days', '')
g5_group['DaystoTop'] = g5_group['DaystoTop'].astype(float)
g5_group['DaystoTop'] = 0 + (g5_group['DaystoTop'] > 50.0) + (g5_group['DaystoTop'] > 100.0) + (g5_group['DaystoTop'] > 200.0)+ (g5_group['DaystoTop'] > 500.0)+ (g5_group['DaystoTop'] > 1000.0)+ (g5_group['DaystoTop'] > 1500.0)

# %%
#add in more columns for data frame
g5_group['HighestPos'] = spot_c['HighestPos']

# %%
X = g5_group.drop('HighestPos', axis=1) # Features
y = g5_group['HighestPos'] # Target variable
#%%
#Split data into training and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# %%
