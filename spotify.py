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




# %%
#clean up data for models
spotfinal = spot_c.drop(['BeginHighWeek', 'Popularity', 'Week_of_Highest_Charting'], axis=1, inplace=True)




# %%
#model version 1 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT VARIABLE USED IN THIS VERSION IS VALENCE
g1 = spot_c.groupby(['Pop','DaystoTop','Valence'])
g1_groupcount = g1.count()
g1_groupsum = g1.sum()
# %%
# merge sums and counts into data frame (and save for easier use)
g1_groupcount.to_csv('g1_groupcount.csv')
g1_groupcount1 = pd.read_csv('g1_groupcount.csv', header=0)
 
#g1_groupsum.to_csv('g1_groupsum.csv')
#g1_groupsum1 = pd.read_csv('g1_groupsum.csv', header=0)

# %%
#create new data frame
g1_group = pd.DataFrame()
#%%
#add in columns for data frame
g1_group['Pop'] = g1_groupcount1['Pop']
g1_group['DaystoTop'] = g1_groupcount1['DaystoTop']
g1_group['DaystoTop'] = g1_group['DaystoTop'].str.replace('days', '')
g1_group['DaystoTop'] = g1_group['DaystoTop'].astype(float)
#g1_group['Valence'] = g1_groupcount1['Valence']
# %%
#add in more columns for data frame
g1_group['HighestPos'] = g1_groupcount1['HighestPos']
#g1_group['HighestPos'] = g1_groupsum1['HighestPos']
#g1_group['Pos'] = g1_group['Total'] - g1_group['HighestPos']

# %%
X = g1_group # Features
y = g1_group['HighestPos'] # Target variable
# %%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
# %%
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=5000)
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
errors = abs(y_pred - y_test)
# %%
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#print("Precision:",  metrics.precision_score(y_test, y_pred))
#print("Recall:",metrics.recall_score(y_test, y_pred))
# %%
