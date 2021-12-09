#%%
#pip install scipy
#pip install -U scikit-learn

import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas._libs.missing import NA
from plotnine import *
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree
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
spot_c['Tempo'] = pd.to_numeric(spot_c['Tempo'],errors = 'coerce')
#divide popularity into multivariate
#spot_c['Pop'] = 0 + (spot_c['Popularity'] > 20) + (spot_c['Popularity'] > 40) + (spot_c['Popularity'] > 60) 
# + (spot_c['Popularity'] > 80)


#%%
### VARIABLE 6 
#create valence multivariate
spot_c['Valence'] = pd.to_numeric(spot_c.Valence, errors='coerce')
spot_c['Valence1'] = 0 + (spot_c['Valence'] > 0.25) + (spot_c['Valence'] > 0.5) + (spot_c['Valence'] > 0.75)

#%%
#look at data features
valencevshighest = (ggplot(data = spot_c, mapping = aes(x='Valence', y='HighestPos'))
)
valencevshighest + geom_jitter()
#%%
tempovshighest = (ggplot(data = spot_c, mapping = aes(x='Tempo', y='HighestPos'))
)
tempovshighest + geom_jitter()

# %%
#clean up data for models
spotfinal = spot_c.drop(['BeginHighWeek', 'Week_of_Highest_Charting'], axis=1, inplace=True)

# %%
#model version 1 - binomial regression model
#group by, get counts, get total number for each column 
#DIFFERENT SPLIT FOR VALENCE
g1 = spot_c.groupby(['Tempo','DaystoTop','Valence1'])
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
g1_group['Tempo'] = g1_groupcount1['Tempo']
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
#Linear Regression Model
logreg = LogisticRegression(solver = 'saga', random_state = 67, max_iter=10000)
logreg.fit(X_train,y_train)
y_pred =  logreg.predict(X_test)
errors = abs(y_pred - y_test)

print("Linear Regression accuracy is:", metrics.accuracy_score(y_test, y_pred))

print("Linear Regression f1 score is:", f1_score(y_test, y_pred))
#%%
clf.fit(X_train, y_train)
SVC(random_state=0)
plot_confusion_matrix(clf, X_test, y_test)  
plt.show()
#%%
plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, y_pred)
auc = metrics.roc_auc_score(y_test, y_pred)
plt.plot(fpr,tpr,label="Linear Regression, auc="+str(auc))

plt.legend(loc=0)
#%%
#Random Forest Model
frst = RandomForestClassifier(n_estimators = 1000, criterion = 'gini', min_samples_leaf = 5, max_depth= 2, random_state = 67)

frst.fit(X_train, y_train)

test_frst = frst.predict(X_test)

print('Forest accuracy is: ', np.mean(test_frst==y_test))
print("Forest Accuracy f1 score is:", f1_score(y_test, test_frst))
#%%
clf.fit(X_train, y_train)
SVC(random_state=0)
plot_confusion_matrix(clf, X_test, y_test)  
plt.show()
#%%
plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, test_frst)
auc = metrics.roc_auc_score(y_test, test_frst)
plt.plot(fpr,tpr,label="Random Forest, auc="+str(auc))

plt.legend(loc=0)
#%%
#Decision Tree Model
clf = tree.DecisionTreeClassifier()
dtclf = clf.fit(X_train, y_train)
preds = clf.predict(X_test)

print('Tree accuracy is: ', np.mean(preds==y_test))
print("Tree f1 score is:", f1_score(y_test, preds))

#%%
plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, preds)
auc = metrics.roc_auc_score(y_test, preds)
plt.plot(fpr,tpr,label="Decision Tree, auc="+str(auc))

plt.legend(loc=0)

#%%
#MLP Classifier Model
MLP = MLPClassifier

data = MLPClassifier(solver='adam', learning_rate_init=0.003, batch_size=24, hidden_layer_sizes=(4, 3, 2), max_iter=1000 )

data.fit(X_train, y_train)
test = data.predict(X_test)

print('MLP accuracy is: ', np.mean(test==y_test))
print("MLP accuracy f1 score is:", f1_score(y_test, test))

# %%
plt.figure(0).clf()

fpr, tpr, thresh = metrics.roc_curve(y_test, test)
auc = metrics.roc_auc_score(y_test, test)
plt.plot(fpr,tpr,label="MLP Classifier, auc="+str(auc))

plt.legend(loc=0)
# %%
