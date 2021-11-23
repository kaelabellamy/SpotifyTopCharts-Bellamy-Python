#%%
import pandas as pd
import numpy as np
import datetime as dt
import scipy.stats
from pandas._libs.missing import NA
from plotnine import *

#%%
#import data
spot = pd.read_csv("C:/Users/kaela/Desktop/datamining/dataminingproject/spotify_dataset.csv") 
    #sep='\t', header=0, index_col=0)
spot.head()

# %%
#look at data and do some light cleaning
spot_c = spot.filter(['Highest Charting Position', 'Week of Highest Charting', 
        'Number of Times Charted', 
        'Release Date', 'Song Name', 'Popularity', 
         'Streams', 'Artist', 'Artist Followers', 'Song ID',
         'Genre', 'Danceability', 'Energy', 'Loudness',
        'Acousticness', 'Tempo',
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
#%%
#check out data set values
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
spot_c['Popularity'] = pd.to_numeric(spot_c['Popularity'],errors = 'coerce')

#scipy.stats.pearsonr(spot_c['Popularity'],spot_c['Artist_Followers'])

# %%
