#%%
import pandas as pd
import numpy as np
import datetime as dt

from plotnine import *

#%%
#import data
spot = pd.read_csv("C:/Users/kaela/Desktop/datamining/dataminingproject/spotify_dataset.csv") 
    #sep='\t', header=0, index_col=0)
spot.head()

# %%
spot_c = spot.filter(['Highest Charting Position', 'Week of Highest Charting', 
        'Number of Times Charted', 
        'Release Date', 'Song Name',  
         'Streams', 'Artist', 'Artist Followers', 'Song ID',
         'Genre', 'Danceability', 'Energy', 'Loudness',
        'Acousticness', 'Liveness', 'Tempo',
         'Duration', 'Valence', 'Chord'])
spot_c.columns = spot_c.columns.str.replace(' ','_')
spot_c = spot_c.dropna()
#%%
spot_c=spot_c.fillna('')
spot_c=spot_c.replace(' ', '')
#%%
#check out data set values
spot_c.Release_Date.value_counts()
#%%
#check out data set values
spot_c.Week_of_Highest_Charting.value_counts()
#%%
#split Week of Highest Charting into two new columns 
spot_c[['BeginHighWeek', 'EndHighWeek']] = spot_c['Week_of_Highest_Charting'].str.split('--', expand=True)
#%%
#change dates to integers
spot_c['BeginHighWeek'] = spot_c['BeginHighWeek'].str.replace('-', '')
spot_c['BeginHighWeek'] = pd.to_numeric(spot_c['BeginHighWeek'])
spot_c['BeginHighWeek'].dtype
#%%
spot_c['Release_Date'] = spot_c['Release_Date'].str.replace('-', '')
spot_c['Release_Date'] = pd.to_numeric(spot_c['Release_Date'])
spot_c['Release_Date'].dtype
# %%
spot_c['DaystoTop'] = spot_c['BeginHighWeek']-spot_c['Release_Date']
# %%
(ggplot(spot_c, aes(x='Highest_Charting_Position', y = 
'DaystoTop')) +
    geom_boxplot())
# %%
# %%
