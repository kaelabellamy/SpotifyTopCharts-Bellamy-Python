# dataminingproject

You must “engineer” (create) 5 new data columns/fields that are used in your models. Provide rationale as to the reason that you chose the creation of these columns.
- Column 1: BeginHighWeek
- Column 2: End High Week
- Column 3: # of days it took to get to highest charting position aks DaystoTop
- Column 4: binomial for highest position aka HighestPos
- Column 5: popularity by category aka Pop


# Binomal Regression
https://timeseriesreasoning.com/contents/binomial-regression-model/
https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python

Create multiple models using different variables: DaystoTop, HighestPos, and ... 

Decide best model.

For this project select a dataset that includes a binary, nominal class or ordinal class value that you wish to predict (or a continuous value data element that could be used to create a label to predict) and create a set of “classifiers” (prediction models) that predict the categorical output. 

Determine which methodology/model is the “winner” and provide your reasoning (“I like this method because it’s really cool” is a valid answer, but for full credit I’m looking for something that’s more substantive). 


features = g1_group.shape[1]
g1_group[features] = 0

m = min(Z)
M = max(Z)
y = (Z-m)/(M-m)
#%%
T, X = g1_group
t = g1_group.HighestPos()
t[features] = 0
test = t.iloc[:,0:(features+1)]

m = min(X)
M = max(X)
y_test = (X-m)/(M-m)
#%%
print(m, M)
print(y[0:10])
print(Z[0:10])
print(g1_group.shape, test.shape, features)
print(np.unique(y, return_counts=True))