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


Method 1: [0,0.5], [0.5,1.0]
0.70987
Method 2: [0,0.33], [0.33,0.67], [0.67,1.0]
0.75739
Method 3: [0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]
0.81021
Method 4: [0,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8], [0.8,1.0]
0.73026
Method 5: [0,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8], 
[0.8,0.9], [0.9,1.0]
0.70987