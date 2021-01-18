import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

housing = pd.read_csv("dataset.csv")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(housing, housing['CHAS']):
    train_set = housing.loc[train_index]
    test_set = housing.loc[test_index]

housing = train_set.copy()

housing = train_set.drop("MEDV",axis=1)
housing_labels = train_set["MEDV"].copy()

my_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("std_scaler",StandardScaler())
    ])
housing_num = my_pipeline.fit_transform(housing)

model = RandomForestRegressor()
model.fit(housing_num,housing_labels) 

scores = cross_val_score(model, housing_num, housing_labels, scoring = "neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores : ",scores)
    print("Mean : ", scores.mean())
    print("Standard deviation : ",scores.std())
    
print_scores(rmse_scores)

from joblib import dump, load
dump(model, "RandomForestRegression_Model_For_Housing_Rate_Prediction.joblib")

# Testing The Test Set

x_test = test_set.drop("MEDV",axis = 1);
y_test = test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print("Final Error of Test Set: ",final_rmse)