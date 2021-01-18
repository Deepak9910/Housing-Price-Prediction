import numpy as np
from joblib import dump, load

model_1=load("LinearRegression_Model_For_Housing_Rate_Prediction.joblib")
model_2=load("DecisionTreeRegression_Model_For_Housing_Rate_Prediction.joblib")
model_3=load("RandomForestRegression_Model_For_Housing_Rate_Prediction.joblib")

features = np.array([[-0.43942006, 3.12628155, -1.12165014, -0.27288841, -1.42262747, -0.24141041,
 -1.31238772,  2.61111401, -1.0016859,  -0.5778192,  -0.97491834,  0.41164221,
 -0.86091034]])

print("Linear Regression Model Predicts : ",model_1.predict(features))
print("Decision Tree Regression Model Predicts : ",model_2.predict(features))
print("Random Forest Regression Model Predicts : ",model_3.predict(features))