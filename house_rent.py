# author: Ruturajsinh Solanki

# importing necessary libraries and modules
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# importing data file for our model
path = r'D:\iNeuron ML Challenge\Bundle_2\houseRent\housing_train.csv'
data = pd.read_csv(path)

y = data.price  # target column

# features used for prediction
features = ['sqfeet', 'beds', 'baths', 'comes_furnished']
X = data[features]  # features/ columns used for prediction

# creating our model using RandomForestRegressor()
rent_model = RandomForestRegressor(random_state=1)  # defining our model
rent_model.fit(X, y)  # making model fit for prediction
preds = rent_model.predict(X)  # predicting the values

# comparing predicted values and original values
print("Predictions are: ", rent_model.predict(X.head()))
print("Original values are: ", y.head().tolist())
